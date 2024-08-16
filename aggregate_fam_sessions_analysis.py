import argparse
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

from behviour_analysis_funcs import get_all_cond_filts, get_sess_name_date_idx
from ephys_analysis_funcs import posix_from_win, load_sess_pkl, get_predictor_from_psth, Decoder
from population_analysis_funcs import compute_eig_vals, compute_trial_averaged_pca, project_pca, plot_pca_ts
from regression_funcs import run_glm


class PopPCA:

    def __init__(self, responses_by_cond: dict):
        self.eig_vals = None
        self.projected_pca_ts_by_cond = None
        self.pca_ts_plot = None
        self.Xa_trial_averaged_pca = None
        assert isinstance(list(responses_by_cond.values())[0], dict)
        self.responses_by_cond = responses_by_cond
        self.conds = list(responses_by_cond.keys())
        self.events = list(responses_by_cond[self.conds[0]].keys())
        self.event_concatenated_responses = self.get_event_concatenated_responses()
        self.get_eig_vals()

    def get_event_concatenated_responses(self):
        event_concatenated_responses = np.hstack(
            [np.hstack(
                [e_responses for e_name, e_responses in cond_responses.items()])
                for cond_responses in self.responses_by_cond.values()])
        event_concatenated_responses = np.squeeze(event_concatenated_responses)
        return event_concatenated_responses

    def get_eig_vals(self):
        self.eig_vals = compute_eig_vals(self.event_concatenated_responses, plot_flag=True)
        self.eig_vals[2][1].set_xlim(0, 30)
        self.eig_vals[2][1].set_ylabel('PC component')
        self.eig_vals[2][1].set_xlabel('Proportion of variance explained')
        # self.eig_vals[2][0].show()

    def get_trial_averaged_pca(self, n_components=15, standardise=False):
        self.Xa_trial_averaged_pca = compute_trial_averaged_pca(self.event_concatenated_responses,
                                                                n_components=n_components, standardise=standardise)

    def get_projected_pca_ts(self, standardise=False):
        self.projected_pca_ts_by_cond = {cond: {event_name: project_pca(event_response, self.Xa_trial_averaged_pca,
                                                                        standardise=standardise)
                                                for event_name, event_response in cond_responses.items()}
                                         for cond, cond_responses in self.responses_by_cond.items()}

    def plot_pca_ts(self, event_window, n_comp_toplot=5, plot_separately=False, fig_kwargs=None,
                    conds2plot=None,**kwargs):
        if conds2plot is None:
            conds2plot = self.conds
        if kwargs.get('events2plot',None) is None:
            events2plot = {cond: list(self.projected_pca_ts_by_cond[cond].keys()) for cond in conds2plot}
        else:
            events2plot = kwargs.get('events2plot')
        if kwargs.get('plot',None) is None:
            self.pca_ts_plot = plt.subplots(len(self.events) if plot_separately else 1, n_comp_toplot, squeeze=False,
                                            **(fig_kwargs if fig_kwargs is not None else {}))
        else:
            self.pca_ts_plot = kwargs.get('plot')

        axes = self.pca_ts_plot[1] if plot_separately else [self.pca_ts_plot[1][0]] * len(self.events)
        lss = kwargs.get('lss', ['-', '--', ':', '-.'])
        plt_cols = kwargs.get('plt_cols', ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'])
        [[plot_pca_ts([projected_responses], [f'{cond} {event}'], event_window, n_components=n_comp_toplot,
                     plot=[self.pca_ts_plot[0], axes[ei]], plot_kwargs={'ls': lss[cond_i], 'c': plt_cols[cond_i],
                                                                        'label': f'{cond} {event}'})
         for ei, (event, projected_responses) in enumerate(zip(events2plot[cond], [self.projected_pca_ts_by_cond[cond][e] for e in events2plot[cond]]))]
         for cond_i, cond in enumerate(conds2plot)]
        [row_axes[0].set_ylabel('PC component') for row_axes in self.pca_ts_plot[1]]
        [row_axes[0].legend(loc='upper center',ncol=4) for row_axes in self.pca_ts_plot[1].T]
        # [ax.legend() for ax in self.pca_ts_plot[1]]
        [ax.set_xlabel('Time from stimulus onset (s)') for ax in self.pca_ts_plot[1][-1]]
        self.pca_ts_plot[0].show()


def load_aggregate_td_df(session_topolgy: pd.DataFrame,home_dir:Path,td_df_query=None) -> pd.DataFrame:
    # get main sess pattern
    td_path_pattern = 'data/Dammy/<name>/TrialData'
    td_paths = [Path(sess_info['sound_bin'].replace('_SoundData', '_TrialData')).with_suffix('.csv').name
                for _,sess_info in session_topolgy.iterrows()]
    sessnames = [Path(sess_info['sound_bin'].replace('_SoundData', '')).stem
                 for _,sess_info in session_topolgy.iterrows()]
    # [get_sess_name_date_idx(sessname,session_topolgy) for sessname in sessnames]
    abs_td_paths = [home_dir / td_path_pattern.replace('<name>', sess_info['name'])/td_path
                    for (td_path,(_,sess_info)) in zip(td_paths,session_topolgy.iterrows())]

    td_df = pd.concat([pd.read_csv(abs_td_path) for abs_td_path in abs_td_paths if abs_td_path.is_file()],
                      axis=0,keys=sessnames,names=['sess'])
    if td_df_query is not None:
        td_df = td_df.query(td_df_query)
    return td_df


def load_aggregate_sessions(pkl_paths, td_df_query=None):
    sess_objs = []
    for pkl_path in tqdm(pkl_paths, total=len(pkl_paths), desc='loading sessions'):
        print(f'loading {pkl_path}')
        sess_obj = load_sess_pkl(pkl_path)
        if sess_obj is None:
            print(f'{pkl_path} error')
            continue
        if sess_obj.td_df is None:
            continue
        if td_df_query is not None:
            if sess_obj.td_df.query(td_df_query).empty:
                continue
        sess_objs.append(sess_obj)
    return {e.sessname: e for e in sess_objs}


def aggregate_event_reponses(sessions: dict, events=None, events2exclude=None, window=(0, 0.25),
                             pred_from_psth_kwargs=None, ):
    common_events = [list(sess.sound_event_dict.keys()) for sess in sessions.values()]
    common_events = np.unique(common_events)
    # print(f'{"A-0" in common_events = } ')
    if events:
        sessions2use = [sess for sess, s_events in zip(sessions,common_events) if all(e in s_events for e in events)]
    else:
        sessions2use = sessions
    if events2exclude:
        common_events = [e for e in common_events if e not in events2exclude]
    if events:
        assert all([e in common_events for e in events]), f'{events} not in {common_events}'
        common_events = events

    event_responses_across_sessions = {sessname: {e: get_predictor_from_psth(sessions[sessname], e, [-2, 3], window,
                                                                             **pred_from_psth_kwargs if pred_from_psth_kwargs else {})
                                                  for e in common_events}
                                       for sessname in
                                       tqdm(sessions2use, total=len(sessions2use), desc='Getting event responses')}
    return event_responses_across_sessions


def aggregate_event_features(sessions: dict, events=None, events2exclude=None):
    common_events = [list(sess.sound_event_dict.keys()) for sess in sessions.values()]
    if events:
        sessions2use = [sess for sess, s_events in zip(sessions, common_events) if all(e in s_events for e in events)]
    else:
        sessions2use = sessions

    common_events = np.unique(common_events)
    if events2exclude:
        common_events = [e for e in common_events if e not in events2exclude]
    if events:
        assert all([e in common_events for e in events]), f'{events} not in {common_events}'
        common_events = events

    event_features_across_sessions = {sessname: {e: {'times': sessions[sessname].sound_event_dict[e].times.values,
                                                     'trial_nums': sessions[sessname].sound_event_dict[e].trial_nums, }
                                                 for e in common_events}
                                      for sessname in
                                      tqdm(sessions2use, total=len(sessions2use), desc='Getting event features')}
    for sessname in tqdm(event_features_across_sessions, total=len(event_features_across_sessions), desc='Adding td_df'):
        sess_td_df = sessions[sessname].td_df
        for e in event_features_across_sessions[sessname]:
            # event_features_across_sessions[sessname][e]['td_df'] = sess_td_df.reset_index(drop=True).loc[
            #     event_features_across_sessions[sessname][e]['trial_nums'] - 1]
            trial_nums = event_features_across_sessions[sessname][e]['trial_nums']-1
            event_features_across_sessions[sessname][e]['td_df'] = sess_td_df.reset_index(drop=True).query('index in @trial_nums')
    return event_features_across_sessions


def concatenate_responses_by_td(responses_by_sess: dict, features_by_sess: dict, td_df_query=None):
    grouped_by_td = {}
    common_events = set.intersection(*[set(list(e.keys())) for e in responses_by_sess.values()])
    for event in tqdm(common_events, total=len(common_events), desc='Concatenating responses'):
        event_across_sess = [sess_responses[event][:len(sess_features[event]['td_df'])] for sess_responses,sess_features
                             in zip(responses_by_sess.values(), features_by_sess.values())]
        event_idxbool = [sess[event]['td_df'].eval(td_df_query).values for sess in features_by_sess.values()]
        try:
            grouped_by_td[event] = np.concatenate([sess_e_responses[sess_idx_bool].mean(axis=0)
                                                   for sess_e_responses, sess_idx_bool in
                                                   tqdm(zip(event_across_sess, event_idxbool), total=len(event_across_sess),
                                                        desc='Concatenating features')], axis=0)
        except IndexError:
            print(f'IndexError for {event}')
            grouped_by_td[event] = None

    return grouped_by_td


def group_responses_by_pip_prop(responses_by_sess: dict, pip_desc:dict, properties=None,concatenate_flag=True):
    pip_groups_by_prop = {prop: {val: [pip for pip in pip_desc[prop].keys() if pip_desc[prop][pip] == val]
                                 for val in np.unique(list(pip_desc[prop].values()))} for prop in pip_desc}
    if not properties:
        properties = list(pip_groups_by_prop.keys())
    if concatenate_flag:
        responses_by_prop = {prop: {val: np.concatenate([responses_by_sess[pip] for pip in pip_groups_by_prop[prop][val]
                                                         if pip in responses_by_sess],
                                                        axis=0)
                                    for val in pip_groups_by_prop[prop]} for prop in
                             tqdm(properties, total=len(pip_groups_by_prop), desc='Concatenating responses')}
    else:
        responses_by_prop = {prop: {val: {pip:responses_by_sess[pip] for pip in pip_groups_by_prop[prop][val]
                                          if pip in responses_by_sess}
                                    for val in pip_groups_by_prop[prop]} for prop in
                             tqdm(properties, total=len(pip_groups_by_prop), desc='Concatenating responses')}

    return responses_by_prop


def decode_responses(predictors, features, model_name='logistic', n_runs=100, dec_kwargs=None):
    decoder = {dec_lbl: Decoder(np.vstack(predictors), np.hstack(features), model_name=model_name)
               for dec_lbl in ['data', 'shuffled']}
    dec_kwargs = dec_kwargs or {}
    [decoder[dec_lbl].decode(dec_kwargs=dec_kwargs | {'shuffle': shuffle_flag}, parallel_flag=False,
                             n_runs=n_runs)
     for dec_lbl, shuffle_flag in zip(decoder, [False, True])]

    return decoder


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('pkldir')
    args = parser.parse_args()
    sys_os = platform.system().lower()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    home_dir = Path(config[f'home_dir_{sys_os}'])
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])

    session_topology_path = ceph_dir / posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology.csv')
    session_topology = pd.read_csv(session_topology_path)

    all_sess_info = session_topology.query('sess_order=="main" & date > 240219')
    all_sess_td_df = load_aggregate_td_df(all_sess_info,home_dir,'Stage==3')
    sessions2use = all_sess_td_df.index.get_level_values('sess').unique()
    # sessnames = [Path(sess_info['sound_bin']).stem.replace('_SoundData', '') for _, sess_info in
    #              sessions2use.iterrows()]

    pkldir = ceph_dir / posix_from_win(args.pkldir)
    pkls = list(pkldir.glob('*pkl'))
    pkls2load = [pkl for pkl in pkls if Path(pkl).stem in sessions2use]
    sessions = load_aggregate_sessions(pkls2load,'Stage==3')

    # update styles
    plt.style.use('figure_stylesheet.mplstyle')
    window = (-0.1, 0.25)
    event_responses = aggregate_event_reponses(sessions, events=[f'{pip}-0' for pip in 'ABCD'],
                                               events2exclude=['trial_start'], window=window,
                                               pred_from_psth_kwargs={'use_unit_zscore': True, 'use_iti_zscore': False,
                                                                      'baseline': 0, 'mean': None, 'mean_axis': 0})

    event_features = aggregate_event_features(sessions, events=[f'{pip}-0' for pip in 'ABCD'],
                                              events2exclude=['trial_start'])
    # construct array
    concatenated_event_responses = {
        e: np.concatenate([event_responses[sessname][e].mean(axis=0) for sessname in event_responses])
        for e in list(event_responses.values())[0].keys()}
    concatenated_event_times = {
        e: np.concatenate([(event_features[sessname][e]['times']) for sessname in event_features])
        for e in list(event_features.values())[0].keys()}
    # get n_units
    n_units = concatenated_event_responses[list(concatenated_event_responses.keys())[0]].shape[0]
    # concat by rare and frequent
    cond_filts = get_all_cond_filts()
    concatenated_events_by_prate = {cond: concatenate_responses_by_td(event_responses, event_features, cond_filts[cond])
                                    for cond in ['rare', 'frequent']}
    # plot mean responses to rare and frequent pips
    rare_vs_freq_plot = plt.subplots()
    for cond_i, (cond, ls) in enumerate(zip(['rare', 'frequent'], ['-', '--'])):
        for pip_i, pip in enumerate(sorted(concatenated_events_by_prate[cond])):
            x_ser = np.linspace(*window, len(concatenated_events_by_prate[cond][pip].mean(axis=0)))
            rare_vs_freq_plot[1].plot(x_ser, concatenated_events_by_prate[cond][pip].mean(axis=0),
                                      c=f'C{pip_i}', label=f'{cond} {pip}', ls=ls)
    # rare_vs_freq_plot[1].legend()
    rare_vs_freq_plot[1].set_title(f'Population responses to rare and frequent pips')
    rare_vs_freq_plot[1].set_xlabel('Time from sound onset (s)')
    rare_vs_freq_plot[1].axvline(0, color='k', linestyle='--')
    rare_vs_freq_plot[1].set_ylabel('Firing rate')
    rare_vs_freq_plot[0].set_layout_engine('tight')
    rare_vs_freq_plot[0].show()

    # bootstrap responses
    # def bootstrap_responses(x,axis=0):
    #     return x.mean(axis=axis)
    n_resamples = 9999
    n_samples = 400 # int(n_units * 0.5)
    bootstrap_responses = {cond: {pip: np.array([concatenated_events_by_prate[cond][pip][sample_idxs]
                                                 for sample_idxs in [np.random.choice(n_units, size=n_samples,
                                                                                      replace=True)
                                                                     for _ in range(n_resamples)]])
                                  for pip in cond_responses}
                           for cond, cond_responses in concatenated_events_by_prate.items()}

    # plot bootstrap
    bootstrap_plot = plt.subplots(ncols=len('ABCD'), sharex=True, sharey=True, figsize=(24, 6))
    for cond_i, (cond, ls) in enumerate(zip(['rare', 'frequent'], ['-', '--'])):
        for pip_i, (pip, ax) in enumerate(zip(sorted(bootstrap_responses[cond]), bootstrap_plot[1])):
            x_ser = np.linspace(*window, bootstrap_responses[cond][pip].shape[-1])
            ax.plot(x_ser, bootstrap_responses[cond][pip].mean(axis=0).mean(axis=0),
                    c=f'C{pip_i}', label=f'{cond}', ls=ls)
            # plot confidence intervals
            ax.fill_between(x_ser, np.quantile(bootstrap_responses[cond][pip].mean(axis=1), 0.025, axis=0),
                            np.quantile(bootstrap_responses[cond][pip].mean(axis=1), 0.975, axis=0),
                            color=f'C{pip_i}', alpha=0.05)
            ax.legend()
            ax.set_title(pip)
            ax.set_xlabel('Time from sound onset (s)')
            ax.axvline(0, color='k', linestyle='--')

    bootstrap_plot[1][0].set_ylabel('Firing rate')
    bootstrap_plot[0].suptitle('Bootstrap responses to rare and frequent pips')
    bootstrap_plot[0].set_layout_engine('tight')
    bootstrap_plot[0].show()

    # glm by unit
    do_glm = False
    if do_glm:
        responses = np.vstack([concatenated_event_responses[e] for e in concatenated_event_responses])
        labels = np.hstack([np.repeat(e, len(concatenated_event_responses[e])) for e in concatenated_event_responses])

        glm_by_units = [run_glm(unit_response, np.arange(len(np.unique(labels)))) for unit_response in
                        # np.array(list(concatenated_event_responses.values())).mean(axis=2).T]
                        np.array(list(concatenated_event_responses.values()))[:, :, -1].T]
        pvals_by_units = pd.DataFrame([unit_glm[1].pvalues for unit_glm in glm_by_units], columns=['pip', 'const'])
        betas_by_units = pd.DataFrame([unit_glm[1].params for unit_glm in glm_by_units])
        glm_pval_plot = plt.subplots(figsize=(12, 6))
        min_pval_to_display = 0.001
        [glm_pval_plot[1].scatter(pvals_by_units.index, pvals_by_units[col], c=f'C{ci}', label=col, s=15)
         for ci, col in enumerate(pvals_by_units.columns)]

        sig_thresh = 0.05
        glm_pval_plot[1].axhline(sig_thresh, ls='--', c='k')
        glm_pval_plot[1].set_xlabel('unit')
        glm_pval_plot[1].set_ylim(0.001, 1)
        glm_pval_plot[1].set_ylabel('pval')
        glm_pval_plot[1].legend(loc='lower right', ncols=len(pvals_by_units.columns))
        glm_pval_plot[0].show()

        sig_regr_by_type = {regr: (pvals_by_units.query(f'{regr} < {sig_thresh}').index)
                            for regr in ['pip']}

        sig_units = np.unique(np.hstack([sig_regr_by_type[regr].to_list()
                                         for regr in ['pip']]))

        # use coefs to project for each time point
        pip_response_proj_plot = plt.subplots()
        # glm_betas = np.random.permutation(betas_by_units[0].values)
        glm_betas = betas_by_units[0].values
        for pi, pip in enumerate('ABCD'):
            projected_w_coefs = np.vstack([concatenated_event_responses[f'{pip}-0'][:, t] * glm_betas
                                           for t in range(concatenated_event_responses[f'{pip}-0'].shape[1])]).T

            pip_response_proj_plot[1].plot(x_ser, projected_w_coefs.mean(axis=0), c=f'C{pi}', label=pip)
            # pip_response_proj_plot[1].plot(x_ser,projected_w_coefs.mean(axis=0),c=f'C{pi}',ls='--', label=pip)
        # pip_response_proj_plot[1].legend()
        pip_response_proj_plot[1].set_title(f'Population responses to pips')
        pip_response_proj_plot[1].set_xlabel('Time from sound onset (s)')
        pip_response_proj_plot[1].axvline(0, color='k', linestyle='--')
        pip_response_proj_plot[1].set_ylabel('Firing rate')
        pip_response_proj_plot[0].set_layout_engine('tight')
        pip_response_proj_plot[0].show()

    # pca
    pca_figdir = ceph_dir / 'Dammy' / 'figures' / 'pca_aggregate_sessions_all_new_fam'
    if not pca_figdir.exists():
        pca_figdir.mkdir()
    full_pattern_responses = aggregate_event_reponses(sessions, events=['A-0', 'base', 'X'],
                                                      events2exclude=['trial_start'], window=[-0.25, 1],
                                                      pred_from_psth_kwargs={'use_unit_zscore': True,
                                                                             'use_iti_zscore': False,
                                                                             'baseline': 0, 'mean': None,
                                                                             'mean_axis': 0})
    full_pattern_event_features = aggregate_event_features(sessions, events=['A-0', 'base', 'X'],
                                                           events2exclude=['trial_start'])
    concatenated_full_pattern_by_prate = {cond: concatenate_responses_by_td(full_pattern_responses,
                                                                            full_pattern_event_features,
                                                                            cond_filts[cond])
                                          for cond in ['rare', 'frequent']}

    # Xa_trial_averaged = np.hstack(list(concatenated_event_responses.values()))

    prate_pca = PopPCA(concatenated_full_pattern_by_prate)
    prate_pca.eig_vals[2][0].show()
    prate_pca.get_trial_averaged_pca()
    prate_pca.get_projected_pca_ts()
    prate_pca.plot_pca_ts([-0.25,1],fig_kwargs={'figsize':(40,10)})
    prate_pca.pca_ts_plot[0].set_layout_engine('tight')
    prate_pca.pca_ts_plot[0].show()
    prate_pca.pca_ts_plot[0].savefig(pca_figdir / 'pca_aggregate_sessions.svg')
