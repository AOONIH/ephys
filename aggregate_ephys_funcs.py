import argparse
import platform
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm

from behviour_analysis_funcs import get_all_cond_filts, get_sess_name_date_idx
from ephys_analysis_funcs import posix_from_win, load_sess_pkl, get_predictor_from_psth, Decoder
from population_analysis_funcs import compute_eig_vals, compute_trial_averaged_pca, project_pca, plot_pca_ts
from regression_funcs import run_glm


class PopPCA:

    def __init__(self, responses_by_cond: dict):
        self.proj_3d_plot = None
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
        event_concatenated_responses = event_concatenated_responses-np.nanmean(event_concatenated_responses,axis=1,
                                                                               keepdims=True)
        return event_concatenated_responses

    def get_eig_vals(self):
        self.eig_vals = compute_eig_vals(self.event_concatenated_responses, plot_flag=True)
        self.eig_vals[2][1].set_xlim(0, 30)
        self.eig_vals[2][1].set_ylabel('PC component')
        self.eig_vals[2][1].set_xlabel('Proportion of variance explained')
        # self.eig_vals[2][0].show()

    def get_trial_averaged_pca(self, n_components=15, standardise=True):
        self.Xa_trial_averaged_pca = compute_trial_averaged_pca(self.event_concatenated_responses,
                                                                n_components=n_components, standardise=standardise)

    def get_projected_pca_ts(self, standardise=True):
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
                     plot=[self.pca_ts_plot[0], axes[ei]], plot_kwargs={'ls': lss[cond_i], #'c': plt_cols[ei],
                                                                        'label': f'{cond} {event}'})
         for ei, (event, projected_responses) in enumerate(zip(events2plot[cond], [self.projected_pca_ts_by_cond[cond][e] for e in events2plot[cond]]))]
         for cond_i, cond in enumerate(conds2plot)]
        [row_axes[0].set_ylabel('PC component') for row_axes in self.pca_ts_plot[1]]
        [row_axes[0].legend(loc='upper center',ncol=4) for row_axes in self.pca_ts_plot[1].T]
        # [ax.legend() for ax in self.pca_ts_plot[1]]
        [ax.set_xlabel('Time from stimulus onset (s)') for ax in self.pca_ts_plot[1][-1]]
        self.pca_ts_plot[0].show()

    def plot_3d_pca_ts(self,prop, event_window, **kwargs):
        pca_comps_2plot = kwargs.get('pca_comps_2plot', [0, 1, 2])
        if kwargs.get('plot', None) is None:
            fig, axes = plt.subplots(ncols=kwargs.get('n_cols', 2), subplot_kw={"projection": "3d"}, figsize=(10, 10))
        else:
            fig, axes = kwargs.get('plot')

        # get projections
        smoothing = kwargs.get('smoothing', 3)
        proj_ts = {e: {pca_comp: self.projected_pca_ts_by_cond[prop][e][pca_comp] for pca_comp in pca_comps_2plot }
                   for e in self.projected_pca_ts_by_cond[prop].keys()}
        # smooth projections
        proj_ts = {e: {pca_comp: gaussian_filter1d(proj_ts[e][pca_comp], smoothing) for pca_comp in pca_comps_2plot}
                   for e in self.projected_pca_ts_by_cond[prop].keys()}
        # get x series and initial points
        t0_time_s = kwargs.get('t0_time', 0)
        t_end_s = kwargs.get('t_end', event_window[1])
        x_ser = kwargs.get('x_ser', None)
        if x_ser is None:
            x_ser = np.round(np.linspace(event_window[0], event_window[1], list(proj_ts.values())[0].shape[-1]), 2)
        t0_time_idx = np.where(x_ser == t0_time_s)[0][0]
        t_end_idx = np.where(x_ser == t_end_s)[0][0]
        for ei, (e,proj_pcas) in enumerate(proj_ts.items()):
            t0_points = [proj_pcas[pca_comp][t0_time_idx] for pca_comp in pca_comps_2plot]
            t_end_points = [proj_pcas[pca_comp][t_end_idx] for pca_comp in pca_comps_2plot]
            for ax in axes:
                idxs_in_event = np.logical_and(x_ser >= t0_time_s, x_ser <= t_end_s)
                idxs_out_event = np.logical_or(x_ser < t0_time_s, x_ser > t_end_s)
                for ii, (in_event, in_event_ls) in enumerate(zip([idxs_in_event, idxs_out_event],
                                                 kwargs.get('in_event_ls', ['-', '--']))):
                    if not kwargs.get('plot_out_event', True) and ii == 1:
                        continue
                    masked_ts = [proj_pcas[pca_comp].copy()
                                             for pca_comp in pca_comps_2plot]
                    # print(f'{len(masked_ts) = }')
                    for pca_comp,_ in enumerate(masked_ts):
                        masked_ts[pca_comp][np.invert(in_event)] = np.nan
                    # print(f'{masked_ts[0] = }')
                    ax.plot(*masked_ts, c=f'C{ei}', label=e if ii == 0 else None,
                            ls=in_event_ls)
                
                ax.scatter(*t0_points, c=f'C{ei}', marker='x',s=50)
                # scatter ends
                ax.scatter(*t_end_points, c=f'C{ei}', marker='s',s=50)

                if kwargs.get('scatter_times'):
                    t_pnts = kwargs.get('scatter_times')
                    scatter_kwargs = kwargs.get('scatter_kwargs', {})
                    if not isinstance(t_pnts, list):
                        t_pnts = [t_pnts]
                    for t_pnt in t_pnts:
                        t_pnt_idx = np.where(x_ser == t_pnt)[0][0]
                        t_pnts_pca = [pca_comp[t_pnt_idx] for pca_comp in proj_pcas.values()]
                        ax.scatter(*t_pnts_pca, **scatter_kwargs)

                ax.set_xlabel(f'PC{pca_comps_2plot[0]}')
                ax.set_ylabel(f'PC{pca_comps_2plot[1]}')
                ax.set_zlabel(f'PC{pca_comps_2plot[2]}')

        axes[0].view_init(elev=22, azim=30)
        axes[0].legend()
        self.proj_3d_plot = fig
        fig.show()

    def plot_2d_pca_ts(self, prop, event_window, **kwargs):
        pca_comps_2plot = kwargs.get('pca_comps_2plot', [0, 1])
        fig, ax = plt.subplots(figsize=(4, 3))

        # Get projections
        smoothing = kwargs.get('smoothing', 3)
        proj_ts = {e: {pca_comp: self.projected_pca_ts_by_cond[prop][e][pca_comp] for pca_comp in pca_comps_2plot}
                   for e in self.projected_pca_ts_by_cond[prop].keys()}

        # Smooth projections
        proj_ts = {e: {pca_comp: gaussian_filter1d(proj_ts[e][pca_comp], smoothing) for pca_comp in pca_comps_2plot}
                   for e in self.projected_pca_ts_by_cond[prop].keys()}

        # Get time series and initial points
        t0_time_s = kwargs.get('t0_time', 0)
        t_end_s = kwargs.get('t_end', event_window[1])
        x_ser = kwargs.get('x_ser', None)
        if x_ser is None:
            x_ser = np.round(np.linspace(event_window[0], event_window[1], list(proj_ts.values())[0].shape[-1]), 2)

        t0_time_idx = np.where(x_ser == t0_time_s)[0][0]
        t_end_idx = np.where(x_ser == t_end_s)[0][0]

        for ei, (e, proj_pcas) in enumerate(proj_ts.items()):
            t0_points = [proj_pcas[pca_comp][t0_time_idx] for pca_comp in pca_comps_2plot]
            t_end_points = [proj_pcas[pca_comp][t_end_idx] for pca_comp in pca_comps_2plot]

            idxs_in_event = np.logical_and(x_ser >= t0_time_s, x_ser <= t_end_s)
            idxs_out_event = np.logical_or(x_ser < t0_time_s, x_ser > t_end_s)

            for ii, (in_event, in_event_ls) in enumerate(zip([idxs_in_event, idxs_out_event],
                                                             kwargs.get('in_event_ls', ['-', '--']))):
                if not kwargs.get('plot_out_event', True) and ii == 1:
                    continue

                masked_ts = [proj_pcas[pca_comp].copy() for pca_comp in pca_comps_2plot]
                for pca_comp, _ in enumerate(masked_ts):
                    masked_ts[pca_comp][np.invert(in_event)] = np.nan

                ax.plot(*masked_ts, c=f'C{ei}', label=e if ii == 0 else None, ls=in_event_ls)

            ax.scatter(*t0_points, c=f'C{ei}', marker='x', s=50)
            ax.scatter(*t_end_points, c=f'C{ei}', marker='s', s=50)

            if kwargs.get('scatter_times'):
                t_pnts = kwargs.get('scatter_times')
                scatter_kwargs = kwargs.get('scatter_kwargs', {})
                if not isinstance(t_pnts, list):
                    t_pnts = [t_pnts]
                for t_pnt in t_pnts:
                    t_pnt_idx = np.where(x_ser == t_pnt)[0][0]
                    t_pnts_pca = [pca_comp[t_pnt_idx] for pca_comp in proj_pcas.values()]
                    ax.scatter(*t_pnts_pca, **scatter_kwargs)

        ax.set_xlabel(f'PC{pca_comps_2plot[0]}')
        ax.set_ylabel(f'PC{pca_comps_2plot[1]}')
        ax.legend()
        fig.show()


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

    td_dfs = {sessname:pd.read_csv(abs_td_path) for sessname, abs_td_path in zip(sessnames,abs_td_paths)
              if abs_td_path.is_file()}
    td_df = pd.concat(list(td_dfs.values()),keys=td_dfs.keys(),names=['sess'],axis=0)
    if td_df_query:
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
            print(f'{pkl_path} no td_df')
            continue
        if td_df_query is not None:
            if sess_obj.td_df.query(td_df_query).empty:
                print(f'{pkl_path} no td_df query')
                continue
        sess_objs.append(sess_obj)
    return {e.sessname: e for e in sess_objs}


def aggregate_event_reponses(sessions: dict, events=None, events2exclude=None, window=(0, 0.25),
                             pred_from_psth_kwargs=None, ):
    events_by_session = [list(sess.sound_event_dict.keys()) for sess in sessions.values()]
    # print(f'common events = {common_events}')
    common_events = events_by_session[0]
    for s_events in events_by_session:
        common_events = np.intersect1d(common_events, s_events)
    # print(f'{"A-0" in common_events = } ')
    if events:
        sessions2use = [sess for sess, s_events in zip(sessions,events_by_session) if all(e in s_events for e in events)]
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
    events_by_session = [list(sess.sound_event_dict.keys()) for sess in sessions.values()]
    # print(f'common events = {common_events}')
    common_events = events_by_session[0]
    for s_events in events_by_session:
        common_events = np.intersect1d(common_events, s_events)
    if events:
        sessions2use = [sess for sess, s_events in zip(sessions, events_by_session) if all(e in s_events for e in events)]
    else:
        sessions2use = sessions

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
        sess_td_df.index = sess_td_df.reset_index(drop=True).index+1
        for e in event_features_across_sessions[sessname]:
            # event_features_across_sessions[sessname][e]['td_df'] = sess_td_df.reset_index(drop=True).loc[
            #     event_features_across_sessions[sessname][e]['trial_nums'] - 1]
            trial_nums = event_features_across_sessions[sessname][e]['trial_nums']
            event_features_across_sessions[sessname][e]['td_df'] = sess_td_df.query('index in @trial_nums')
    return event_features_across_sessions


def concatenate_responses_by_td(responses_by_sess: dict, features_by_sess: dict, td_df_query=None):
    grouped_by_td = {}
    common_events = set.intersection(*[set(list(e.keys())) for e in responses_by_sess.values()])
    for event in tqdm(common_events, total=len(common_events), desc='Concatenating responses'):
        event_across_sess = [sess_responses[event][:len(sess_features[event]['td_df'])] for sess_responses,sess_features
                             in zip(responses_by_sess.values(), features_by_sess.values())]
        event_idxbool = [sess[event]['td_df'].eval(td_df_query).values for sess in features_by_sess.values()]
        try:
            grouped_by_td[event] = np.concatenate([np.nanmean(sess_e_responses[sess_idx_bool], axis=0)
                                                   for sess_e_responses, sess_idx_bool in
                                                   tqdm(zip(event_across_sess, event_idxbool), total=len(event_across_sess),
                                                        desc='Concatenating features')], axis=0)
        except IndexError:
            print(f'IndexError for {event}')
            grouped_by_td[event] = None

    return grouped_by_td


def group_ephys_resps(event_name:str,responses_by_sess: dict, features_by_sess: dict, td_df_query=None,
                      trial_nums2use=None,sess_list=None):
    resps_across_sess = {}
    trial_nums = {}
    for sess in sess_list if sess_list else responses_by_sess:
        if td_df_query:
            sess_resp_idxs = features_by_sess[sess][event_name]['td_df'].eval(td_df_query).values
        elif trial_nums2use is not None:
            sess_resp_idxs = np.isin(features_by_sess[sess][event_name]['trial_nums'], trial_nums2use)
        else:
            raise ValueError('must specify td_df_query or trial_nums2use')
        # print(sess_resp_idxs)
        if not np.any(sess_resp_idxs) or sess_resp_idxs.sum()<10:
            print(f'{sess} has not enough responses')
            # resps_across_sess[sess] = None
            continue
        sess_resps = responses_by_sess[sess][event_name]
        # print(f'{sess} has {len(sess_resps)} responses, {len(sess_resp_idxs)} indices')
        resps_across_sess[sess] = sess_resps[:len(sess_resp_idxs)][sess_resp_idxs]
        trial_nums[sess] = features_by_sess[sess][event_name]['trial_nums'][:len(sess_resp_idxs)][sess_resp_idxs]
    return resps_across_sess,trial_nums


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


def plot_aggr_cm(pip_cms, im_kwargs=None, **cm_kwargs):
    cm_plot = ConfusionMatrixDisplay(np.mean(pip_cms,axis=0) if pip_cms.ndim==3 else pip_cms,
                                     display_labels=cm_kwargs.get('labels',list(range(pip_cms.shape[-1]))))
    cm_plot.plot(cmap=cm_kwargs.get('cmap','bwr'),
                 include_values=cm_kwargs.get('include_values',False),colorbar=cm_kwargs.get('colorbar',True),
                 im_kw=im_kwargs if im_kwargs is not None else {})

    cm_plot.ax_.invert_yaxis()
    cm_plot.ax_.set_xlabel('')
    cm_plot.ax_.set_ylabel('')
    cm_plot.figure_.set_size_inches(cm_kwargs.get('figsize',(2,2)))
    # cm_plot.figure_.set_layout_engine('constrained')

    return cm_plot.figure_, cm_plot.ax_


def decode_over_sliding_t(resps_dict: dict, window_s, resp_window, pips_as_ints_dict, pips2decode: list,
                          animals_to_use:list=None):
    resp_width = list(resps_dict.values())[0][pips2decode[0]].shape[-1]
    resp_x_ser = np.round(np.linspace(resp_window[0], resp_window[1], resp_width), 2)
    window_size = int(np.round(window_s / (resp_x_ser[1] - resp_x_ser[0])))

    dec_res_dict = {}
    bad_dec_sess = set()
    for sessname in tqdm(list(resps_dict.keys()), total=len(resps_dict),
                         desc='decoding across sessions'):
        # if not any(e in sessname for e in ['DO79', 'DO81']):
        if animals_to_use:
            if not any(e in sessname for e in animals_to_use):
                continue

        for t in tqdm(range(resp_width - window_size), total=resp_width - window_size, desc='decoding across time'):
            if 'A-0' in pips2decode:
                xys = [(
                    resps_dict[sessname][pip][150:200][::3] if pip.split('-')[1] == '0' else
                    resps_dict[sessname][pip],
                    np.full_like(resps_dict[sessname][pip][:, 0, 0], pips_as_ints_dict[pip]))
                    for pip in pips2decode]
            else:
                xys = [(resps_dict[sessname][pip],
                        np.full_like(resps_dict[sessname][pip][:, 0, 0], pips_as_ints_dict[pip]))
                       for pip in pips2decode]
            ys = [np.full(xy[0].shape[0], pips_as_ints_dict[pip]) for xy, pip in zip(xys, pips2decode)]
            xs = np.vstack([xy[0][:, :, t:t + window_size].mean(axis=-1) for xy in xys])
            # xs = [xy[0][:,:,15:].mean(axis=-1) for xy in xys]
            ys = np.hstack(ys)
            # if np.unique(ys).shape[0] < len(patt_is):
            #     continue
            try:
                dec_res_dict[f'{sessname}-{t}:{t + window_size}s'] = decode_responses(xs, ys, n_runs=50,
                                                                                      dec_kwargs={'cv_folds': 10})
            except ValueError:
                print(f'{sessname} failed')

                bad_dec_sess.add(sessname)
                continue
    # [decoders_dict[dec_name]['data'].plot_confusion_matrix([f'{pip}-{pip_i}' for pip in 'D' for pip_i in patt_is])
    #  for dec_name in decoders_dict.keys()]
    sess2use = [dec_name.split('-')[0] for dec_name in dec_res_dict.keys()]
    norm_dev_accs_ts_dict = {
        sessname: {t: np.mean(dec_res_dict[f'{sessname}-{t}:{t + window_size}s']['data'].accuracy)
                   for t in range(resp_width - window_size)}
        for sessname in sess2use if sessname not in bad_dec_sess}

    norm_dev_accs_ts_df = pd.DataFrame(norm_dev_accs_ts_dict).T
    norm_dev_accs_ts_df.columns = np.round(resp_x_ser[window_size:], 2)

    return norm_dev_accs_ts_df, norm_dev_accs_ts_dict


def predict_from_responses(dec_model,responses):
    return dec_model.predict(responses)

def predict_over_sliding_t(dec_model,resps_dict,pips2predict,window_s,resp_window):

    resp_width = list(resps_dict.values())[0][pips2predict[0]].shape[-1]
    resp_x_ser = np.round(np.linspace(resp_window[0], resp_window[1], resp_width), 2)
    window_size = int(np.round(window_s / (resp_x_ser[1] - resp_x_ser[0])))

    responses = []
    for pips2predict in pips2predict:
        pip_resp = resps_dict[pips2predict]
        if pip_resp.ndim == 3:
            pass


    return dec_model.predict(responses[:,:,window_size:])


def get_active_units(resp_mat: np.ndarray, resp_window: tuple|list, activity_window_S: list|tqdm,
                     active_thresh: float, **kwargs):
    # get response x series and window in idxs
    x_ser = np.round(np.linspace(*resp_window, resp_mat.shape[-1]),2)
    x_ser = np.round(x_ser, 2)
    window_idxs = [np.where(x_ser == t)[0][0] for t in activity_window_S]
    # get active units
    max_func = partial(kwargs.get('max_func', np.max), axis=-1)
    max_activity = max_func(resp_mat[:,window_idxs[0]:window_idxs[1]] if resp_mat.ndim == 2 else
                            resp_mat[:, :, window_idxs[0]:window_idxs[1]])
    active_units_bool = max_activity > active_thresh
    if resp_mat.ndim == 3:
        active_units_bool = active_units_bool.mean(axis=0)

    return active_units_bool


if __name__ == '__main__':
    pass