import pickle
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import ttest_ind, ttest_1samp, f_oneway, tukey_hsd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

from aggregate_ephys_funcs import *
from ephys_analysis_funcs import get_pip_desc, format_axis, plot_sorted_psth
from neural_similarity_funcs import get_list_pips_by_property, plot_similarity_mat, compute_self_similarity, \
    compare_pip_sims_2way, plot_sim_by_grouping
from population_analysis_funcs import compute_mi
# from npeet import entropy_estimators as ee
# from pyentrp import entropy as ent


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

    all_sess_info = session_topology.query('sess_order=="main" ')
    all_sess_td_df = load_aggregate_td_df(all_sess_info,home_dir,'Stage==5')
    sessions2use = all_sess_td_df.index.get_level_values('sess').unique()
    # sessnames = [Path(sess_info['sound_bin']).stem.replace('_SoundData', '') for _, sess_info in
    #              sessions2use.iterrows()]
    hipp_animals = ['DO79','DO81']
    animals = session_topology['name'].unique().tolist()
    # pca
    pca_figdir = ceph_dir / 'Dammy' / 'figures' / 'pca_aggregate_sessions_all_new_abstraction_new'
    if not pca_figdir.exists():
        pca_figdir.mkdir()
    aggr_abstr_figdir = pca_figdir.parent / 'aggregate_abstraction_sessions'
    if not aggr_abstr_figdir.exists():
        aggr_abstr_figdir.mkdir()

    pkldir = ceph_dir / posix_from_win(args.pkldir)
    pkls = list(pkldir.glob('*pkl'))
    pkls2load = [pkl for pkl in pkls if Path(pkl).stem in sessions2use]
    sessions = load_aggregate_sessions(pkls2load,'Stage==5')

    [sessions.pop(sess) for sess in list(sessions.keys()) if 'A-0' not in list(sessions[sess].sound_event_dict.keys())]
    [sessions.pop(sess) for sess in list(sessions.keys()) if 'A-3' not in list(sessions[sess].sound_event_dict.keys())]
    [sessions.pop(sess) for sess in list(sessions.keys()) if 'D-3' not in list(sessions[sess].sound_event_dict.keys())]
    [sessions.pop(sess) for sess in list(sessions.keys()) if 'A-4' in list(sessions[sess].sound_event_dict.keys())]
    [print(sess,sessions[sess].sound_event_dict.keys()) for sess in list(sessions.keys()) ]

    # update styles
    plt.style.use('figure_stylesheet.mplstyle')

    full_pattern_responses = aggregate_event_reponses(sessions, events=None,
                                                      events2exclude=['trial_start'], window=[-0.25, 1],
                                                      pred_from_psth_kwargs={'use_unit_zscore': True,
                                                                             'use_iti_zscore': False,
                                                                             'baseline': 0, 'mean': None,
                                                                             'mean_axis': 0})

    # dump full pattern responses
    full_patt_dict_path = ceph_dir / Path(posix_from_win(r'X:\for_will')) / 'full_patt_dict_ABCD_vs_ABBA.pkl'
    if not full_patt_dict_path.is_file():
        with open(full_patt_dict_path, 'wb') as f:
            pickle.dump(full_pattern_responses, f)
    # assert False

    window = (-0.1, 0.25)
    event_responses = aggregate_event_reponses(sessions,
                                               events2exclude=['trial_start'], window=window,
                                               pred_from_psth_kwargs={'use_unit_zscore': True, 'use_iti_zscore': False,
                                                                      'baseline': 0, 'mean': None, 'mean_axis': 0})

    event_features = aggregate_event_features(sessions,
                                              events2exclude=['trial_start'])
    # construct array
    concatenated_event_responses = {
        e: np.concatenate([event_responses[sessname][e].mean(axis=0) for sessname in event_responses])
        for e in list(event_responses.values())[0].keys()}
    concatenated_event_times = {
        e: np.concatenate([(event_features[sessname][e]['times']) for sessname in event_features])
        for e in list(event_features.values())[0].keys()}

    #
    # get n_units
    n_units = concatenated_event_responses[list(concatenated_event_responses.keys())[0]].shape[0]
    cond_filts = get_all_cond_filts()
    # group by property
    common_events = sorted(set.intersection(*[set(list(e.keys())) for e in event_responses.values()]))
    n_per_rule = 2
    events_by_property = {
        'position': {pip: ord(pip.split('-')[0]) - ord('A') + 1 for pip in common_events
                                       if pip.split('-')[0] in 'ABCD'},
        'group': {pip: int(int(pip.split('-')[1]) / n_per_rule) for pip in common_events
                  if pip.split('-')[0] in 'ABCD'},
        'ptype_i': {pip: 0 if int(pip.split('-')[1]) % n_per_rule == 0 else 1 for pip in common_events
                    if pip.split('-')[0] in 'ABCD'},
        'name': {pip: pip for pip in common_events
                 if pip.split('-')[0] in 'ABCD'},
        'pattern_i': {pip: pip.split('-')[1] for pip in common_events
                      if pip.split('-')[0] in 'ABCD'},
    }
    pip_desc = sessions[list(sessions.keys())[1]].pip_desc
    # events_by_property['id'] = {pip: pip if events_by_property['ptype_i'][pip] == 0 else
    #                             f'{"ABBA"[events_by_property["position"][pip]-1]}-{pip.split("-")[1]}'
    #                             for pip in common_events
    #                             if pip.split('-')[0] in 'ABCD'}
    events_by_property['id'] = {pip: pip_desc[pip]['idx']
                                for pip in common_events
                                if pip.split('-')[0] in 'ABCD'}

    concatenated_events_by_pip_prop = group_responses_by_pip_prop(concatenated_event_responses, events_by_property)
    # plot mean responses to rare and frequent pips
    for prop, prop_val in concatenated_events_by_pip_prop.items():
        property_response_plot = plt.subplots()
        for cond_i, (cond, ls) in enumerate(zip(['ptype_i'], ['-', '--'])):
            for pip_i, pip in enumerate(sorted(concatenated_events_by_pip_prop[cond])):
                x_ser = np.linspace(*window, len(concatenated_events_by_pip_prop[cond][pip].mean(axis=0)))
                property_response_plot[1].plot(x_ser, concatenated_events_by_pip_prop[cond][pip].mean(axis=0),
                                               c=f'C{pip_i}', label=f'{cond} {pip}', ls=ls)
        property_response_plot[1].legend()
        property_response_plot[1].set_title(f'Population responses to rare and frequent pips')
        property_response_plot[1].set_xlabel('Time from sound onset (s)')
        property_response_plot[1].axvline(0, color='k', linestyle='--')
        property_response_plot[1].set_ylabel('Firing rate')
        property_response_plot[0].set_layout_engine('tight')
        property_response_plot[0].show()

    # # bootstrap responses
    # # def bootstrap_responses(x,axis=0):
    # #     return x.mean(axis=axis)
    # n_resamples = 9999
    # n_samples = 400  # int(n_units * 0.5)
    # bootstrap_responses = {cond: {pip: np.array([concatenated_events_by_prate[cond][pip][sample_idxs]
    #                                              for sample_idxs in [np.random.choice(n_units, size=n_samples,
    #                                                                                   replace=True)
    #                                                                  for _ in range(n_resamples)]])
    #                               for pip in cond_responses}
    #                        for cond, cond_responses in concatenated_events_by_prate.items()}
    #
    # # plot bootstrap
    # bootstrap_plot = plt.subplots(ncols=len('ABCD'), sharex=True, sharey=True, figsize=(24, 6))
    # for cond_i, (cond, ls) in enumerate(zip(['rare', 'frequent'], ['-', '--'])):
    #     for pip_i, (pip, ax) in enumerate(zip(sorted(bootstrap_responses[cond]), bootstrap_plot[1])):
    #         x_ser = np.linspace(*window, bootstrap_responses[cond][pip].shape[-1])
    #         ax.plot(x_ser, bootstrap_responses[cond][pip].mean(axis=0).mean(axis=0),
    #                 c=f'C{pip_i}', label=f'{cond}', ls=ls)
    #         # plot confidence intervals
    #         ax.fill_between(x_ser, np.quantile(bootstrap_responses[cond][pip].mean(axis=1), 0.025, axis=0),
    #                         np.quantile(bootstrap_responses[cond][pip].mean(axis=1), 0.975, axis=0),
    #                         color=f'C{pip_i}', alpha=0.05)
    #         ax.legend()
    #         ax.set_title(pip)
    #         ax.set_xlabel('Time from sound onset (s)')
    #         ax.axvline(0, color='k', linestyle='--')
    #
    # bootstrap_plot[1][0].set_ylabel('Firing rate')
    # bootstrap_plot[0].suptitle('Bootstrap responses to rare and frequent pips')
    # bootstrap_plot[0].set_layout_engine('tight')
    # bootstrap_plot[0].show()

    # glm by unit


    concatenated_full_pattern_responses = {
        e: np.concatenate([full_pattern_responses[sessname][e].mean(axis=0) for sessname in full_pattern_responses])
        for e in [f'A-{i}' for i in range(4)]}

    concatenated_full_pattern_by_pip_prop = group_responses_by_pip_prop(concatenated_full_pattern_responses,
                                                                        events_by_property, ['ptype_i'])

    # mutual info
    events = [f'{pip}-{i}' for i in range(4) for pip in 'ABCD']
    resps2use = aggregate_event_reponses(sessions,
                                               events2exclude=['trial_start'], window=window,
                                               pred_from_psth_kwargs={'use_unit_zscore': True, 'use_iti_zscore': False,
                                                                      'baseline': 0, 'mean': None, 'mean_axis': 0})
    # resps2use = full_pattern_responses
    # window2use= [-0.25, 1]
    window2use= [-0.1, 0.25]
    x_ser = np.round(np.linspace(*window2use, resps2use[list(resps2use.keys())[0]]['A-0'].shape[-1]),2)
    # ts_s = np.arange(window2use[0], window2use[1], 0.1)
    # ts_s = np.round(np.arange(window2use[0], window2use[1], 0.1),1)
    # window2use= np.arange(-0.2, 1.1, 0.1)
    # ts_s = [0.5,1]
    ts_s = [0.25]
    # ts_s = [0.5]
    ts_idx = [np.where(x_ser==tt)[0][0] for tt in ts_s if tt in x_ser]
    # ts_idx = [0,-1]
    prop = 'ptype_i'
    mi_by_t_by_pip = {}
    for pip in 'BD':
        mi_by_t = {}
        for t_idx in ts_idx:
            mi_by_sess = {}
            for sessname in tqdm(list(resps2use.keys()), total=len(resps2use), desc='computing MI'):
                n_units= resps2use[sessname][events[0]].shape[1]
                # all_responses = [np.squeeze(resps2use[sessname][e][:,:,t_idx]) for e in events+['X','base'] ]  #
                # resp_min = np.min([np.min(r[r>0]) for r in all_responses])
                # resp_max = np.max([np.max(r[r>0]) for r in all_responses])
                # bins = np.linspace(resp_min, resp_max, 100)
                prop_mi = {}

                rule_lbls = np.unique(list(events_by_property[prop].values()))
                prop_events = [[e for e in events if events_by_property[prop][e] == prop_val if any(p in e for p in pip)]
                               for prop_val in rule_lbls]
                cond_responses = [[np.nanmean(np.squeeze(resps2use[sessname][e])[:, :,t_idx-5:t_idx], axis=-1)
                                   for e in rule_events]
                                  for rule_events in prop_events]
                responses_by_rule = {rule: np.vstack([e for e in responses_to_stims if len(e) > 0])
                                     for responses_to_stims, rule in zip(cond_responses, rule_lbls)
                                     if len(responses_to_stims) > 0}
                rule_sim = [[cosine_similarity()] for rule_respones in responses_by_rule.values()]


                # mi_by_sess[sessname] = np.mean(something)
                # mi_by_sess[sessname] = [glm[1].params[0] for glm in mi_by_unit if glm[1].pvalues[0] < 0.05]
            mi_by_t[x_ser[t_idx]] = mi_by_sess
        mi_by_t_by_pip[pip] = mi_by_t

    ts2plot = ts_s
    mi_dist_plot = plt.subplots(len(mi_by_t_by_pip),sharey=True, sharex=True, figsize=(6, 4*len(mi_by_t_by_pip)),squeeze=False)
    for (pip,mi_by_t),plot in zip(mi_by_t_by_pip.items(),mi_dist_plot[1].flatten()):
        for t_lbl,prop_mi in mi_by_t.items():
            # if t_lbl not in ts2plot:
            #     continue
            mi_across_sessions = np.hstack(list(prop_mi.values()))
            plot.hist(mi_across_sessions,bins='fd',alpha=0.2,label=t_lbl,density=True)
            print(f'{pip}, {t_lbl}, {mi_across_sessions.shape}, {np.nanmean(mi_across_sessions)}')
        plot.legend()
        plot.set_ylabel('')

        plot.set_title(f'MI distribution for {prop}')
        # [print(pip,np.nanpercentile(np.hstack(list(prop_mi.values())),90)) for prop_mi, t_lbl in zip(mi_by_t, ts_s)]
        # [print(pip,np.nanmean(np.hstack(list(prop_mi.values())))) for prop_mi, t_lbl in zip(mi_by_t, ts_s)]
        # print(pip,ttest_ind(*[np.hstack(list(prop_mi.values())) for prop_mi, t_lbl in zip(mi_by_t, ts_s)],
        #                     alternative='less'))
    mi_dist_plot[0].show()

    mi_ts_plot = plt.subplots()
    # ts =
    mi_over_t  = [0
                  for (pip,mi_by_t),plot in zip(mi_by_t_by_pip.items(),mi_dist_plot[1].flatten())]

    [(t_lbl,np.hstack(list(prop_mi.values())).mean()) for prop_mi, t_lbl in zip(mi_by_t, ts_s)]
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



    # Xa_trial_averaged = np.hstack(list(concatenated_event_responses.values()))
    ptype_pca = PopPCA(concatenated_full_pattern_by_pip_prop)
    ptype_pca.eig_vals[2][0].show()
    ptype_pca.get_trial_averaged_pca(standardise=True)
    ptype_pca.get_projected_pca_ts(standardise=True)
    ptype_pca.plot_pca_ts([-0.25,1],fig_kwargs={'figsize':(120,10)},plot_separately=False,n_comp_toplot=15)
    [[ax.axvspan(t, t+0.15, color='k', alpha=0.1) for t in np.arange(0,1,0.25)]
     for ax in ptype_pca.pca_ts_plot[1].flatten()]
    ptype_pca.pca_ts_plot[0].set_layout_engine('tight')
    ptype_pca.pca_ts_plot[0].show()
    ptype_pca.pca_ts_plot[0].savefig(pca_figdir / f'all_groupings_pca_aggregate_sessions.png')

    # indv_pips pca
    indv_pip_pca_figdir = ceph_dir / 'Dammy' / 'figures' / 'indv_pip_pca_aggregate_sessions_all_new_abstraction_new'
    if not indv_pip_pca_figdir.is_dir():
        indv_pip_pca_figdir.mkdir()
    concatenated_events_by_pip_prop = group_responses_by_pip_prop(concatenated_event_responses,
                                                                  events_by_property,['pattern_i'],
                                                                  concatenate_flag=False)
    concatenated_full_pattern_by_pip_prop = group_responses_by_pip_prop(concatenated_full_pattern_responses,
                                                                        events_by_property, ['pattern_i'],
                                                                        concatenate_flag=False)

    indv_pip_pca = PopPCA(concatenated_events_by_pip_prop['pattern_i'])
    # indv_pip_pca.eig_vals[2][0].show()
    indv_pip_pca.get_trial_averaged_pca()
    indv_pip_pca.get_projected_pca_ts()
    # events2plot = {'name':[f'A-{i}' for i in range(4)]}
        # events2plot = {'name':[f'{pip}-{i}' for i in range(4)]}
    events2plot = None
    indv_pip_pca.plot_pca_ts([-.1,0.25],fig_kwargs={'figsize':(120,10)},plot_separately=True,n_comp_toplot=15,
                             conds2plot=None,events2plot=events2plot,
                             # plot=(indv_pip_plot[0],[indv_pip_plot[1][pi]]),
                             plot=None, lss=['-','--','-','--'],
                             plt_cols=['C'+str(i) for i in [0,0,1,1]])
    # [[ax.axvspan(t, t+0.15, color='k', alpha=0.1) for t in np.arange(0,1,0.25)]
    #  for ax in indv_pip_pca.pca_ts_plot[1].flatten()]
    indv_pip_pca.pca_ts_plot[0].set_size_inches(150,30)
    indv_pip_pca.pca_ts_plot[0].set_layout_engine('tight')
    indv_pip_pca.pca_ts_plot[0].show()
    indv_pip_pca.pca_ts_plot[0].savefig(indv_pip_pca_figdir / f'by_pip_pca_aggregate_sessions.pdf')

    full_pattern_pca = PopPCA(concatenated_full_pattern_by_pip_prop['pattern_i'])
    full_pattern_pca.eig_vals[2][0].show()
    full_pattern_pca.get_trial_averaged_pca(standardise=False)
    full_pattern_pca.get_projected_pca_ts(standardise=False)
    full_pattern_pca.plot_pca_ts([-0.25,1],fig_kwargs={'figsize':(120,8)},plot_separately=False,n_comp_toplot=15,
                                 lss=['-', '--', '-', '--'], plt_cols=['C' + str(i) for i in [0, 0, 1, 1]]
                                 )
    [[ax.axvspan(t, t+0.15, color='grey', alpha=0.1) for t in np.arange(0,1,0.25)]
     for ax in full_pattern_pca.pca_ts_plot[1].flatten()]
    full_pattern_pca.pca_ts_plot[0].set_size_inches(150, 10)
    full_pattern_pca.pca_ts_plot[0].set_layout_engine('tight')
    full_pattern_pca.pca_ts_plot[0].show()
    full_pattern_pca.pca_ts_plot[0].savefig(indv_pip_pca_figdir / f'full_pattern_pca_aggregate_sessions.pdf')

    # cosine sim mega session

    # cosine sims across sessions
    pips_as_ints = {pip: pip_i for pip_i, pip in enumerate([f'{pip}-{pip_i}' for pip in 'ABCD' for pip_i in range(4)])}
    event_names = list(pips_as_ints.keys())
    prop = 'ptype_i'
    non_prop = 'group'
    prop_sim_diff_by_sess = {}
    prop_sim_by_sess = {}
    sess_self_sims = {}
    for sessname in tqdm(list(event_responses.keys())[:], total=len(event_responses), desc='similarities over sessions'):
        event_resps_by_pip = [event_responses[sessname][e] for e in event_names]
        self_sims = [compare_pip_sims_2way([e_resps[:,:,10:30]],mean_flag=True,n_shuffles=50)[0].mean(axis=0)[0,1]
                     for e_resps in event_resps_by_pip]
        sess_self_sims[sessname] = self_sims
        sim_mat = cosine_similarity([event_resps[:,:,10:30].mean(axis=-1).mean(axis=0)
                                     for event_resps in event_resps_by_pip])
        sim_mat = np.array([sim_row/self_sim for self_sim, sim_row in zip(self_sims,sim_mat)])
        if not all([self_sim>0.4  for self_sim in self_sims]):
            continue
        sim_mat_plot = plot_similarity_mat(sim_mat,event_names,cmap='Reds')
        sim_mat_plot[1].set_title(sessname)
        # sim_mat_plot[0].show()
        # print(f'{sessname} mean sim',sim_mat[~np.eye(sim_mat.shape[0],dtype=bool)].reshape(sim_mat.shape[0],-1).mean())
        within_prop_idxs = [[events_by_property[prop][ee] == events_by_property[prop][e] and e[0] == ee[0] for ee in event_names]
                            for e in event_names]
        within_prop_sim = [sim_mat[ei][e_idxs] for ei, e_idxs in enumerate(within_prop_idxs)]
        within_prop_sim_means = {e:within_prop_sim[ei][within_prop_sim[ei] != 1].mean()
                                 for ei, e in enumerate(event_names)}
        non_prop_sim_idxs = [[events_by_property[non_prop][ee] == events_by_property[non_prop][e] and e[0] == ee[0] for ee in event_names]
                             for e in event_names]
        # non_prop_sim_idxs = [np.invert(within_prop_idxs[ei]) for ei in range(len(within_prop_idxs))]
        non_prop_sim = [sim_mat[ei][e_idxs] for ei, e_idxs in enumerate(non_prop_sim_idxs)]
        non_prop_sim_means = {e:non_prop_sim[ei][non_prop_sim[ei] != 1].mean() for ei, e in enumerate(event_names)}
        prop_sim_diff = {e:within_prop_sim_means[e] - non_prop_sim_means[e] for e in event_names}
        prop_sim_diff_by_sess[sessname] = prop_sim_diff
        prop_sim_by_sess[sessname] = {'prop':within_prop_sim_means,'non_prop': non_prop_sim_means}
        # within_prop_sim = np.array([np.mean(e) for e in within_prop_sim])
        # print(f'{sessname}: {ttest_1samp(within_prop_sim, 0, alternative="greater")}')

    prop_sim_diff_df = pd.DataFrame(prop_sim_diff_by_sess).T
    prop_sim_diff_plot = plt.subplots()
    # prop_sim_diff_plot[1].boxplot([[prop_sim_diff_by_sess[sess][e] for sess in prop_sim_diff_by_sess.keys()]
    #                                for e in event_names], labels=event_names,showfliers=True)
    # prop_sim_diff_plot[1].boxplot([[prop_sim_diff_by_sess[sess][e] for sess in prop_sim_diff_by_sess.keys()]
    #                                for e in [f'{p}-1' for p in 'ABCD']], labels=[f'{p}-1' for p in 'ABCD'],
    #                               showfliers=True)
    prop_sim_diff_plot[1].boxplot([prop_sim_diff_df[[f'{p}-{pi}' for pi in [0,1,2,3][1::2]]].mean(axis=1)
                                   for p in 'ABCD'],labels=list('ABCD'),
                                  showfliers=False,meanline=True,widths=0.6,patch_artist=False,
                                  medianprops={"color":"darkred","linewidth":3},bootstrap=1000,whis=[2.5,97.5],
                                  boxprops={"linewidth":1},
                                  # autorange=True,
                                  meanprops={"marker":"^","markerfacecolor":"k",},
                                  flierprops={"marker":"o","markerfacecolor":"w","markeredgecolor":"k",
                                              "markersize":5,"markeredgewidth":1})
    # prop_sim_diff_plot[1].set_title(f'Similarity difference of {prop} vs {non_prop}')
    # prop_sim_diff_plot[1].axhline(0, color='k', ls='--')
    # prop_sim_diff_plot[1].set_ylim(-0.6,0.2)
    format_axis(prop_sim_diff_plot[1], hlines=[0])
    # prop_sim_diff_plot[1].set_yticklabels('')
    # prop_sim_diff_plot[1].set_xticklabels('')
    prop_sim_diff_plot[0].set_size_inches(3,3)
    prop_sim_diff_plot[0].set_layout_engine('tight')
    prop_sim_diff_plot[0].show()
    prop_sim_diff_plot[0].savefig(ceph_dir/posix_from_win(r'X:\Dammy\figures\abstraction_figs') /
                                  f'{prop}_vs_{non_prop}_sim_diff_plot_no_norm_same_pos_ci_whisks.png')

    pip_sim_diff = [[prop_sim_diff_by_sess[sess][e] for sess in prop_sim_diff_by_sess.keys()]
                                   for e in [f'{p}-1' for p in 'ABCD']]
    [print(f'{e}: {ttest_1samp(pip_sim_diff[ei], 0, alternative="greater")}') for ei, e in enumerate([f'{p}-1' for p in 'ABCD'])]
    print(f'A vs D: {ttest_ind(pip_sim_diff[0], pip_sim_diff[-1] , alternative="two-sided")}')
    print(tukey_hsd(*[prop_sim_diff_df[[f'{p}-{pi}' for pi in [0,1,2,3]]].mean(axis=1).values
                     for p in 'ABCD']))

    self_sim_by_sess_df = pd.DataFrame.from_dict(sess_self_sims,orient='index',columns=event_names)
    self_sim_plot = plt.subplots()
    self_sim_plot[1].boxplot([e for e in self_sim_by_sess_df.values.T],labels=event_names,showfliers=False,
                             bootstrap=1000,whis=[2.5,97.5],
                             medianprops={"color":"darkred","linewidth":3})
    self_sim_plot[1].locator_params(axis='y', nbins=5)
    self_sim_plot[0].show()
    self_sim_plot[0].savefig(ceph_dir/posix_from_win(r'X:\Dammy\figures\abstraction_figs') /
                             'self_sim_plot_across_sessions.png')

    prop_non_prop_sim_df = pd.concat([pd.concat([pd.DataFrame(sims[sim_sffx],index=[sess],)
                                      for sess,sims in prop_sim_by_sess.items()],axis=0,)
                                      for sim_i,sim_sffx in enumerate(['prop','non_prop'])],
                                     axis=1)
    prop_non_prop_sim_df.columns = [f'{event}_{sffx}' for sffx in ['prop','non_prop'] for event in event_names]
    prop_non_prop_sim_df = prop_non_prop_sim_df[[f'{event}_{sffx}' for event in event_names for sffx in ['prop','non_prop']]]

    prop_vs_non_prop_plot = plt.subplots()
    prop_sim_by_pip_i = {f'{pip}_{sffx}':prop_non_prop_sim_df[[col for col in prop_non_prop_sim_df.columns if pip in col and sffx in col]].mean(axis=1) for pip in 'ABCD'
                                     for sffx in ['prop','non_prop',]}
    prop_sim_by_pip_i_by_sess = {sess: {e:prop_sim_by_pip_i[e][sess] for e in prop_sim_by_pip_i.keys()}
                                 for sess in prop_non_prop_sim_df.index}
    prop_vs_non_prop_plot[1].boxplot(prop_sim_by_pip_i.values(),
                                     labels=prop_sim_by_pip_i.keys(),showfliers=False,)
    prop_vs_non_prop_plot[1].locator_params(axis='y', nbins=5)
    prop_vs_non_prop_plot[0].set_size_inches(20,6)
    prop_vs_non_prop_plot[0].show()
    sim_ts_plot = plt.subplots()
    [sim_ts_plot[1].plot(list(sess_sims.values()),
                         c='k',lw=0.2) for sess,sess_sims in prop_sim_by_pip_i_by_sess.items()]
    sim_ts_plot[0].set_size_inches(20,6)

    sim_ts_plot[0].show()

    full_pattern_responses_4_psth = aggregate_event_reponses(sessions, events=[e for e in concatenated_event_responses.keys()
                                                                        if 'A' in e],
                                                      events2exclude=['trial_start'], window=[-0.25, 1],
                                                      pred_from_psth_kwargs={'use_unit_zscore': True,
                                                                             'use_iti_zscore': False,
                                                                             'baseline': 0, 'mean': None,
                                                                             'mean_axis': 0})
    psth_figdir = ceph_dir/posix_from_win(r'X:\Dammy\figures\abstraction_figs')
    all_resps_psth = plt.subplots(len([e for e in concatenated_event_responses.keys() if 'A' in e]),4,figsize=(8,10),
                                  sharex=True,sharey='row',gridspec_kw={'wspace':0.05,'hspace':0.1})
    for pi,pip in enumerate([e for e in concatenated_event_responses.keys() if 'A' in e]):
        for ai,animal in enumerate(animals):
            cmap_norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
            plot_sorted_psth(full_pattern_responses_4_psth,pip,'A-0',window=[-0.25,1],sort_window=[0.1,1],
                                              sessname_filter=animal,
                             im_kwargs=dict(norm=cmap_norm,cmap='bwr',
                                            plot_cbar=True if pi == len(all_resps_psth[1][0])-1 else False),
                                              plot_ts=False,plot=(all_resps_psth[0],all_resps_psth[1][ai][pi]),
                             )
            all_resps_psth[0].set_layout_engine('tight')
            format_axis(all_resps_psth[1][ai][pi],vlines=np.arange(0,1,0.25).tolist(),ylabel='Unit #',lw=1,
                        xlabel=f'')
            if ai != len(animals)-1:
                # all_resps_psth[1][ai][pi].set_xticks([])
                all_resps_psth[1][ai][pi].set_xlabel('')
            if pi != 0:
                # all_resps_psth[1][ai][pi].set_yticks([])
                all_resps_psth[1][ai][pi].set_ylabel('')
            # format_axis(all_resps_psth[1][0],vlines=[0])
            # all_resps_psth[1][0].locator_params(axis='y', nbins=2)
            # all_resps_psth[0].suptitle(f'{animal}: {pip}')
    all_resps_psth[0].show()
    all_resps_psth[0].savefig(psth_figdir/f'all_mice_all_pips_psth_A0_sort.png')

    pvp_dec_dict = {}
    # pips2decode = [['A-0', 'A-1'], ['A-0', 'A-2'], ['A-1', 'A-2'],['A-0','A-0']]
    pips2decode = [['A-0;A-2', 'A-1;A-3'], ['A-0;A-1', 'A-2;A-3'], ['A-0', 'A-1'],['A-0','A-2'],['A-0','A-3'],
                   ['A-0','A-2','A-1','A-3']]
    for pips in pips2decode[:]:
        dec_sffx = "_vs_".join(pips)
        for sessname in tqdm(full_pattern_responses.keys(), desc='decoding',
                             total=len(full_pattern_responses.keys())):
        #     if sessname != 'DO79_240717a':
        #         continue
        # for sessname in ['DO79_240719a']:
            xs_list = [np.vstack([full_pattern_responses[sessname][p] for p in pip.split(';')]) for pip in pips]
            xs = np.vstack([x[:, :, -50:].mean(axis=-1) for x in xs_list])
            ys = np.hstack([np.full(x.shape[0], ci) for ci, x in enumerate(xs_list)])
            # pre_split_idx = xs_list[0].shape[0]
            pvp_dec_dict[f'{sessname}_{dec_sffx}'] = decode_responses(xs, ys, n_runs=1,dec_kwargs={'cv_folds':10,})
            pvp_dec_dict[f'{sessname}_{dec_sffx}']['data'].plot_confusion_matrix(pips)
        # plot accuracy
        pvp_accuracy = np.array([pvp_dec_dict[dec_name]['data'].accuracy
                                 for dec_name in pvp_dec_dict.keys() if dec_sffx in dec_name
                                 and any([e in dec_name for e in hipp_animals])])
        pvp_shuffle_accs = np.array([pvp_dec_dict[dec_name]['shuffled'].accuracy
                                     for dec_name in pvp_dec_dict.keys() if dec_sffx in dec_name
                                     and any([e in dec_name for e in hipp_animals])])
        pvp_accuracy_plot = plt.subplots()
        pvp_accuracy_plot[1].boxplot([pvp_accuracy.mean(axis=1),
                                      pvp_shuffle_accs.mean(axis=1)], labels=['data', 'shuffle'],
                                     showmeans=False, meanprops=dict(mfc='k'), )
        pvp_accuracy_plot[1].set_title(f'{dec_sffx}')
        pvp_accuracy_plot[1].set_ylabel('Accuracy')
        format_axis(pvp_accuracy_plot[1],hlines=[1/len(pips)])
        pvp_accuracy_plot[0].show()
        # pvp_accuracy_plot[0].savefig(dev_ABBA1_figdir / f'{dec_sffx}_accuracy.pdf')
        # ttest
        ttest = ttest_ind(pvp_accuracy.mean(axis=1), pvp_shuffle_accs.mean(axis=1),
                          alternative='greater', equal_var=False)
        print(f'{dec_sffx} ttest: pval =  {ttest[1]}. Mean accuracy of {pips} is {pvp_accuracy.mean():.3f}.')

    # outlier_sessions
    rule_dec_tag = "_vs_".join(pips2decode[-1])
    rule_dec_accs = pd.DataFrame({sess_dec_name: sess_dec['data'].accuracy for sess_dec_name, sess_dec in
                                  pvp_dec_dict.items() if rule_dec_tag in sess_dec_name
                                  and any([e in sess_dec_name for e in hipp_animals])}).T
    rule_dec_accs['shuffled'] = pd.DataFrame({sess_dec_name: sess_dec['shuffled'].accuracy
                                              for sess_dec_name, sess_dec in pvp_dec_dict.items()
                                              if rule_dec_tag in sess_dec_name
                                              and any([e in sess_dec_name for e in hipp_animals])}
                                             ).T

    cmap_norm_4way = TwoSlopeNorm(vmin=0.15, vcenter=0.25, vmax=0.35)
    all_v_all_tag = "_vs_".join(pips2decode[-1])
    for animal in animals:
        all_cms = np.array([pvp_dec_dict[dec_name]['data'].cm for dec_name in pvp_dec_dict if all_v_all_tag in dec_name
                            and animal in dec_name])
        cm_plot = plot_aggr_cm(all_cms,im_kwargs=dict(norm=cmap_norm_4way),include_values=True,
                               labels=['A-0','A-2','A-1','A-3'], cmap='bwr')

        cm_plot[0].set_size_inches((3, 3))
        # cm_plot[0].show()
        cm_plot[0].savefig(aggr_abstr_figdir / f'{all_v_all_tag}_{all_v_all_tag}.pdf')

        all_accs_plot = plt.subplots()
        all_accs_by_rule = [np.array([pvp_dec_dict[dec_name]['data'].accuracy[0]
                                      for dec_name in pvp_dec_dict if '_vs_'.join(pips) in dec_name
                            and animal in dec_name]) for pips in pips2decode[:2]+pips2decode[-1:]]
        all_shuffs_by_rule = [np.array([pvp_dec_dict[dec_name]['shuffled'].accuracy[0]
                                        for dec_name in pvp_dec_dict if '_vs_'.join(pips) in dec_name
                                       and animal in dec_name]) for pips in pips2decode[:2]+pips2decode[-1:]]
        accs_w_shuffle = list(zip(all_accs_by_rule,all_shuffs_by_rule))
        all_accs_plot[1].boxplot(sum([[accs[0],accs[1]] for accs in accs_w_shuffle],[]), labels=None)
        all_accs_plot[1].set_ylabel('Accuracy')
        format_axis(all_accs_plot[1],ylim=[0.1,0.8],hlines=[.5,.25],)
        all_accs_plot[0].set_size_inches((3, 3))
        all_accs_plot[0].set_layout_engine('tight')
        all_accs_plot[0].show()
        all_accs_plot[0].savefig(aggr_abstr_figdir / f'accs_by_dec_{animal}.pdf')

        [print(f'{animal}, {tag} ttest pval: {ttest_ind(all_accs_by_rule[tag_ix], all_shuffs_by_rule[tag_ix],alternative="greater", equal_var=False)[1]}')
         for tag_ix,tag in enumerate(['rule','group','sequence'])]

    rule_dec_eg_sess = 'DO79_240717a'
    eg_rule_cm_plot = plot_aggr_cm(pvp_dec_dict[f'{rule_dec_eg_sess}_{all_v_all_tag}']['data'].cm,  include_values=True,
                                   labels=['A-0','A-2','A-1','A-3'], cmap='bwr',plot_cbar=False)
    eg_rule_cm_plot[0].set_size_inches((3, 3))
    eg_rule_cm_plot[0].show()
    eg_rule_cm_plot[0].savefig(aggr_abstr_figdir / f'{rule_dec_eg_sess}_{all_v_all_tag}.pdf')

    group_dec_eg_sess = 'DO81_240717a'
    eg_group_cm_plot = plot_aggr_cm(pvp_dec_dict[f'{group_dec_eg_sess}_{all_v_all_tag}']['data'].cm, include_values=True,
                                    labels=['A-0', 'A-2', 'A-1', 'A-3'], cmap='bwr', plot_cbar=False)
    eg_group_cm_plot[0].set_size_inches((3, 3))
    eg_group_cm_plot[0].show()
    eg_group_cm_plot[0].savefig(aggr_abstr_figdir / f'{group_dec_eg_sess}_{all_v_all_tag}.pdf')

    event_resps_by_pip = [full_pattern_responses[rule_dec_eg_sess][e] for e in [f'A-{i}' for i in range(4)]]

    rule_dec_tag = "_vs_".join(pips2decode[0])
    rule_dec_accs_by_animal_plot = plt.subplots()
    rule_dec_accs_by_animal_plot[1].boxplot([[pvp_dec_dict[dec_name]['data'].accuracy[0] for dec_name in pvp_dec_dict
                                             if rule_dec_tag in dec_name and a in dec_name]
                                             for a in animals], labels=animals)
    rule_dec_accs_by_animal_plot[1].set_ylabel('Accuracy')
    format_axis(rule_dec_accs_by_animal_plot[1],hlines=[1/len(pips2decode[0])])
    rule_dec_accs_by_animal_plot[0].show()


    sim_mat = cosine_similarity([event_resps[:, :, -50:].mean(axis=-1).mean(axis=0)
                                 for event_resps in event_resps_by_pip])
    sim_mat_plot = plot_similarity_mat(sim_mat,[f'A-{i}' for i in range(4)],cmap='Reds')
    # sim_mat_plot = plot_sim_by_grouping(sim_mat,['ptype_i','group','position'],pip_desc,cmap='Reds')
    # sim_mat_plot = plot_sim_by_grouping(sim_mat,[''],pip_desc,cmap='Reds')
    sim_mat_plot[0].set_size_inches((16, 16))
    sim_mat_plot[0].show()

    rule_non_rule_sim_plot = plt.subplots()
    _sims = {e:within_prop_sim[ei][within_prop_sim[ei] != 1]
                             for ei, e in enumerate(event_names)}
    rule_non_rule_sim_plot[1].scatter()