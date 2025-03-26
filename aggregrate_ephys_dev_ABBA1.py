import argparse
import pickle
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import ttest_ind, ttest_1samp
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics.pairwise import cosine_similarity

from aggregate_ephys_funcs import *
from behviour_analysis_funcs import get_all_cond_filts
from ephys_analysis_funcs import posix_from_win, plot_2d_array_with_subplots, format_axis
from neural_similarity_funcs import plot_similarity_mat, compare_pip_sims_2way
from regression_funcs import run_glm

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

    session_topology_path = ceph_dir / posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_ephys_2401.csv')
    session_topology = pd.read_csv(session_topology_path)

    cond_filts = get_all_cond_filts()

    all_sess_info = session_topology.query('sess_order=="main" & date > 240219')
    all_sess_td_df = load_aggregate_td_df(all_sess_info,home_dir,cond_filts['dev_ABBA1'])
    sessions2use = all_sess_td_df.index.get_level_values('sess').unique()
    # sessnames = [Path(sess_info['sound_bin']).stem.replace('_SoundData', '') for _, sess_info in
    #              sessions2use.iterrows()]

    pkldir = ceph_dir / posix_from_win(args.pkldir)
    pkls = list(pkldir.glob('*pkl'))

    sess_ephys_dict_path = Path(r'D:\ephys')/'aggr_ABCD_vs_ABAD_dict.pkl'
    ow_flag = False
    if sess_ephys_dict_path.is_file() and not ow_flag:
        with open(sess_ephys_dict_path, 'rb') as f:
            sessions = pickle.load(f)
    else:
        pkls2load = [pkl for pkl in pkls if Path(pkl).stem in sessions2use]
        sessions = load_aggregate_sessions(pkls2load,)

        with open(sess_ephys_dict_path, 'wb') as f:
            pickle.dump(sessions, f)
    # exit()


    # sessions = load_aggregate_sessions(pkls2load,)

    hipp_animals = ['DO79','DO81']

    # [sess for sess in sessions2use if not any([sess in Path(pkl).stem for pkl in pkls2load])]

    [sessions.pop(sess) for sess in list(sessions.keys()) if 'A-1' not in list(sessions[sess].sound_event_dict.keys())]
    [print(sess,sessions[sess].sound_event_dict.keys()) for sess in list(sessions.keys()) ]

    # update styles
    plt.style.use('figure_stylesheet.mplstyle')
    window = (-0.1, 0.25)
    event_responses = aggregate_event_reponses(sessions, events=None,  # [f'{pip}-0' for pip in 'ABCD']
                                               events2exclude=['trial_start','base', 'X'], window=window,
                                               pred_from_psth_kwargs={'use_unit_zscore': True, 'use_iti_zscore': False,
                                                                      'baseline': 0, 'mean': None, 'mean_axis': 0})
    concatenated_event_responses = {
        e: np.concatenate([event_responses[sessname][e].mean(axis=0) for sessname in event_responses])
        for e in list(event_responses.values())[0].keys()}

    sessnames = list(event_responses.keys())
    event_features = aggregate_event_features(sessions, events=[e for e in concatenated_event_responses.keys()
                                                                        if 'A' in e],
                                              events2exclude=['trial_start'])

    # do some decoding
    event_responses_by_features_by_sess = {
        e: [event_responses[sessname][e] for sessname in event_responses]
        for e in list(event_responses.values())[0].keys()}

    full_pattern_responses = aggregate_event_reponses(sessions, events=[e for e in concatenated_event_responses.keys()
                                                                        if 'A' in e],
                                                      events2exclude=['trial_start'], window=[-1, 2],
                                                      pred_from_psth_kwargs={'use_unit_zscore': True,
                                                                             'use_iti_zscore': False,
                                                                             'baseline': 0, 'mean': None,
                                                                             'mean_axis': 0})

    concatenated_full_pattern_responses = {
        e: np.concatenate([full_pattern_responses[sessname][e].mean(axis=0) for sessname in full_pattern_responses])
        for e in [f'A-{i}' for i in range(3)]}
    concatenated_event_responses_by_ptype = {
        'ABCD': np.concatenate([np.concatenate([full_pattern_responses[sessname][f'A-0'][100:115],
                                                full_pattern_responses[sessname][f'A-1']],axis=0).mean(axis=0)
                                for sessname in full_pattern_responses]),
        'ABBA': concatenated_full_pattern_responses[f'A-2']
    }
    concatenated_event_responses_by_group = {
        '1': np.concatenate([np.concatenate([full_pattern_responses[sessname][f'A-1'],
                                                full_pattern_responses[sessname][f'A-2']],axis=0).mean(axis=0)
                                for sessname in full_pattern_responses]),
        '0': np.concatenate([full_pattern_responses[sessname][f'A-0'][100:115].mean(axis=0)
                                for sessname in full_pattern_responses]),
    }

    events_by_property = {
        'ptype_i': {pip: 0 if int(pip.split('-')[1]) < 2 else 1 for pip in [f'{p}-{i}' for i in range(3) for p in 'ABCD']},
    }

    pip_desc = sessions[list(sessions.keys())[0]].pip_desc
    events_by_property['id'] = {pip: pip_desc[pip]['idx']
                                for pip in list(events_by_property['ptype_i'])
                                if pip.split('-')[0] in 'ABCD'}
    events_by_property['position'] = {pip: ord(pip.split('-')[0]) - ord('A') + 1
                                      for pip in list(events_by_property['ptype_i'])
                                      if pip.split('-')[0] in 'ABCD'}
    events_by_property['group'] = {pip: 0 if int(pip.split('-')[1]) <1 else 1 for pip in list(events_by_property['ptype_i'])}
    events_by_property['pattern_i'] = {pip: pip.split('-')[1] for pip in list(events_by_property['ptype_i'])}

    concatenated_full_pattern_by_pip_prop = group_responses_by_pip_prop(concatenated_full_pattern_responses,
                                                                        events_by_property, ['ptype_i'])

    # indv_pips pca
    indv_pip_pca_figdir = ceph_dir / 'Dammy' / 'figures' / 'indv_pip_pca_aggregate_sessions_normdev_abstraction'
    if not indv_pip_pca_figdir.is_dir():
        indv_pip_pca_figdir.mkdir()
    dev_ABBA1_figdir = ceph_dir / 'Dammy' / 'figures' / 'dev_ABBA1_aggregate_sessions'
    if not dev_ABBA1_figdir.is_dir():
        dev_ABBA1_figdir.mkdir()

    concatenated_full_pattern_by_pip_prop = group_responses_by_pip_prop(concatenated_full_pattern_responses,
                                                                        events_by_property, ['ptype_i','group',
                                                                                             'pattern_i'],
                                                                        concatenate_flag=False)
    #
    # concatenated_full_pattern_by_pip_prop = group_responses_by_pip_prop(concatenated_full_pattern_responses,
    #                                                                     events_by_property, ['ptype_i',''],
    #                                                                     concatenate_flag=False)
    dict_4_pca = {'by_type': {
        'normal': concatenated_full_pattern_responses['A-0'],
        'dev_ABCD': concatenated_full_pattern_responses['A-1'],
        'dev_ABBA': concatenated_full_pattern_responses['A-2']
    }}

    # full_pattern_pca = PopPCA({'by_group':concatenated_event_responses_by_group})
    full_pattern_pca = PopPCA(dict_4_pca)
    # full_pattern_pca = PopPCA(concatenated_full_pattern_by_pip_prop['group'])
    # full_pattern_pca.eig_vals[2][0].show()
    full_pattern_pca.get_trial_averaged_pca(standardise=True)
    full_pattern_pca.get_projected_pca_ts(standardise=True)
    full_pattern_pca.plot_pca_ts([-1, 2], fig_kwargs={'figsize': (120, 8)}, plot_separately=False, n_comp_toplot=15,
                                 lss=['-', '--', '-', '--'], plt_cols=['C' + str(i) for i in [0, 0, 1, 1]]
                                 )
    # plot 3d projections
    x_ser = np.round(np.linspace(-1, 2, concatenated_full_pattern_responses['A-0'].shape[-1]), 2)
    full_pattern_pca.plot_2d_pca_ts('by_pattern', [-1, 2], x_ser=x_ser, smoothing=3, pca_comps_2plot=[1,4], t_end=1,
                                    plot_out_event=False,scatter_times=[0.5],scatter_kwargs={'marker':'*','s':50,'c':'k'})
    full_pattern_pca.plot_3d_pca_ts('by_type', [-1, 2], x_ser=x_ser, smoothing=10, pca_comps_2plot=[1,2,4][::1], t_end=1,
                                    scatter_times=[0.65],scatter_kwargs={'marker':'*','s':50,'c':'k'},
                                    plot_out_event=True)

    # get [0,1,2] components can compute euc distance over time
    pca_ts_0_1_2 = {e: full_pattern_pca.projected_pca_ts_by_cond['by_type'][e][[1,2,3]]
                    for e in full_pattern_pca.projected_pca_ts_by_cond['by_type'].keys()}
    euc_dist = np.linalg.norm(pca_ts_0_1_2['normal'] - pca_ts_0_1_2['deviant'], axis=0)
    euc_dist_plot = plt.subplots()
    # causally smooth euc dist
    euc_dist = np.convolve(euc_dist, np.ones(25) / 25, mode='same')
    euc_dist_plot[1].plot(x_ser, euc_dist)
    format_axis(euc_dist_plot[1], vspan=[[t, t + 0.15] for t in np.arange(0, 1, 0.25)],
                xlim=[-0.75, 1.75], xticks=np.arange(0, 1, 0.25), xticklabels=np.arange(0, 1, 0.25))
    euc_dist_plot[1].set_title('Euclidean distance between rare and frequent')
    euc_dist_plot[1].set_xlabel('Time')
    euc_dist_plot[1].set_ylabel('Euclidean distance')
    # euc_dist_plot[0].set_layout_engine('tight')
    euc_dist_plot[0].show()

    # events = [f'{pip}-{i}' for i in range(4) for pip in 'ABCD']
    # resps2use = aggregate_event_reponses(sessions,
    #                                            events2exclude=['trial_start'], window=window,
    #                                            pred_from_psth_kwargs={'use_unit_zscore': True, 'use_iti_zscore': False,
    #                                                                   'baseline': 0, 'mean': None, 'mean_axis': 0})
    events = [e for e in concatenated_event_responses.keys() if 'A' in e]
    resps2use = full_pattern_responses

    # resps2use = full_pattern_responses
    abstraction_figdir = ceph_dir / 'Dammy' / 'figures' / 'abstraction_figs'
    if not abstraction_figdir.is_dir():
        abstraction_figdir.mkdir()

    window2use= [-0.25, 1]
    # window2use= [-0.1, 0.25]
    x_ser = np.round(np.linspace(*window2use, resps2use[list(resps2use.keys())[0]]['A-0'].shape[-1]),2)
    # ts_s = np.arange(window2use[0], window2use[1], 0.1)
    # ts_s = np.round(np.arange(window2use[0], window2use[1], 0.1),1)
    # window2use= np.arange(-0.2, 1.1, 0.1)
    ts_s = [-0.2,0,0.2,0.4,0.6,0.8,1.0]
    # ts_s = [0.25]
    # ts_s = [0.5]
    ts_idx = [np.where(x_ser==tt)[0][0] for tt in ts_s if tt in x_ser]
    # ts_idx = [0,-1]
    prop = 'ptype_i'
    mi_by_t_by_pip = {}
    for pip in 'A':
        mi_by_t = {}
        for t_idx in ts_idx:
            mi_by_sess = {}
            for sessname in tqdm(list(resps2use.keys()), total=len(resps2use), desc='computing MI'):
                n_units= resps2use[sessname][events[0]].shape[1]
                # all_responses = [np.squeeze(resps2use[sessname][e][:,:,t_idx]) for e in events+['X','base'] ]  #
                # resp_min = np.min([np.min(r[r>0]) for r in all_responses])
                # resp_max = np.max([np.max(r[r>0]) for r in all_responses])
                # bins = np.linspace(resp_min, resp_max, 100)
                # prop_mi = {}

                rule_lbls = np.unique(list(events_by_property[prop].values()))
                prop_events = [[e for e in events if events_by_property[prop][e] == prop_val if any(p in e for p in pip)]
                               for prop_val in rule_lbls]
                # prop_events = [['A-1'],['A-2']]
                cond_responses = [[np.nanmean(np.squeeze(resps2use[sessname][e])[-40:, :,t_idx-10:t_idx], axis=-1)
                                   for e in rule_events]
                                  for rule_events in prop_events]
                responses_by_rule = {rule: np.vstack([e for e in responses_to_stims if len(e) > 0])
                                     for responses_to_stims, rule in zip(cond_responses, rule_lbls)
                                     if len(responses_to_stims) > 0}
                xs = [np.hstack([rule_resps[:, unit]
                                 for rule_i, rule_resps in enumerate(responses_by_rule.values())])
                      for unit in range(n_units)]
                # xs = [np.hstack([(np.full_like(rule_resps[:,unit], rule_i +0.1 ) if t_idx==35 else
                #                   np.random.uniform(size=rule_resps[:,unit].shape[0]))
                #                  for rule_i,rule_resps in enumerate(responses_by_rule.values())])
                #       for unit in range(n_units)]
                ys = [np.hstack([np.full(rule_resps[:,unit].shape[0],rule) for rule, (_,rule_resps) in enumerate(responses_by_rule.items())])
                      for unit in range(n_units)]
                xs_nonzero_idx = [np.where(x>1e-1)[0] for x in xs]
                xs_nonzero = [x[x_idx] for x,x_idx in zip(xs,xs_nonzero_idx) if len(x[x_idx])>10]
                ys_nonzero = [y[x_idx] for y,x_idx in zip(ys,xs_nonzero_idx) if len(y[x_idx])>10]
                mi_by_unit = [mutual_info_classif(xs_nonzero[unit].reshape(-1,1),ys_nonzero[unit],
                                                  discrete_features=[False],n_neighbors=3)[0]
                              for unit in range(len(xs_nonzero))]
                # mi_by_unit = [comp]
                mi_by_sess[sessname] = mi_by_unit

                # mi_by_unit = [run_glm(xs_nonzero[unit].reshape(-1, 1), ys_nonzero[unit].reshape(-1, 1))
                #               for unit in range(n_units)]
                #
                # mi_by_sess[sessname] = [glm[1].params[0] for glm in mi_by_unit if glm[1].pvalues[0] < 0.05]
            mi_by_t[x_ser[t_idx]] = mi_by_sess
        mi_by_t_by_pip[pip] = mi_by_t

    ts2plot = [0.0,0.8]
    mi_dist_plot = plt.subplots(len(mi_by_t_by_pip),sharey=True, sharex=True, figsize=(6, 4*len(mi_by_t_by_pip)),squeeze=False)
    for (pip,mi_by_t),plot in zip(mi_by_t_by_pip.items(),mi_dist_plot[1].flatten()):
        for t_lbl,prop_mi in mi_by_t.items():
            if t_lbl not in ts2plot:
                continue
            mi_across_sessions = np.hstack(list(prop_mi.values()))
            plot.hist(mi_across_sessions[mi_across_sessions>-1],bins='fd',alpha=0.2,label=t_lbl,density=True)
            print(f'{pip}, {t_lbl}, {mi_across_sessions.shape}, {np.percentile(mi_across_sessions,90)}, {np.mean(mi_across_sessions)}')
        plot.legend()
        plot.set_ylabel('')

        plot.set_title(f'MI distribution for {prop}')
        # [print(pip,np.nanpercentile(np.hstack(list(prop_mi.values())),90)) for prop_mi, t_lbl in zip(mi_by_t, ts_s)]
        # [print(pip,np.nanmean(np.hstack(list(prop_mi.values())))) for prop_mi, t_lbl in zip(mi_by_t, ts_s)]
    mi_dist_plot[0].show()
    print(pip,ttest_ind(np.hstack(list(mi_by_t_by_pip['A'][1.0].values())),
                        np.hstack(list(mi_by_t_by_pip['A'][.0].values())),
                        alternative='greater'))

    # decode events
    pips_as_ints = {pip: pip_i for pip_i, pip in enumerate([f'{pip}-{pip_i}' for pip in 'ABCD' for pip_i in range(3)])}
    # train on A decode on B C D
    sessname = list(resps2use.keys())[0]
    decoders_dict = {}
    patt_is = [1, 2]
    bad_dec_sess = set()
    for sessname in tqdm(list(resps2use.keys()), total=len(resps2use), desc='decoding across sessions'):
        for p in 'ABCD':
            xys = [(event_responses[sessname][f'{p}-{pip_i}'][-25:],
                    np.full_like(event_responses[sessname][f'{p}-{pip_i}'][:,0,0], pips_as_ints[f'{p}-{pip_i}'])[-25:])
                   for pip_i in patt_is]
            xs = np.vstack([xy[0][:,:,-10:].mean(axis=-1) for xy in xys])
            ys = np.hstack([xy[1] for xy in xys])
            if np.unique(ys).shape[0] < len(patt_is):
                continue
            try:
                decoders_dict[f'{sessname}-{p}s'] = decode_responses(xs, ys,n_runs=100)
            except ValueError:
                print(f'{sessname}-{p}s failed')
                bad_dec_sess.add(sessname)
                continue
    [decoders_dict[dec_name]['data'].plot_confusion_matrix([f'{pip}-{pip_i}' for pip in 'D' for pip_i in patt_is])
     for dec_name in decoders_dict.keys()]
    # [decoders_dict[dec_name]['data'].cm_plot[0].show()
    #  for dec_name in decoders_dict.keys()]

    [decoders_dict.pop(dec_name) for dec_name in [k for k in decoders_dict.keys() for sess in bad_dec_sess if
                                                  k.startswith(sess)]
     if dec_name in decoders_dict.keys()]
    all_cms_by_pip =[[dec['data'].cm for dec_name, dec in decoders_dict.items() if dec_name.endswith(f'{pip}s') and
                      dec['data'].cm is not None and any(e in dec_name for e in ['DO79','DO81'])]
                     for pip in 'BD']
    all_cms_by_pip_arr = np.array(all_cms_by_pip)
    all_cms_by_pip_plot = plt.subplots(ncols=len(all_cms_by_pip))
    cmap_norm = TwoSlopeNorm(vmin=np.quantile(all_cms_by_pip_arr,0.1), vmax=np.quantile(all_cms_by_pip_arr,0.75),
                             vcenter=1/all_cms_by_pip_arr.shape[-1])
    [plot_2d_array_with_subplots(np.mean(pip_cms,axis=0),plot=(all_cms_by_pip_plot[0],all_cms_by_pip_plot[1][pip_i]),
                                 cmap='bwr', norm=cmap_norm)
     for pip_i, pip_cms in enumerate(all_cms_by_pip)]
    [ax.invert_yaxis() for ax in all_cms_by_pip_plot[1]]
    all_cms_by_pip_plot[0].set_size_inches(10, 6)
    all_cms_by_pip_plot[0].show()
    # all_cms_by_pip_plot[0].savefig(abstraction_figdir / 'decoding_cms_across_pips_dev_ABBA1.svg')
    dec_acc_plot = plt.subplots()
    accuracy_across_sessions = [np.diagonal(pip_cm,axis1=1, axis2=2) for pip_cm in all_cms_by_pip_arr]
    accuracy_across_sessions_dfs = [pd.DataFrame(pip_accs,index=list([sess for sess in resps2use.keys()
                                                                      if sess not in bad_dec_sess
                                                                      and any(e in sess for e in ['DO79','DO81'])]),)
                                    for pip_accs in accuracy_across_sessions]
    accuracy_across_sessions_df = pd.concat(accuracy_across_sessions_dfs, axis=1)
    [print(ttest_1samp(acc, 1/all_cms_by_pip_arr.shape[-1], alternative='greater')) for acc in accuracy_across_sessions]
    print(ttest_ind(accuracy_across_sessions[0][0],accuracy_across_sessions[1][0], alternative='two-sided'))
    print(ttest_ind(accuracy_across_sessions[0][1],accuracy_across_sessions[1][1], alternative='two-sided'))

    [dec_acc_plot[1].boxplot(pip_accs, positions=np.arange(pip_i, pip_i+pip_accs.shape[1]*len(accuracy_across_sessions),
                                                          len(accuracy_across_sessions)),
                             patch_artist=False, labels=[f'{pip}-{pip_i}' for pip_i in patt_is],
                             showmeans=True, meanprops=dict(mfc='k'), showfliers=False,
                             medianprops=dict(color=f'C{pip_i}'))
     for pip_i,(pip,pip_accs) in enumerate(zip('AD',accuracy_across_sessions))]
    dec_acc_plot[1].axhline(1/all_cms_by_pip_arr.shape[-1], color='k',ls='--',lw=1)

    dec_acc_plot[0].set_layout_engine('tight')
    dec_acc_plot[0].show()
    # dec_acc_plot[0].savefig(abstraction_figdir / 'decoding_across_sessions_dev_ABBA1.svg')

    for lbl, decoder in decoders_dict.items():
        if lbl not in [f'DO81_240924a-{p}s' for p in 'BD']:
            continue
        decoders_dict[lbl]['data'].cm_plot[0].show()
        # plot accuracy

        decoder_acc_plot = plt.subplots()
        decoder_acc_plot[1].boxplot([np.array(dec.accuracy) for dec in decoder.values()],
                                    labels=decoder.keys(),showmeans=True, meanprops=dict(mfc='k'),)
        decoder_acc_plot[1].set_title(f'Accuracy of {lbl}')
        decoder_acc_plot[0].set_layout_engine('tight')
        decoder_acc_plot[0].show()
        # decoder_acc_plot[0].savefig(abstraction_figdir / f'{lbl}_decoding_accuracy_dev_ABBA1.svg')

        # ttest on fold accuracy
        ttest = ttest_ind(*[np.array(dec.fold_accuracy).flatten() for dec in decoder.values()],
                          alternative='greater', equal_var=True)
        print(f'{lbl} ttest: {ttest}')

    # decode devs seperately
    # get norms 1 before dev
    norm_idxs_by_dev_pip = {}
    for dev_pip in ['A-1', 'A-2']:
        norm_idxs_by_dev_pip[dev_pip] = {sess: np.isin(event_features[sess]['A-0']['trial_nums'],
                                                       event_features[sess][dev_pip]['trial_nums']-2)
                                         for sess in event_features.keys()}
    pvp_dec_dict = {}
    pips2decode = [['A-0', 'A-1'], ['A-0', 'A-2'], ['A-1', 'A-2'],['A-0','A-0'],['A-0','A-1','A-2'],['A-0','A-1;A-2']]
    resp_x_ser = np.round(np.linspace(-0.25,1,full_pattern_responses[sessname][pips2decode[0][0]].shape[-1]),2)
    window_s = [0.8,1.0]
    window_idxs = np.logical_and(resp_x_ser >= window_s[0], resp_x_ser <= window_s[1])
    for pips in pips2decode[:-1]:
        dec_sffx = "_vs_".join(pips)
        for sessname in tqdm(full_pattern_responses.keys(), desc='decoding',
                             total=len(full_pattern_responses.keys())):
            xs_list = []
            if pips[0] == pips[1]:
                # split in half
                xs_list = np.array_split(full_pattern_responses[sessname][pips[0]][30:90],2)
            else:
                for pip in pips:
                    norm_idxs = np.logical_or(*[norm_idxs_by_dev_pip[pip][sessname] for pip in ['A-1', 'A-2']])
                    if pip == 'A-0' and any([p in sum(pips2decode,[]) for p in ['A-1', 'A-2']]):
                        # xs_list.append(full_pattern_responses[sessname][pip][norm_idxs])
                        xs_list.append(full_pattern_responses[sessname][pip][100:300][::15])
                    else:
                        xs_list.append(full_pattern_responses[sessname][pip])
                        # xs_list.append(full_pattern_responses[sessname]['A-0'][norm_idxs])
                if any([x.shape[0]<8 for x in xs_list]):
                    continue
            xs = np.vstack([x[:, :, window_idxs].max(axis=-1) for x in xs_list])
            ys = np.hstack([np.full(x.shape[0], ci) for ci, x in enumerate(xs_list)])
            pvp_dec_dict[f'{sessname}_{dec_sffx}'] = decode_responses(xs, ys, n_runs=20,dec_kwargs={'cv_folds':4})
            pvp_dec_dict[f'{sessname}_{dec_sffx}']['data'].plot_confusion_matrix(dec_sffx.split('_vs_'),)
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
        pvp_accuracy_plot[1].set_ylabel('Accuracy')
        pvp_accuracy_plot[1].set_title(f'Accuracy of {dec_sffx}')
        format_axis(pvp_accuracy_plot[1],hlines=[0.5])
        pvp_accuracy_plot[0].show()
        # pvp_accuracy_plot[0].savefig(dev_ABBA1_figdir / f'{dec_sffx}_accuracy.pdf')
        # ttest
        ttest = ttest_ind(pvp_accuracy.mean(axis=1), pvp_shuffle_accs.mean(axis=1),
                          alternative='greater', equal_var=False)
        print(f'{dec_sffx} ttest: pval =  {ttest[1]}. Mean accuracy of {pips} is {pvp_accuracy.mean():.3f}.')

    cmap_norm_3way = TwoSlopeNorm(vmin=0.2, vcenter=0.33333, vmax=0.45)
    all_v_all_tag = "_vs_".join(pips2decode[-2])
    all_cms = np.array([pvp_dec_dict[dec_name]['data'].cm for dec_name in pvp_dec_dict if all_v_all_tag in dec_name
                        and any([e in dec_name for e in hipp_animals])])
    cm_plot = plot_aggr_cm(all_cms, im_kwargs=dict(norm=cmap_norm_3way), include_values=True,
                           labels=['A-0','A-1', 'A-2'], cmap='bwr')

    cm_plot[0].set_size_inches((3, 3))
    cm_plot[0].show()

    for a_sub in [['DO81', 'DO79'],['DO79'],['DO81'],['DO82']]:
        cmap_norm_3way = TwoSlopeNorm(vmin=0.2, vcenter=0.33333, vmax=0.45)
        all_v_all_tag = "_vs_".join(pips2decode[-2])
        all_cms = np.array([pvp_dec_dict[dec_name]['data'].cm for dec_name in pvp_dec_dict if all_v_all_tag in dec_name
                            and any([e in dec_name for e in a_sub])])
        cm_plot = plot_aggr_cm(all_cms, im_kwargs=dict(norm=cmap_norm_3way), include_values=True,
                               labels=['A-0', 'A-1', 'A-2'], cmap='bwr')
        cm_plot[1].set_title(f'{a_sub}')

        cm_plot[0].set_size_inches((3, 3))
        cm_plot[0].show()

    # decode over time
    dec_over_time_window = [-0.5, 1.5]
    window_size = 0.25
    full_pattern_responses_4_ts_dec = aggregate_event_reponses(sessions,
                                                               events=[e for e in concatenated_event_responses.keys()
                                                                       if 'A' in e],
                                                               events2exclude=['trial_start'],
                                                               window=dec_over_time_window,
                                                               pred_from_psth_kwargs={'use_unit_zscore': True,
                                                                                      'use_iti_zscore': False,
                                                                                      'baseline': 0, 'mean': None,
                                                                                      'mean_axis': 0})
    good_ABBA1_sess_resps = {sessname: resps for sessname,resps in full_pattern_responses_4_ts_dec.items()
                             if all([resps[pip].shape[0]>=9 for pip in ['A-1', 'A-2']])
                             and any(e in sessname for e in ['DO81', 'DO79'])}

    resp_width = full_pattern_responses_4_ts_dec[sessname]['A-0'].shape[-1]
    resp_x_ser = np.linspace(dec_over_time_window[0], dec_over_time_window[1], resp_width)

    ABCDvsABBA1_dec_over_t, ABCDvsABBA1_dec_over_t_dict = decode_over_sliding_t(good_ABBA1_sess_resps,
                                                                                window_size,
                                                                                dec_over_time_window,
                                                                                pips_as_ints, ['A-1', 'A-2'],
                                                                                animals_to_use=['DO79', 'DO81'])
    dec_over_time_plot = plt.subplots()
    dec_over_time_plot[1].plot(ABCDvsABBA1_dec_over_t.mean(axis=0), )
    dec_over_time_plot[1].fill_between(ABCDvsABBA1_dec_over_t.columns.tolist(),
                                       ABCDvsABBA1_dec_over_t.mean(axis=0) - ABCDvsABBA1_dec_over_t.sem(axis=0),
                                       ABCDvsABBA1_dec_over_t.mean(axis=0) + ABCDvsABBA1_dec_over_t.sem(axis=0),
                                       alpha=0.2)

    dec_over_time_plot[1].set_xlabel('time (s)')
    dec_over_time_plot[1].set_ylabel('accuracy')
    format_axis(dec_over_time_plot[1], vspan=[[t, t + 0.15] for t in np.arange(0, 1, 0.25)])
    dec_over_time_plot[0].set_layout_engine('tight')
    dec_over_time_plot[0].show()

    # cosine sims across sessions
    event_names = list(pips_as_ints.keys())
    prop = 'ptype_i'
    non_prop = 'group'
    prop_sim_diff_by_sess = {}
    prop_sim_by_sess = {}
    self_sims_by_sess = {}
    for sessname in tqdm(list(event_responses.keys()), total=len(event_responses), desc='decoding across sessions'):
        event_resps_by_pip = [event_responses[sessname][e] for e in event_names]
        sim_mat = cosine_similarity([event_resps[:,:,-10:].mean(axis=-1).mean(axis=0)
                                     for event_resps in event_resps_by_pip])
        self_sims = [compare_pip_sims_2way([e_resps[:, :, -10:]], mean_flag=True, n_shuffles=50)[0].mean(axis=0)[0, 1]
                     for e_resps in event_resps_by_pip]
        self_sims_by_sess[sessname] = self_sims
        sim_mat_plot = plot_similarity_mat(sim_mat,event_names,cmap='Reds')
        sim_mat_plot[1].set_title(sessname)
        # sim_mat_plot[0].show()
        print(f'{sessname} mean sim',sim_mat[~np.eye(sim_mat.shape[0],dtype=bool)].reshape(sim_mat.shape[0],-1).mean())
        within_prop_idxs = [[events_by_property[prop][ee] == events_by_property[prop][e] for ee in event_names]
                            for e in event_names]
        within_prop_sim = [sim_mat[ei][e_idxs] for ei, e_idxs in enumerate(within_prop_idxs)]
        within_prop_sim_means = {e:within_prop_sim[ei][within_prop_sim[ei] != 1].mean()
                                 for ei, e in enumerate(event_names)}
        non_prop_sim_idxs = [[events_by_property[non_prop][ee] == events_by_property[non_prop][e] for ee in event_names]
                             for e in event_names]
        # non_prop_sim_idxs = [np.invert(within_prop_idxs[ei]) for ei in range(len(within_prop_idxs))]
        non_prop_sim = [sim_mat[ei][e_idxs] for ei, e_idxs in enumerate(non_prop_sim_idxs)]
        non_prop_sim_means = {e:non_prop_sim[ei][non_prop_sim[ei] != 1].mean() for ei, e in enumerate(event_names)}
        prop_sim_diff = {e:within_prop_sim_means[e] - non_prop_sim_means[e] for e in event_names}
        prop_sim_diff_by_sess[sessname] = prop_sim_diff
        prop_sim_by_sess[sessname] = (within_prop_sim_means, non_prop_sim_means)
        # within_prop_sim = np.array([np.mean(e) for e in within_prop_sim])
        # print(f'{sessname}: {ttest_1samp(within_prop_sim, 0, alternative="greater")}')

    prop_sim_diff_plot = plt.subplots()
    # prop_sim_diff_plot[1].boxplot([[prop_sim_diff_by_sess[sess][e] for sess in prop_sim_diff_by_sess.keys()]
    #                                for e in event_names], labels=event_names,showfliers=True)
    prop_sim_diff_plot[1].boxplot([[prop_sim_diff_by_sess[sess][e] for sess in prop_sim_diff_by_sess.keys()]
                                   for e in [f'{p}-1' for p in 'ABCD']], labels=[f'{p}-1' for p in 'ABCD'],
                                  showfliers=True)
    prop_sim_diff_plot[1].set_title(f'Similarity difference of {prop} vs {non_prop}')
    prop_sim_diff_plot[1].axhline(0, color='k', ls='--')
    prop_sim_diff_plot[0].set_layout_engine('tight')
    prop_sim_diff_plot[0].show()
    # prop_sim_diff_plot[0].savefig(abstraction_figdir / f'{prop}_vs_{non_prop}_sim_diff_plot.svg')

    pip_sim_diff = [[prop_sim_diff_by_sess[sess][e] for sess in prop_sim_diff_by_sess.keys()]
                                   for e in [f'{p}-1' for p in 'ABCD']]
    [print(f'{e}: {ttest_1samp(pip_sim_diff[ei], 0, alternative="greater")}') for ei, e in enumerate([f'{p}-1' for p in 'ABCD'])]
