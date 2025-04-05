import argparse
import pickle
import platform
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import ttest_ind, ttest_1samp, sem, tukey_hsd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics.pairwise import cosine_similarity


from aggregate_ephys_funcs import *
from behviour_analysis_funcs import get_all_cond_filts, get_n_since_last, get_prate_block_num, get_cumsum_columns, \
    add_datetimecol, get_lick_in_patt_trials, get_earlyX_trials, get_last_pattern
from ephys_analysis_funcs import posix_from_win, plot_2d_array_with_subplots, plot_psth, format_axis, plot_sorted_psth
from neural_similarity_funcs import plot_similarity_mat
from regression_funcs import run_glm
from unit_analysis import get_participation_rate

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

    cond_filters = get_all_cond_filts()

    all_sess_info = session_topology.query('sess_order=="main" & date >= 240219')
    all_sess_td_df = load_aggregate_td_df(all_sess_info,home_dir,'Stage==3')
    sessions2use = all_sess_td_df.index.get_level_values('sess').unique()
    # sessnames = [Path(sess_info['sound_bin']).stem.replace('_SoundData', '') for _, sess_info in
    #              sessions2use.iterrows()]

    pkldir = ceph_dir / posix_from_win(args.pkldir)
    pkls = list(pkldir.glob('*pkl'))
    fam_aggr_sess_pkl_path = Path(r'D:\ephys')/'fam_aggr_sessions_event_free_fix.pkl'
    ow_flag = False
    if fam_aggr_sess_pkl_path.is_file() and not ow_flag:
        with open(fam_aggr_sess_pkl_path, 'rb') as f:
            sessions = pickle.load(f)
    else:
        print('loading sessions from previous pkl...')
        pkls2load = [pkl for pkl in pkls if Path(pkl).stem in sessions2use]
        sessions = load_aggregate_sessions(pkls2load,)

        with open(fam_aggr_sess_pkl_path, 'wb') as f:
            pickle.dump(sessions, f)

    # [sess for sess in sessions2use if not any([sess in Path(pkl).stem for pkl in pkls2load])]

    [sessions.pop(sess) for sess in list(sessions.keys()) if 'A-0' not in list(sessions[sess].sound_event_dict.keys())]
    # [sessions.pop(sess) for sess in list(sessions.keys()) if 'D-1' not in list(sessions[sess].sound_event_dict.keys())]
    # [sessions.pop(sess) for sess in list(sessions.keys()) if 'A-2' in list(sessions[sess].sound_event_dict.keys())]
    [print(sess,sessions[sess].sound_event_dict.keys()) for sess in list(sessions.keys()) ]

    # update styles
    plt.style.use('figure_stylesheet.mplstyle')
    matplotlib.rcParams['figure.figsize'] = (4,3)

    aggr_figdir = ceph_dir / 'Dammy' / 'figures' / 'fam_aggr_plots'
    if not aggr_figdir.is_dir():
        aggr_figdir.mkdir()

    window = (-0.1, 0.25)
    hipp_animals = ['DO79','DO81']
    event_responses = aggregate_event_reponses(sessions, events=None,  # [f'{pip}-0' for pip in 'ABCD']
                                               events2exclude=['trial_start',], window=window,
                                               pred_from_psth_kwargs={'use_unit_zscore': True, 'use_iti_zscore': False,
                                                                      'baseline': 0, 'mean': None, 'mean_axis': 0})
    concatenated_event_responses = {
        e: np.concatenate([event_responses[sessname][e].mean(axis=0) for sessname in event_responses])
        for e in list(event_responses.values())[0].keys()}

    concatenated_event_responses_hipp_only = {
        e: np.concatenate([event_responses[sessname][e].mean(axis=0) for sessname in event_responses
                           if any([animal in sessname for animal in hipp_animals+['DO80']])])
        for e in list(event_responses.values())[0].keys()}

    sessnames = list(event_responses.keys())
    event_features = aggregate_event_features(sessions, events=[f'{pip}-0' for pip in 'ABCD']+['X'],
                                              events2exclude=['trial_start'])

    # some psth plots
    psth_figdir = aggr_figdir.parent / 'psth_plots_aggr_sessions'
    if not psth_figdir.is_dir():
        psth_figdir.mkdir()
    full_patt_window = (-1, 2)
    full_pattern_responses_4_psth = aggregate_event_reponses(sessions, events=[e for e in concatenated_event_responses.keys()
                                                                        if 'A' in e],
                                                      events2exclude=['trial_start'], window=full_patt_window,
                                                      pred_from_psth_kwargs={'use_unit_zscore': True,
                                                                             'use_iti_zscore': False,
                                                                             'baseline': 0, 'mean': None,
                                                                             'mean_axis': 0})
    concat_full_patt_resps = {
        e: np.concatenate([full_pattern_responses_4_psth[sessname][e].mean(axis=0)
                           for sessname in full_pattern_responses_4_psth])
        for e in [e for e in concatenated_event_responses.keys() if 'A' in e]}

    event_responses_4_psth = aggregate_event_reponses(sessions, events=None,  # [f'{pip}-0' for pip in 'ABCD']
                                               events2exclude=None, window=[-0.25,1],
                                               pred_from_psth_kwargs={'use_unit_zscore': True, 'use_iti_zscore': False,
                                                                      'baseline': 0, 'mean': None, 'mean_axis': 0})
    concatenated_event_responses_4_psth = {
        e: np.concatenate([event_responses_4_psth[sessname][e].mean(axis=0) for sessname in event_responses_4_psth])
        for e in list(event_responses_4_psth.values())[0].keys()
    }
    concatenated_sem = {
        e: np.concatenate([sem(event_responses_4_psth[sessname][e])
                           for sessname in event_responses_4_psth])
        for e in list(event_responses_4_psth.values())[0].keys()}


    psth_figdir = aggr_figdir.parent / 'psth_plots_aggr_sessions'
    for pip in [e for e in concatenated_event_responses.keys() if 'A' in e]:
        for animal in hipp_animals:
            cmap_norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
            all_resps_psth = plot_sorted_psth(full_pattern_responses_4_psth,pip,'A-0',window=[-0.25,1],sort_window=[0,1],
                                              sessname_filter=animal,im_kwargs=dict(norm=cmap_norm,cmap='bwr'))
            all_resps_psth[0].set_layout_engine('tight')
            format_axis(all_resps_psth[1][1],vlines=np.arange(0,1,0.25).tolist(),ylabel='Unit #',
                        xlabel=f'Time from {pip} onset (s)')
            format_axis(all_resps_psth[1][0],vlines=[0])
            all_resps_psth[1][0].set_ylim(-0.15,0.1)
            all_resps_psth[1][0].locator_params(axis='y', nbins=2)
            all_resps_psth[0].show()

            all_resps_psth[0].savefig(psth_figdir / f'{pip}_{animal}_A_resps_psth_aggr_fam_sessions.pdf')

    pips_2_plot = ['base','X','trial_start'][2:]
    for pip in pips_2_plot:
        cmap_norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
        psth_plot = plot_sorted_psth(event_responses_4_psth,pip,pip,window=[-0.25,1],sort_window=[0.1,0.25],
                                     im_kwargs=dict(norm=cmap_norm,cmap='bwr'),sessname_filter=hipp_animals)
        psth_plot[0].set_layout_engine('tight')
        format_axis(psth_plot[1][1],vlines=[0],ylabel='Unit #',
                    xlabel=f'Time from {pip} onset (s)')
        format_axis(psth_plot[1][0],vlines=[0])
        psth_plot[1][0].locator_params(axis='y', nbins=2)
        if pip == 'X':
            psth_plot[1][1].set_ylabel('')

        psth_plot[0].show()

        psth_plot[0].savefig(psth_figdir / f'{pip}_all_resps_psth_aggr_fam_sessions.pdf')

    # psth for miss trials only
    miss_trials_idxs_by_sess = {sess: event_features[sess]['X']['td_df'].eval(cond_filters['miss_all'])
                                for sess in event_responses_4_psth}
    resps_miss_trials = {sess: {'X_miss': event_responses_4_psth[sess]['X'][:len(miss_trials_idxs_by_sess[sess])][miss_trials_idxs_by_sess[sess]]}
                         for sess in event_responses_4_psth if miss_trials_idxs_by_sess[sess].sum()>=5}
    for sess in event_responses_4_psth:
        if miss_trials_idxs_by_sess[sess].sum()<5:
            continue
        resps_miss_trials[sess]['X_hit'] = event_responses_4_psth[sess]['X'][:len(miss_trials_idxs_by_sess[sess])][~miss_trials_idxs_by_sess[sess]]

    # plot psth for miss trials
    # hit_miss_psth_plot = plt.subplots(ncols=2, sharey=True,gridspec_kw={'wspace': 0.05})
    for pip in ['X_miss','X_hit']:
        cmap_norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
        psth_plot = plot_sorted_psth(resps_miss_trials,pip,'X_miss',window=[-0.25,1],sort_window=[0.05,0.2],
                                     im_kwargs=dict(norm=cmap_norm,cmap='bwr',plot_cbar=False),
                                     sessname_filter=hipp_animals)
        psth_plot[0].set_layout_engine('tight')
        format_axis(psth_plot[1][1],vlines=[0],ylabel='Unit #',
                    xlabel=f'Time from {pip} onset (s)')
        format_axis(psth_plot[1][0],vlines=[0],ylim=[-0.1,0.5])
        psth_plot[1][0].locator_params(axis='y', nbins=2,)
        if pip == 'X':
            psth_plot[1][1].set_ylabel('')

        psth_plot[0].show()

        psth_plot[0].savefig(psth_figdir / f'{pip}_resps_psth_aggr_fam_sessions.pdf')

    for sess in resps_miss_trials:
        event_responses[sess]['X_miss'] = resps_miss_trials[sess]['X_miss']

    # do some decoding
    event_responses_by_features_by_sess = {
        e: [event_responses[sessname][e] for sessname in event_responses]
        for e in list(event_responses.values())[0].keys() if e != 'X_miss'}

    full_pattern_responses = aggregate_event_reponses(sessions, events=[e for e in concatenated_event_responses.keys()
                                                                        if 'A' in e],
                                                      events2exclude=['trial_start'], window=[-0.25, 1],
                                                      pred_from_psth_kwargs={'use_unit_zscore': True,
                                                                             'use_iti_zscore': False,
                                                                             'baseline': 0, 'mean': None,
                                                                             'mean_axis': 0})

    concatenated_full_pattern_responses = {
        e: np.concatenate([full_pattern_responses[sessname][e].mean(axis=0) for sessname in full_pattern_responses])
        for e in [e for e in concatenated_event_responses.keys() if 'A' in e]}

    events_by_property = {
        'ptype_i': {pip: 0 if int(pip.split('-')[1]) < 1 else 1 for pip in [f'{p}-{i}' for i in range(3) for p in 'ABCD']},
    }

    # pip_desc = sessions[list(sessions.keys())[0]].pip_desc
    # events_by_property['id'] = {pip: pip_desc[pip]['idx']
    #                             for pip in list(events_by_property['ptype_i'])
    #                             if pip.split('-')[0] in 'ABCD'}
    # events_by_property['position'] = {pip: ord(pip.split('-')[0]) - ord('A') + 1
    #                                   for pip in list(events_by_property['ptype_i'])
    #                                   if pip.split('-')[0] in 'ABCD'}
    # events_by_property['group'] = {pip: 0 if int(pip.split('-')[1]) <1 else 1 for pip in list(events_by_property['ptype_i'])}
    # events_by_property['id'] = {pip: pip if events_by_property['ptype_i'][pip] == 0 else
    #                             f'{"ABBA"[events_by_property["position"][pip]-1]}-{pip.split("-")[1]}'
    #                             for pip in list(events_by_property['ptype_i'])
    #                             if pip.split('-')[0] in 'ABCD'}

    concatenated_full_pattern_by_pip_prop = group_responses_by_pip_prop(concatenated_full_pattern_responses,
                                                                        events_by_property, ['ptype_i'])


    # decode events
    resps2use = event_responses
    pips_as_ints = {pip: pip_i for pip_i, pip in enumerate([f'{pip}-{pip_i}' for pip in 'ABCD' for pip_i in range(1)])}
    # train on A decode on B C D
    sessname = list(resps2use.keys())[0]
    decoders_dict = {}
    # patt_is = [1, 2]
    bad_dec_sess = set()
    for sessname in tqdm(list(resps2use.keys()), total=len(resps2use), desc='decoding across sessions'):
        xys = [(event_responses[sessname][f'{pip}'],
                np.full_like(event_responses[sessname][pip][:,0,0], pips_as_ints[pip]))
               for pip in [f'{p}-0' for p in 'ABCD']]
        xs = np.vstack([xy[0][:,:,-15:-5].mean(axis=-1) for xy in xys])
        ys = np.hstack([xy[1] for xy in xys])
        try:
            decoders_dict[f'{sessname}-allvall'] = decode_responses(xs, ys,n_runs=100)
        except ValueError:
            print(f'{sessname}-allvall failed')
            bad_dec_sess.add(sessname)
            continue
    [decoders_dict[dec_name]['data'].plot_confusion_matrix([f'{p}-{0}' for p in 'ABCD'],)
     for dec_name in decoders_dict.keys()]
    [decoders_dict[dec_name]['shuffled'].plot_confusion_matrix([f'{p}-{0}' for p in 'ABCD'],)
     for dec_name in decoders_dict.keys()]
    # [decoders_dict[dec_name]['data'].cm_plot[0].show()
    #  for dec_name in decoders_dict.keys()]

    [decoders_dict.pop(dec_name) for dec_name in [k for k in decoders_dict.keys() for sess in bad_dec_sess if
                                                  k.startswith(sess)]
     if dec_name in decoders_dict.keys()]
    animals = ['DO79','DO81']
    all_cms_by_pip =[[dec['data'].cm for dec_name, dec in decoders_dict.items() if dec['data'].cm is not None
                      and any([e in dec_name for e in animals])]]
    all_cms_by_pip_arr = np.squeeze(np.array(all_cms_by_pip))
    all_cms_by_pip_plot = plt.subplots(ncols=len(all_cms_by_pip))
    # cmap_norm = TwoSlopeNorm(vmin=np.quantile(all_cms_by_pip_arr,0.05), vmax=np.quantile(all_cms_by_pip_arr,0.95),
    #                          vcenter=1/all_cms_by_pip_arr.shape[-1])
    cmap_norm = TwoSlopeNorm(vmin=0, vmax=0.5,
                             vcenter=1/all_cms_by_pip_arr.shape[-1])
    cm_plots = [ConfusionMatrixDisplay(np.mean(pip_cms,axis=0),display_labels=list('ABCD'))
                for pip_cms in all_cms_by_pip]
    [cm_plot.plot(cmap='bwr',include_values=False,colorbar=False, im_kw=dict(norm=cmap_norm))
     for cm_plot in cm_plots]
    [cm_plot.ax_.invert_yaxis() for cm_plot in cm_plots]
    [cm_plot.ax_.set_xlabel('') for cm_plot in cm_plots]
    [cm_plot.ax_.set_ylabel('') for cm_plot in cm_plots]
    [cm_plot.figure_.set_size_inches(2, 2) for cm_plot in cm_plots]
    [cm_plot.figure_.set_layout_engine('constrained') for cm_plot in cm_plots]
    [cm_plot.figure_.show() for cm_plot in cm_plots]
    cm_plots[0].figure_.savefig(aggr_figdir / 'decoding_cms_all_v_all_hipp_only_no_cbar.pdf')
    dec_acc_plot = plt.subplots()
    accuracy_across_sessions = [np.diagonal(pip_cm,axis1=1, axis2=2) for pip_cm in all_cms_by_pip]
    accuracy_across_sessions_dfs = [pd.DataFrame(pip_accs,index=list([sess for sess in resps2use.keys()
                                                                      if sess not in bad_dec_sess
                                                                      and any([e in sess for e in animals])]),)
                                    for pip_accs in accuracy_across_sessions]

    # barplot for each pip
    accuracy_across_sessions_df = pd.concat(accuracy_across_sessions_dfs, axis=1)
    pip_vs_pip_acc_plot = plt.subplots()
    pip_vs_pip_acc_plot[1].boxplot(accuracy_across_sessions_df, labels=list('ABCD'))
    pip_vs_pip_acc_plot[1].set_ylabel('Decoding accuracy')
    format_axis(pip_vs_pip_acc_plot[1],hlines=[0.25])
    pip_vs_pip_acc_plot[0].show()
    pip_vs_pip_acc_plot[0].set_size_inches(3.5,2.2)
    pip_vs_pip_acc_plot[0].savefig(aggr_figdir / 'decoding_across_pips.pdf')

    [print(ttest_1samp(accuracy_across_sessions_df[acc], 1/all_cms_by_pip_arr.shape[-1], alternative='greater'))
     for acc in accuracy_across_sessions_df]
    [print((i,j),ttest_ind(accuracy_across_sessions_df[i],accuracy_across_sessions_df[j], alternative='greater'))
     for i,j in list(combinations(accuracy_across_sessions_df.columns,2))]
    print(ttest_ind(accuracy_across_sessions[0][0],accuracy_across_sessions[1][0], alternative='two-sided'))
    print(ttest_ind(accuracy_across_sessions[0][1],accuracy_across_sessions[1][1], alternative='two-sided'))

    ### delete ###
    # predict b deviant
    dev_preds = {f'{pip}-1':[[model[0].predict(event_responses[dec_name.split('-')[0]][f'{pip}-1'][:, :, -15:-5].mean(axis=-1))
                      for model in dec['data'].models]
                     for dec_name, dec in decoders_dict.items() if any(e in dec_name for e in animals)]
                    for pip in 'ABCD'}
    dev_preds_cm = {}
    for pip in 'ABCD':
        sess_preds_cm = []
        for sess_preds in dev_preds[f'{pip}-1']:
            all_preds = np.hstack(sess_preds)
            sess_preds_cm += [[(all_preds == pred_i).mean() for pred_i in range(len('ABCD'))]]
        dev_preds_cm[f'{pip}-1'] = sess_preds_cm
    dev_preds_cm_arr = np.array(list(dev_preds_cm.values()))
    dev_preds_cm_arr = dev_preds_cm_arr.transpose((1, 0, 2))
    # plot cm
    cm_dev_preds_plot = ConfusionMatrixDisplay(np.mean(dev_preds_cm_arr, axis=0),display_labels=list('ABCD'))
    cm_dev_preds_plot.plot(cmap='bwr', include_values=True, colorbar=False, im_kw=dict(norm=cmap_norm))
    cm_dev_preds_plot.ax_.invert_yaxis()
    cm_dev_preds_plot.ax_.set_xlabel('')
    cm_dev_preds_plot.ax_.set_ylabel('')
    cm_dev_preds_plot.figure_.set_size_inches(2, 2)
    cm_dev_preds_plot.figure_.set_layout_engine('constrained')
    cm_dev_preds_plot.figure_.show()
    cm_dev_preds_plot.figure_.savefig(aggr_figdir / 'decoding_dev_preds_ABCD1.pdf')

    # boxplot of accuracy for each pip
    dev_preds_acc_plot = plt.subplots()
    # dev_pred_acc_by_sess = dev_preds_cm_arr.diagonal(axis1=1, axis2=2).T
    [print(ttest_1samp(dev_preds_cm_arr[:,i,:], 1/dev_preds_cm_arr.shape[1], alternative='greater'))
     for i in range(dev_preds_cm_arr.shape[1])]
    dev_pred_acc_by_sess = dev_preds_cm_arr[:,1,:].T
    [print(ttest_1samp(sess_accs, 1/dev_pred_acc_by_sess.shape[0], alternative='greater'))
     for sess_accs in dev_pred_acc_by_sess for i in range(dev_pred_acc_by_sess.shape[0])]
    [dev_preds_acc_plot[1].scatter(['ABCD'[i]] * dev_pred_acc_by_sess.shape[1], accs,c='grey',marker='x')
     for i, accs in enumerate(dev_pred_acc_by_sess)]
    [dev_preds_acc_plot[1].scatter('ABCD'[i], accs.mean(), c='darkred', marker='^', s=25,)
     for i, accs in enumerate(dev_pred_acc_by_sess)]
    dev_preds_acc_plot[1].set_ylabel('Deviant prediction accuracy')
    format_axis(dev_preds_acc_plot[1], hlines=[0.25])
    dev_preds_acc_plot[0].show()
    dev_preds_acc_plot[0].set_size_inches(3.5, 2.2)
    dev_preds_acc_plot[0].savefig(aggr_figdir / 'dev_acc_dev_preds.pdf')



        # all_resps_psth[0].savefig(psth_figdir / f'{pip}_all_resps_psth_abstraction_sessions.pdf')

        # format axis
    # td_df stuff
    for sessname in sessions:
        if 'date' not in sessions[sessname].td_df.index.names:
            sessions[sessname].td_df['date'] = [sessname.split('_')[1][:-1]]*len(sessions[sessname].td_df)
            sessions[sessname].td_df.set_index('date', append=True, inplace=True,drop=True)
        if 'n_since_last' in sessions[sessname].td_df:
            sessions[sessname].td_df = sessions[sessname].td_df.drop(columns=['n_since_last'])
        sessions[sessname].td_df['n_since_last'] = get_last_pattern(sessions[sessname].td_df['Tone_Position']==1)
        get_n_since_last(sessions[sessname].td_df, 'Trial_Outcome', 0)
        [get_prate_block_num(sessions[sessname].td_df, prate, rate_name) for prate, rate_name in
         zip([0.1, 0.9], ['frequent', 'rare'])]
        [get_prate_block_num(sessions[sessname].td_df, prate, rate_name) for prate, rate_name in
         zip([0.1, 0.9], ['recent', 'distant'])]
        get_cumsum_columns(sessions, sessname)
        [add_datetimecol(sessions[sessname].td_df, col) for col in ['Trial_Start', 'ToneTime', 'Trial_End', 'Gap_Time',
                                                                    'Bonsai_Time']]
        get_lick_in_patt_trials(sessions[sessname].td_df, sessname[:-1])
        get_earlyX_trials(sessions[sessname].td_df)

    event_features_all = aggregate_event_features(sessions,
                                              events2exclude=['trial_start'])


    full_pattern_responses = aggregate_event_reponses(sessions, events=[e for e in concatenated_event_responses.keys()
                                                                        if 'A' in e],
                                                      events2exclude=['trial_start'], window=[-0.25, 1],
                                                      pred_from_psth_kwargs={'use_unit_zscore': True,
                                                                             'use_iti_zscore': False,
                                                                             'baseline': 0, 'mean': None,
                                                                             'mean_axis': 0})

    # decode pattern from base
    decode_patt_base_dict = {}
    pips2decode = [['A-0','base'],['X','base'],['A-0','X'],['X_miss','A-0']]
    for pips in pips2decode[-2:]:
        for sessname in tqdm(event_responses.keys(), desc='decoding', total=len(event_responses.keys())):
            if not all(p in list(event_responses[sessname].keys()) for p in pips):
                continue
            dec_sffx = "_vs_".join(pips)
            xs_list = [event_responses[sessname][pip] for pip in pips]
            xs = np.vstack([x[:,:,10:].mean(axis=-1) for x in xs_list])
            ys = np.hstack([np.full(x.shape[0], ci) for ci, x in enumerate(xs_list)])
            decode_patt_base_dict[f'{sessname}_{dec_sffx}'] = decode_responses(xs, ys, n_runs=100)
        # plot accuracy
        patt_v_base_accuracy = np.array([decode_patt_base_dict[dec_name]['data'].accuracy
                                         for dec_name in decode_patt_base_dict.keys() if dec_sffx in dec_name
                                         and any([e in dec_name for e in hipp_animals])])
        patt_v_base_shuff_accuracy = np.array([decode_patt_base_dict[dec_name]['shuffled'].accuracy
                                               for dec_name in decode_patt_base_dict.keys() if dec_sffx in dec_name
                                               and any([e in dec_name for e in hipp_animals])])
        patt_v_base_accuracy_plot = plt.subplots()
        patt_v_base_accuracy_plot[1].boxplot([patt_v_base_accuracy.mean(axis=1),
                                                  patt_v_base_shuff_accuracy.mean(axis=1)],labels=['data','shuffle'],
                                                 showmeans=False, meanprops=dict(mfc='k'),)
        patt_v_base_accuracy_plot[1].set_ylabel('Accuracy')
        patt_v_base_accuracy_plot[0].show()
        patt_v_base_accuracy_plot[0].savefig(aggr_figdir / f'{dec_sffx}_accuracy.pdf')
        # ttest
        ttest = ttest_ind(patt_v_base_accuracy.mean(axis=1), patt_v_base_shuff_accuracy.mean(axis=1),
                          alternative='greater', equal_var=True)
        print(f'{dec_sffx} ttest: {ttest}')
        all_patt_v_base_accs = {sess: np.mean(decode_patt_base_dict[f'{sess}_A-0_vs_base']['data'].accuracy )
                                for sess in accuracy_across_sessions_df.index}
        patt_base_allvall_accs = {sess: {'patt_base':all_patt_v_base_accs[sess],
                                         'all_v_all': accuracy_across_sessions_df.loc[sess].mean()}
                                  for sess in accuracy_across_sessions_df.index}
        patt_base_allvall_accs_df = pd.DataFrame(patt_base_allvall_accs).T
        patt_base_allvall_accs_plot = plt.subplots()
        [patt_base_allvall_accs_plot[1].scatter(accs['patt_base'],accs['all_v_all'],c='darkblue' if 'DO79' in sess else 'darkred',
                                               alpha=0.8)
         for sess,accs in patt_base_allvall_accs_df.iterrows()]
        format_axis(patt_base_allvall_accs_plot[1])
        patt_base_allvall_accs_plot[0].show()

        run_glm(patt_base_allvall_accs_df['patt_base'], patt_base_allvall_accs_df['all_v_all'],)[1].summary()


    # decode by condition
    pip = 'A-0'
    # window = (-0.1, 0.25)
    # event_responses = aggregate_event_reponses(sessions, events=None,  # [f'{pip}-0' for pip in 'ABCD']
    #                                            events2exclude=['trial_start','base', 'X'], window=window,
    #                                            pred_from_psth_kwargs={'use_unit_zscore': True, 'use_iti_zscore': False,
    #                                                                   'baseline': 0, 'mean': None, 'mean_axis': 0})
    resps_by_cond = {}
    conds = ['rare', 'frequent','distant', 'recent']
    # conds = ['distant', 'recent']

    for cond in conds:
        resps_by_cond[cond] = group_ephys_resps('A-0', full_pattern_responses, event_features_all, cond_filters[cond])[0]
    decode_by_cond_dict = {}
    for conds2dec in [['rare', 'frequent'], ['distant', 'recent']][:1]:
        dec_sffx = "_v_".join(conds2dec)
        for sessname in tqdm(resps_by_cond[conds[0]].keys(),desc='decoding', total=len(resps_by_cond[conds[0]].keys())):
            try:
                xs_list = [resps_by_cond[cond][sessname] if cond != 'distant' else
                           resps_by_cond[cond][sessname][::2]
                           for cond in conds2dec]
            except:
                continue
            xs = np.vstack([x[:,:,70:].mean(axis=-1) for x in xs_list])
            ys = np.hstack([np.full(x.shape[0],ci) for ci,x in enumerate(xs_list)])
            decode_by_cond_dict[f'{sessname}-{dec_sffx}'] = decode_responses(xs, ys, n_runs=50,
                                                                                      dec_kwargs={'n_jobs': 8,
                                                                                                  'cv_folds': 5,})
        rec_dec_decoder_accuracy = np.array([decode_by_cond_dict[dec_name]['data'].accuracy
                                             for dec_name in decode_by_cond_dict.keys() if dec_sffx in dec_name
                                             and any([e in dec_name for e in hipp_animals])])
        rec_dec_decoder_shuff_accuracy = np.array([decode_by_cond_dict[dec_name]['shuffled'].accuracy
                                             for dec_name in decode_by_cond_dict.keys() if dec_sffx in dec_name
                                             and any([e in dec_name for e in hipp_animals])])

        rec_dec_decoder_accuracy_plot = plt.subplots()
        rec_dec_decoder_accuracy_plot[1].boxplot([rec_dec_decoder_accuracy.mean(axis=1),
                                                  rec_dec_decoder_shuff_accuracy.mean(axis=1)],labels=['data','shuffle'],
                                                 widths=0.4,
                                                 showmeans=False, meanprops=dict(mfc='k'),
                                                 medianprops=dict(color='k',linewidth=2))

        rec_dec_decoder_accuracy_plot[1].set_title(dec_sffx)
        format_axis(rec_dec_decoder_accuracy_plot[1],hlines=[0.5])
        rec_dec_decoder_accuracy_plot[1].set_ylim(0.3,0.8)
        rec_dec_decoder_accuracy_plot[0].set_layout_engine('tight')
        rec_dec_decoder_accuracy_plot[0].set_size_inches(2,2)
        rec_dec_decoder_accuracy_plot[0].show()
        rec_dec_decoder_accuracy_plot[0].savefig(aggr_figdir / f'{dec_sffx}_decoding_accuracy_hipp_animals.pdf')

        # ttest
        ttest = ttest_ind(rec_dec_decoder_accuracy.mean(axis=1), rec_dec_decoder_shuff_accuracy.mean(axis=1),
                          alternative='greater', equal_var=False)
        print(f'{" vs ".join(conds)} ttest: {ttest}')
        format_axis(rec_dec_decoder_accuracy_plot[1])
        rec_dec_decoder_accuracy_plot[0].savefig(aggr_figdir / f'{dec_sffx}_decoding_accuracy.pdf')

    # look at coefficients
    dec_sffx = "_v_".join(['rare', 'frequent'])
    dec_coeff_dict = {}
    for dec_name in decode_by_cond_dict.keys():
        if dec_sffx in dec_name:
            dec_coeff_dict[dec_name] = np.mean(np.array([np.squeeze([ee.coef_ for ee in e])
                                                for e in decode_by_cond_dict[dec_name]['shuffled'].models]),
                                               axis=0).mean(axis=0)
    hist_all_coefs = plt.subplots()
    hist_all_coefs[1].hist(np.hstack(list(dec_coeff_dict.values())), bins='fd', alpha=0.5, fc='k')
    hist_all_coefs[0].show()


    # psth rare and frequent
    response_dict_w_label = {}
    for sess in resps_by_cond[conds[0]].keys():
        try:
            response_dict_w_label[sess] = {f'{cond}':resps_by_cond[cond][sess] for cond in conds}
        except KeyError:
            continue
    own_sorting_list = [False, True]
    pip = 'pattern'
    for own_sorting in own_sorting_list:
        for cond in conds:
            for animal in ['DO79', 'DO81', 'DO82']:
                cmap_norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
                cond_psths_plot = plot_sorted_psth(response_dict_w_label, cond, 'rare' if not own_sorting else cond,
                                                   window=[-0.25, 1],
                                                  sort_window=[0, 1],
                                                  sessname_filter=animal, im_kwargs=dict(norm=cmap_norm, cmap='bwr'))
                cond_psths_plot[0].set_layout_engine('tight')
                format_axis(cond_psths_plot[1][1], vlines=np.arange(0, 1, 0.25).tolist(), ylabel='Unit #',
                            xlabel=f'Time from {pip} onset (s)')
                format_axis(cond_psths_plot[1][0], vlines=[0])
                cond_psths_plot[1][0].locator_params(axis='y', nbins=2)
                cond_psths_plot[0].suptitle(f'{animal}: {cond} responses')
                cond_psths_plot[0].show()
                f_name = f'{pip}_{animal}_resps_{cond}_psth' + ('_own_sorting' if own_sorting else '')

                cond_psths_plot[0].savefig(psth_figdir / f'{f_name}.pdf')
    for own_sorting in own_sorting_list:
        for cond in conds:
            # cmap_norm = TwoSlopeNorm(vcenter=0, vmin=-4, vmax=4)
            cmap_norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
            cond_psths_plot = plot_sorted_psth(response_dict_w_label, cond, 'rare' if not own_sorting else cond,
                                               window=[-0.25, 1],
                                              sort_window=[0, 1],
                                              # sessname_filter=None, im_kwargs=dict(norm=None, cmap='bwr'))
                                              sessname_filter=hipp_animals, im_kwargs=dict(norm=cmap_norm, cmap='bwr'))
            cond_psths_plot[0].set_layout_engine('tight')
            format_axis(cond_psths_plot[1][1], vlines=np.arange(0, 1, 0.25).tolist(), ylabel='Unit #',
                        xlabel=f'Time from {pip} onset (s)')
            format_axis(cond_psths_plot[1][0], vlines=[0])
            cond_psths_plot[1][0].locator_params(axis='y', nbins=2)
            cond_psths_plot[0].suptitle(f'all animals: {cond} responses')
            cond_psths_plot[1][0].set_ylim(-.1,0.1)
            cond_psths_plot[0].set_size_inches(3, 3)
            cond_psths_plot[0].show()
            f_name = f'{pip}_all_animals_resps_{cond}_psth' + ('_own_sorting' if own_sorting else '')
            cond_psths_plot[0].savefig(psth_figdir / f'{f_name}.pdf')

    # compare with pupil
    A_by_cond_path = ceph_dir/r'X:\Dammy\Xdetection_mouse_hf_test\processed_data\A_by_cond_ephys_2401_musc_2406.pkl'
    with open(A_by_cond_path, 'rb') as f:
        A_by_cond = pickle.load(f)

    resps_by_cond_trial_nums = {}
    for cond in conds:
        resps_by_cond_trial_nums[cond] = group_ephys_resps('A-0', full_pattern_responses, event_features_all,
                                                           # cond_filters[cond]+'&n_since_last_Trial_Outcome <=5')[1]
                                                           cond_filters[cond])[1]
    matched_pupil_ephys_popmean = []
    matched_pupil_ephys_byunits = []
    for sessname in response_dict_w_label:
        if sessname[:-1] not in A_by_cond['rare'].index.get_level_values('sess').unique().values:
            print(f'{sessname[:-1]} not in A_by_cond')
            continue
        if sessname not in resps_by_cond_trial_nums['rare']:
            print(f'{sessname} not in resps_by_cond_trial_nums')
            continue

        if not any([e in sessname for e in hipp_animals]):
            continue

        mutual_trials = np.intersect1d(A_by_cond['rare'].xs(sessname[:-1],level='sess').index.get_level_values('trial').values,
                                       resps_by_cond_trial_nums['rare'][sessname])
        ephys_resps = group_ephys_resps('A-0', full_pattern_responses, event_features_all,sess_list=[sessname],
                                        trial_nums2use=mutual_trials)[0]
        if not ephys_resps:
            continue
        pupil_resps = A_by_cond['rare'].xs(sessname[:-1], level='sess').query('trial in @mutual_trials')
        pupil_resps = pupil_resps.T.rolling(window=25).mean().T
        matched_pupil_ephys_popmean.append(pd.DataFrame({'pupil':np.nanmean(pupil_resps.loc[:, 1:2.5], axis=1),
                                                         'ephys':ephys_resps[sessname][:,:,25:50].mean(axis=-1).mean(axis=-1)},
                                                        index=[sessname]*len(mutual_trials)))
        matched_pupil_ephys_byunits.append(pd.DataFrame({'pupil': np.nanmean(pupil_resps.loc[:, 1:2.5], axis=1),
                                                         'ephys': ephys_resps[sessname][:, :, 25:50].mean(axis=-1).tolist()},
                                                        index=[sessname] * len(mutual_trials)))
    pupil_ephys_df = pd.concat(matched_pupil_ephys_popmean)
    pupil_ephys_byunits_df = pd.concat(matched_pupil_ephys_byunits)
    ephys_pupil_plot = plt.subplots()
    [ephys_pupil_plot[1].scatter([row['pupil']]*len([e for e in row['ephys'] if e>1]),[e for e in row['ephys'] if e>1],
                                 c='k',alpha=0.2)
     for _, row in pupil_ephys_byunits_df.iterrows()]
    ephys_pupil_plot[1].set_xlabel('Pupil')
    ephys_pupil_plot[1].set_ylabel('Ephys')
    ephys_pupil_plot[1].set_title('Pupil vs. Ephys')
    ephys_pupil_plot[0].set_layout_engine('tight')
    ephys_pupil_plot[0].show()

    ephys_pupil_plot[0].savefig(psth_figdir / 'ephys_pupil.pdf')


    pupil_ephys_glm = run_glm(pupil_ephys_byunits_df['ephys'],pupil_ephys_byunits_df['pupil'])
    print(pupil_ephys_glm[1].summary())

    decode_w_pupil = {sessname: decode_responses(
        np.array([np.array(e) for e in pupil_ephys_byunits_df.loc[sessname]['ephys']]),
        pupil_ephys_byunits_df.loc[sessname]['pupil'].values,
        'linear',
                                        )
                      for sessname in pupil_ephys_byunits_df.index}
    [print(e[1].summary()) for e in decode_w_pupil.values()]

    decode_w_pupil_plot = plt.subplots()
    all_predictions = {sess: np.abs(np.squeeze(np.diff(np.squeeze(np.array(e['data'].predictions)),axis=-2)))
                       for sess, e in decode_w_pupil.items()}
    shuffled_predictions = {sess: np.abs(np.squeeze(np.diff(np.squeeze(np.array(e['shuffled'].predictions)),axis=-2)))
                       for sess, e in decode_w_pupil.items()}


    # all v all with rare freq
    resps_by_cond_by_pip = {}
    full_resps_by_cond_by_pip = {}
    conds = ['rare', 'frequent']
    for cond in conds:
        for pip in [f'{e}-0' for e in 'ABCD']:
            resps_by_cond_by_pip[f'{pip}_{cond}'] = group_ephys_resps(pip, event_responses, event_features_all,
                                                                      cond_filters[cond])[0]
        for pip in [f'{e}-0' for e in 'A']:
            full_resps_by_cond_by_pip[f'{pip}_{cond}'] = group_ephys_resps(pip, full_pattern_responses_4_psth, event_features_all,
                                                                      cond_filters[cond])[0]
    # decode events
    resps2use = resps_by_cond_by_pip
    pips_as_ints = {pip: pip_i for pip_i, pip in enumerate([f'{pip}-{pip_i}' for pip in 'ABCD' for pip_i in range(1)])}
    # train on A decode on B C D
    decoders_dict = {}
    # patt_is = [1, 2]
    bad_dec_sess = set()
    # for cond in conds:
    mutual_sess_for_conds = np.intersect1d(*[list(resps_by_cond_by_pip[f'A-0_{cond}'].keys()) for cond in conds])
    for sessname in mutual_sess_for_conds:
        xys_by_cond = [[(resps_by_cond_by_pip[f'{pip}_{cond}'][sessname],
                np.full_like(resps_by_cond_by_pip[f'{pip}_{cond}'][sessname][:, 0, 0],
                             pips_as_ints[pip.split('_')[0]]))
               for pip in [f'{p}-0' for p in 'ABCD']]
               for cond in conds]

        xs_by_cond = [np.vstack([xy[0][:, :,10:20].mean(axis=-1) for xy in xys])
              for xys in xys_by_cond]
        n_cond1 = len(xs_by_cond[0])
        xs = np.vstack(xs_by_cond)
        ys = np.hstack([np.hstack([xy[1] for xy in xys]) for xys in xys_by_cond])
        try:
            decoders_dict[f'{sessname}-allvall_{"_".join(conds)}'] = decode_responses(xs, ys, n_runs=100,
                                                                                      dec_kwargs={'pre_split': n_cond1})
        except ValueError:
            print(f'{sessname}-allvall failed')
            bad_dec_sess.add(sessname)
            continue
    [decoders_dict[dec_name]['data'].plot_confusion_matrix([f'{p}-{0}' for p in 'ABCD'], )
     for dec_name in decoders_dict.keys()]
    # [decoders_dict[dec_name]['data'].cm_plot[0].show()
    #  for dec_name in decoders_dict.keys()]

    [decoders_dict.pop(dec_name) for dec_name in [k for k in decoders_dict.keys() for sess in bad_dec_sess if
                                                  k.startswith(sess)]
     if dec_name in decoders_dict.keys()]
    all_cms_by_pip = [[dec['data'].cm for dec_name, dec in decoders_dict.items() if dec['data'].cm is not None
                       if cond in dec_name and any([e in dec_name for e in hipp_animals])]
                      for cond in conds[:1]]
    all_cms_by_pip_arr = np.squeeze(np.array(all_cms_by_pip))
    # cmap_norm = TwoSlopeNorm(vmin=np.quantile(all_cms_by_pip_arr, 0.1), vmax=np.quantile(all_cms_by_pip_arr, 0.9),
    #                          vcenter=1 / all_cms_by_pip_arr.shape[-1])
    cmap_norm = TwoSlopeNorm(vmin=0.1, vmax=0.6,
                             vcenter=1 / all_cms_by_pip_arr.shape[-1])
    all_cms_by_pip_plot = plot_aggr_cm(all_cms_by_pip_arr,im_kwargs={'norm':cmap_norm},labels=list('ABCD'),
                                       cmap='bwr',colorbar=True)
    # all_cms_by_pip_plot[0].set_layout_engine('constrained')

    all_cms_by_pip_plot[0].show()
    all_cms_by_pip_plot[0].savefig(aggr_figdir / f'decoding_cms_all_v_all_cross_cond_only_hipp_animals_no_cbar.pdf')

    accuracy_across_sessions = [np.diagonal(pip_cm, axis1=1, axis2=2) for pip_cm in all_cms_by_pip]
    accuracy_across_sessions_dfs = [pd.DataFrame(pip_accs, index=list([sess for sess in mutual_sess_for_conds
                                                                       if sess not in bad_dec_sess
                                                                       and any([e in sess for e in hipp_animals])]))
                                    for pip_accs in accuracy_across_sessions]

    # barplot for each pip
    pip_vs_pip_acc_plot = plt.subplots()
    [pip_vs_pip_acc_plot[1].boxplot([acc for acc in accs.T],meanline=True,
                                   medianprops={'color': f'C{ii}'},
                                   labels=list('ABCD'),positions=np.arange(0,4)*2+ii)
     for ii, accs in enumerate(accuracy_across_sessions)]
    format_axis(pip_vs_pip_acc_plot[1],hlines=[0.25])
    pip_vs_pip_acc_plot[0].show()
    pip_vs_pip_acc_plot[0].savefig(aggr_figdir / f'pip_vs_pip_accuracy_across_sessions_cross_cond_only.pdf')

    # separately by cond
    resps2use = resps_by_cond_by_pip
    pips_as_ints = {pip: pip_i for pip_i, pip in enumerate([f'{pip}-{pip_i}' for pip in 'ABCD' for pip_i in range(1)])}
    # train on A decode on B C D
    decoders_dict_seperated_conds = {}
    # patt_is = [1, 2]
    bad_dec_sess = set()
    for cond in conds:
        for sessname in tqdm(mutual_sess_for_conds, total=len(resps2use),
                             desc='decoding across sessions'):
            xys = [(resps_by_cond_by_pip[f'{pip}_{cond}'][sessname],
                    np.full_like(resps_by_cond_by_pip[f'{pip}_{cond}'][sessname][:, 0, 0],
                                 pips_as_ints[pip.split('_')[0]]))
                   for pip in [f'{p}-0' for p in 'ABCD']]
            xs = np.vstack([xy[0][:, :, 10:20].mean(axis=-1) for xy in xys])
            ys = np.hstack([xy[1] for xy in xys])
            try:
                decoders_dict_seperated_conds[f'{sessname}-allvall_{cond}'] = decode_responses([xs], [ys], n_runs=1000)
            except ValueError:
                print(f'{sessname}-allvall failed')
                bad_dec_sess.add(sessname)
                continue
    [decoders_dict_seperated_conds[dec_name]['data'].plot_confusion_matrix([f'{p}-{0}' for p in 'ABCD'], )
     for dec_name in decoders_dict_seperated_conds.keys()]
    # [decoders_dict[dec_name]['data'].cm_plot[0].show()
    #  for dec_name in decoders_dict.keys()]

    [decoders_dict_seperated_conds.pop(dec_name) for dec_name in [k for k in decoders_dict_seperated_conds.keys() for sess in bad_dec_sess if
                                                  k.startswith(sess) ]
     if dec_name in decoders_dict_seperated_conds.keys() ]
    all_cms_by_pip_all_conds = [[[dec['data'].cm for dec_name, dec in decoders_dict_seperated_conds.items() if dec['data'].cm is not None
                       if cond in dec_name and any([e in dec_name for e in hipp_animals])]]
                      for cond in conds]
    all_cms_by_pip_all_conds = [np.squeeze(arr) for arr in all_cms_by_pip_all_conds]
    for cond_2plot in conds:
        all_cms_by_pip = [all_cms_by_pip_all_conds[conds.index(cond_2plot)]]
        all_cms_by_pip_arr = np.squeeze(np.array(all_cms_by_pip))
        # cmap_norm = TwoSlopeNorm(vmin=np.quantile(all_cms_by_pip_arr, 0.1), vmax=np.quantile(all_cms_by_pip_arr, 0.9),
        #                          vcenter=1 / all_cms_by_pip_arr.shape[-1])
        cmap_norm = TwoSlopeNorm(vmin=0.1, vmax=0.6,
                                 vcenter=1 / all_cms_by_pip_arr.shape[-1])
        all_cms_by_pip_plot = plot_aggr_cm(all_cms_by_pip_arr, im_kwargs={'norm': cmap_norm}, labels=list('ABCD'),
                                           colorbar=False, cmap='bwr')
        all_cms_by_pip_plot[0].set_layout_engine('constrained')
        all_cms_by_pip_plot[0].show()

        all_cms_by_pip_plot[0].savefig(aggr_figdir / f'decoding_cms_all_v_all_{cond_2plot}_only.pdf')
    accuracy_across_sessions_separated = [np.diagonal(pip_cm, axis1=1, axis2=2) for pip_cm in all_cms_by_pip_all_conds]
    # accuracy_across_sessions_dfs = [pd.DataFrame(pip_accs, index=mutual_sess_for_conds)
    #                                 for pip_accs in accuracy_across_sessions]

    # barplot for each pip
    accs_sep_and_cross = accuracy_across_sessions_separated + accuracy_across_sessions
    pip_vs_pip_acc_plot = plt.subplots()
    [pip_vs_pip_acc_plot[1].boxplot([acc for acc in accs.T], meanline=True,
                                    medianprops={'color': f'C{ii}'},
                                    labels=list('ABCD'), positions=np.arange(0, 4) * len(accs_sep_and_cross) + ii)
     for ii, accs in enumerate(accs_sep_and_cross)]
    pip_vs_pip_acc_plot[1].axhline(0.25, color='k',ls='--')
    pip_vs_pip_acc_plot[0].set_size_inches((6, 3))
    pip_vs_pip_acc_plot[0].show()
    pip_vs_pip_acc_plot[0].savefig(aggr_figdir / f'decoding_accuracy_across_sessions_hipp_only.pdf')


    # plot decoding ts for each pip
    # rolling window of norm dev decoding



    #  leave one out rare freq decoding
    rare_freq_loo_dict = {}
    for sessname in tqdm(pupil_ephys_byunits_df.index.unique(), total=len(pupil_ephys_byunits_df.index.unique()), desc='loo decoding'):
        xs_by_cond = [resps_by_cond_by_pip[f'D-0_{cond}'][sessname] for cond in conds]
        ys_by_cond = [np.full_like(x[:, 0, 0], xi) for xi,x in enumerate(xs_by_cond)]
        ys = np.hstack(ys_by_cond)
        xs = np.vstack(xs_by_cond)[:,:,-20:].mean(axis=-1)
        dec_res = decode_responses(xs, ys, n_runs=1,dec_kwargs={'loo_cv':True})
        rare_freq_loo_dict[sessname] = dec_res

    loo_accs_by_sess = {sessname: np.mean(rare_freq_loo_dict[sessname]['data'].accuracy)
                         for sessname, dec_res in rare_freq_loo_dict.items()}
    # plot accs boxplot
    loo_accs_by_sess_df = pd.DataFrame(loo_accs_by_sess.values(), index=list(loo_accs_by_sess.keys()))
    loo_accs_by_sess_plot = plt.subplots()
    loo_accs_by_sess_plot[1].boxplot(loo_accs_by_sess_df[0].values)
    loo_accs_by_sess_plot[0].show()

    loo_preds_by_sess = {sessname: np.array(np.squeeze(rare_freq_loo_dict[sessname]['data'].predictions[0]))
                         for sessname, dec_res in rare_freq_loo_dict.items()}

    matched_pupil_ephys_dec_label = {}
    for sessname in loo_preds_by_sess:

        sess_cond_trial_nums = np.hstack([resps_by_cond_trial_nums[cond][sessname] for cond in conds])
        mutual_trials = np.hstack([np.intersect1d(
            A_by_cond[cond].xs(sessname[:-1], level='sess').index.get_level_values('trial').values,
            resps_by_cond_trial_nums[cond][sessname]) for cond in conds])
        ephys_resps_idxs = np.isin(sess_cond_trial_nums, mutual_trials)
        ephys_dec_lbl = loo_preds_by_sess[sessname][1][ephys_resps_idxs]
        sess_pupil = pd.concat([A_by_cond[cond].xs(sessname[:-1], level='sess') for cond in conds],axis=0)
        sess_pupil_smoothed = sess_pupil.T.rolling(window=25).mean()[1.5:2.5].T
        sess_pupil_ephys_df = pd.DataFrame({'dec_lbl': ephys_dec_lbl, 'pupil_max': sess_pupil_smoothed.max(axis=1),
                                            'pupil_mean': sess_pupil_smoothed.mean(axis=1)},)
        sess_pupil_ephys_df.set_index('dec_lbl',inplace=True,append=True,drop=True)
        matched_pupil_ephys_dec_label[sessname] = sess_pupil_ephys_df

    pupil_dec_lbl_df = pd.concat(list(matched_pupil_ephys_dec_label.values()),axis=0)
    for metric2plot in ['pupil_max','pupil_mean']:
        pupil_metric_by_dec_lbl_hist = plt.subplots()
        [pupil_metric_by_dec_lbl_hist[1].hist(pupil_dec_lbl_df.xs(dec_lbl,level='dec_lbl')[metric2plot],
                                              label=cond, alpha=0.5, bins='fd', density=True,fc=color)
         for dec_lbl,color,cond in zip(pupil_dec_lbl_df.index.get_level_values('dec_lbl').unique(),['darkblue','darkgreen'],
                            conds)]

        pupil_metric_by_dec_lbl_hist[1].set_xlabel(metric2plot.replace('_',' '))
        # pupil_metric_by_dec_lbl_hist[1].set_ylabel('count')
        pupil_metric_by_dec_lbl_hist[1].legend()
        pupil_metric_by_dec_lbl_hist[0].set_layout_engine('tight')
        pupil_metric_by_dec_lbl_hist[0].show()
        pupil_metric_by_dec_lbl_hist[0].savefig(f'pupil_by_dec_lbl_hist_{metric2plot}_{"_".join(conds)}.pdf')

    # ttest on pupil max
    ttest_res = ttest_ind(pupil_dec_lbl_df.xs(0, level='dec_lbl')['pupil_max'],
                          pupil_dec_lbl_df.xs(1, level='dec_lbl')['pupil_max'])
    print(ttest_res)

    window_size = 0.1
    # plot decoding over time
    dec_over_time_window = [-1, 1.5]
    full_pattern_responses_4_ts_dec = aggregate_event_reponses(sessions,
                                                               events=[e for e in concatenated_event_responses.keys()
                                                                       if 'A' in e],
                                                               events2exclude=['trial_start'],
                                                               window=dec_over_time_window,
                                                               pred_from_psth_kwargs={'use_unit_zscore': True,
                                                                                      'use_iti_zscore': False,
                                                                                      'baseline': 0, 'mean': None,
                                                                                      'mean_axis': 0})

    resps_by_cond_4_ts_dec = {}
    conds = ['rare', 'frequent']
    for cond in conds:
        resps_by_cond_4_ts_dec[cond] = group_ephys_resps('A-0', full_pattern_responses_4_ts_dec, event_features_all,
                                                         cond_filters[cond])[0]

    response_dict_w_label_4_ts_dec = {}
    for sess in resps_by_cond_4_ts_dec[conds[0]].keys():
        try:
            response_dict_w_label_4_ts_dec[sess] = {f'{cond}': resps_by_cond_4_ts_dec[cond][sess] for cond in conds}
        except KeyError:
            continue

    resp_width = response_dict_w_label_4_ts_dec[sessname]['rare'].shape[-1]
    resp_x_ser = np.linspace(dec_over_time_window[0], dec_over_time_window[1], resp_width)

    rare_freq_dec_over_time, rare_freq_dec_over_time_dict = decode_over_sliding_t(response_dict_w_label_4_ts_dec,window_size,
                                                                                  dec_over_time_window,
                                                                                  {'rare':0,'frequent':1},
                                                                                  ['rare','frequent'],
                                                                                  animals_to_use=hipp_animals)
    dec_over_time_plot = plt.subplots()
    dec_over_time_plot[1].plot(rare_freq_dec_over_time.mean(axis=0),)
    dec_over_time_plot[1].fill_between(rare_freq_dec_over_time.columns.tolist(),
                                       rare_freq_dec_over_time.mean(axis=0) - rare_freq_dec_over_time.sem(axis=0),
                                       rare_freq_dec_over_time.mean(axis=0) + rare_freq_dec_over_time.sem(axis=0),
                                       alpha=0.2)

    dec_over_time_plot[1].set_xlabel('time (s)')
    dec_over_time_plot[1].set_ylabel('accuracy')
    format_axis(dec_over_time_plot[1],vspan=[[t,t+0.15] for t in np.arange(0,1,0.25)])
    dec_over_time_plot[0].set_layout_engine('tight')
    dec_over_time_plot[0].show()


    # get active units
    event_responses_4_active_units = aggregate_event_reponses(sessions, events=None,  # [f'{pip}-0' for pip in 'ABCD']
                                               events2exclude=['trial_start',], window=window,
                                               pred_from_psth_kwargs={'use_unit_zscore': True, 'use_iti_zscore': False,
                                                                      'baseline': 0, 'mean': None, 'mean_axis': 0})
    active_units_by_pip = {pip: np.hstack([get_participation_rate(event_responses[sess][pip], [-0.1, 0.25],
                                                                  [0.1,0.25], 2, max_func=np.max)
                                           for sess in event_responses_4_active_units
                                           if any(e in sess for e in hipp_animals)])
                           for pip in concatenated_event_responses_hipp_only.keys()}
    activity_map_arr = np.array(list(active_units_by_pip.values()))

    partipation_rate_by_pip_plot = plt.subplots(len(active_units_by_pip))
    for ((pip, pip_map), pip_map_ax) in zip(active_units_by_pip.items(),partipation_rate_by_pip_plot[1]):
        pip_map_ax.hist(pip_map, bins='fd', density=True)
        pip_map_ax.set_title(pip)
        pip_map_ax.text(pip_map.mean(), 0.8, f'{pip_map.mean():.2f}')
    partipation_rate_by_pip_plot[0].set_layout_engine('tight')
    partipation_rate_by_pip_plot[0].set_size_inches(5, 10)
    partipation_rate_by_pip_plot[0].show()

    activity_map = plot_2d_array_with_subplots(activity_map_arr.T>0.5,plot_cbar=False,
                                               cmap='seismic',interpolation='none', norm=matplotlib.colors.PowerNorm(1))
    activity_map[1].set_xticks(np.arange(len(active_units_by_pip)))
    activity_map[1].set_yticklabels([])
    [activity_map[1].axvline(i + 0.5, c='k', lw=2) for i in range(len(active_units_by_pip))]
    activity_map[1].set_xticklabels([e.split('-')[0] for e in list(active_units_by_pip.keys())])
    activity_map[0].set_size_inches(2,6)
    activity_map[0].set_layout_engine('tight')
    activity_map[0].show()
    activity_map[0].savefig(aggr_figdir/'activity_map_hipp_only.pdf')


    # recursive sorting plot
    activity_map_bool_arr = activity_map_arr.T
    r_maps = [activity_map_bool_arr[np.argsort(activity_map_bool_arr[:,pip_i])[::-1]]
              for pip_i in range(activity_map_bool_arr.shape[1])]
    r_sorted_map = plot_2d_array_with_subplots(np.vstack(r_maps),plot_cbar=False, cmap='Purples',interpolation='none',
                                               norm=matplotlib.colors.PowerNorm(1))
    r_sorted_map[1].set_xticks(np.arange(len(active_units_by_pip)))
    r_sorted_map[1].set_yticklabels([])
    [r_sorted_map[1].axvline(i + 0.5, c='k', lw=0.5) for i in range(len(active_units_by_pip))]
    [r_sorted_map[1].axhline(i*activity_map_bool_arr.shape[0],c='k',lw=0.5) for i in range(len(active_units_by_pip))]
    r_sorted_map[1].set_xticklabels([e.split('-')[0] for e in list(active_units_by_pip.keys())])
    r_sorted_map[0].set_size_inches(2,6)
    r_sorted_map[0].set_layout_engine('tight')
    r_sorted_map[0].show()

    r_sorted_map[0].savefig(aggr_figdir/'activity_map_hipp_only_sorted_no_cbar.pdf')

    by_pip_activity_sums = (activity_map_arr>0.5).sum(axis=1)
    by_pip_activity_sums_plot = plt.subplots()
    by_pip_activity_sums_plot[1].bar(np.arange(len(by_pip_activity_sums)),by_pip_activity_sums,
                                     tick_label=[e.split('-')[0] for e in list(active_units_by_pip.keys())],
                                     fc='lightgrey',ec='k')

    by_pip_activity_sums_plot[0].set_layout_engine('tight')
    by_pip_activity_sums_plot[0].show()

    by_pip_activity_sums_plot[0].savefig(aggr_figdir/'activity_sums_hipp_only.pdf')

    active_units_by_across_pips_patt_only = [(np.argwhere(unit>0.5)) for unit in activity_map_arr[[0,1,2,3,],:].T]
    active_units_by_across_pips = [(np.argwhere(unit>0.5)) for unit in activity_map_arr.T]
    selective_active_units_plot = plt.subplots()
    selective_pips = [[pip_i in unit and len(unit)==1 for pip_i in range(4)]
                      for unit in active_units_by_across_pips_patt_only]
    selective_units_sum = np.array(selective_pips).sum(axis=0)
    print('Proportion of active units: ', [round(e/activity_map_arr.shape[1]*100,2) for e in selective_units_sum])
    selective_active_units_plot[1].bar(np.arange(4),selective_units_sum,
                                      tick_label=[e.split('-')[0] for e in list(active_units_by_pip.keys())[:4]],
                                      fc='lightgrey',ec='k')
    format_axis(selective_active_units_plot[1])
    selective_active_units_plot[1].locator_params(axis='y', nbins=4, tight=True, integer=True)
    selective_active_units_plot[0].set_layout_engine('tight')
    selective_active_units_plot[0].show()
    selective_active_units_plot[0].savefig(aggr_figdir/'selective_units.pdf')

    # plot active A unit
    # eg_A_units = np.where(np.array(selective_pips)[:,1]==True)[0]
    plot_cols = ['#2B0504', '#4C6085','#874000']
    eg_A_units = [218,331,714,2531][:-1]
    unit_psth = plt.subplots(ncols=len(eg_A_units),sharey=True,sharex=True, figsize=(5.5,2))
    for eg_A_unit,ax in zip(eg_A_units,unit_psth[1].flatten()):
        [ax.plot(np.linspace(-0.25,1,126),
                           concatenated_event_responses_4_psth[pip][eg_A_unit], lw=1,
                           label=pip, color=c,) for pip,c in zip(['A-0','X','base'],plot_cols)]
        [ax.fill_between(np.linspace(-0.25,1,126),
                         concatenated_event_responses_4_psth[pip][eg_A_unit] - concatenated_sem[pip][eg_A_unit],
                         concatenated_event_responses_4_psth[pip][eg_A_unit] + concatenated_sem[pip][eg_A_unit],
                         fc=c, alpha=0.2) for pip,c in zip(['A-0','X','base'],plot_cols)]
        ax.set_title(f'Unit {eg_A_unit}',fontsize=10)
        format_axis(ax,ylim=[-.5,1.5])
        ax.locator_params(axis='both', nbins=2, tight=True, integer=True)
        # ax.legend()
    unit_psth[0].set_layout_engine('tight')
    unit_psth[0].show()
    unit_psth[0].savefig(aggr_figdir/f'unit_psth_A_unit_all_units.pdf')

    cross_active_units = [[[all([p in unit for p in [pip_i,pip_j]]) and len(unit)==1
                            for pip_i in range(len(active_units_by_pip))]
                           for pip_j in range(len(active_units_by_pip))]
                           for unit in active_units_by_across_pips]
    cross_active_units_arr = np.array(cross_active_units)
    # cross_active_units_arr = cross_active_units_arr.transpose((2,0,1))
    cross_active_units_arr = cross_active_units_arr.sum(axis=0)
    print('Total active units: ',[np.sum(row) for row in cross_active_units_arr])
    print('Proportion active units: ',[np.sum(row)/activity_map_arr.shape[1] for row in cross_active_units_arr])
    # cross_active_units_arr = np.array([row/row_diag for row,row_diag in zip(cross_active_units_arr,
    #                                                                        np.diagonal(cross_active_units_arr))])

    cross_active_units_plot = plot_2d_array_with_subplots(cross_active_units_arr, plot_cbar=True,
                                                          cmap='RdPu',
                                                          norm=matplotlib.colors.PowerNorm(.4))
    cross_active_units_plot[1].set_xticks(np.arange(len(active_units_by_pip)))
    cross_active_units_plot[1].set_yticks(np.arange(len(active_units_by_pip)))
    cross_active_units_plot[1].set_xticklabels([e.split('-')[0] for e in list(active_units_by_pip.keys())])
    cross_active_units_plot[1].set_yticklabels([e.split('-')[0] for e in list(active_units_by_pip.keys())])
    cross_active_units_plot[1].invert_yaxis()
    cross_active_units_plot[0].set_layout_engine('tight')
    cross_active_units_plot[0].show()
    cross_active_units_plot[0].savefig(aggr_figdir / 'cross_active_units.pdf')

    # cross active units abcd only
    cross_active_units_abcd_only_plot = plot_2d_array_with_subplots(cross_active_units_arr[0:4,0:4], plot_cbar=True,
                                                          cmap='RdPu',
                                                          norm=matplotlib.colors.PowerNorm(.4))
    cross_active_units_abcd_only_plot[1].set_xticks(np.arange(len('ABCD')))
    cross_active_units_abcd_only_plot[1].set_yticks(np.arange(len('ABCD')))
    cross_active_units_abcd_only_plot[1].set_xticklabels([e.split('-')[0] for e in list(active_units_by_pip.keys())
                                                          if e.split('-')[0] in 'ABCD'])
    cross_active_units_abcd_only_plot[1].set_yticklabels([e.split('-')[0] for e in list(active_units_by_pip.keys())
                                                          if e.split('-')[0] in 'ABCD'])
    cross_active_units_abcd_only_plot[1].invert_yaxis()
    cross_active_units_abcd_only_plot[0].set_layout_engine('tight')
    cross_active_units_abcd_only_plot[0].show()
    cross_active_units_abcd_only_plot[0].savefig(aggr_figdir / 'cross_active_units_abcd_only.pdf')

    # for conds

    rare_freq_sess = np.intersect1d(list(full_resps_by_cond_by_pip['A-0_rare'].keys()),
                                    list(full_resps_by_cond_by_pip['A-0_rare'].keys())).tolist()
    rare_freq_sess.remove('DO79_240606a')
    pips_by_cond = [pip for pip in [f'{p}-0_{cond}' for p in 'A' for cond in ['rare','frequent']]]
    concat_rare_freq_hipp_only = {
        e: np.concatenate([full_resps_by_cond_by_pip[e][sessname].mean(axis=0) for sessname in rare_freq_sess
                           if any([animal in sessname for animal in hipp_animals])])
        for e in pips_by_cond}
    concat_rare_freq_sem_hipp_only = {
        e: np.concatenate([sem(full_resps_by_cond_by_pip[e][sessname]) for sessname in rare_freq_sess
                           if any([animal in sessname for animal in hipp_animals])])
        for e in pips_by_cond}
    x_ser = np.round(np.linspace(*full_patt_window,concat_rare_freq_hipp_only[f'A-0_rare'].shape[-1]),2)

    # pca
    full_pattern_pca = PopPCA({'by_rate':concat_rare_freq_hipp_only})
    # full_pattern_pca.eig_vals[2][0].show()
    full_pattern_pca.get_trial_averaged_pca(standardise=False,n_components=15)
    full_pattern_pca.get_projected_pca_ts(standardise=False)

    full_pattern_pca.plot_pca_ts([-1,2],fig_kwargs={'figsize':(120,8)},plot_separately=False,n_comp_toplot=15,
                                 lss=['-', '--', '-', '--'], plt_cols=['C' + str(i) for i in [0, 0, 1, 1]]
                                 )
    [[ax.axvspan(t, t+0.15, color='grey', alpha=0.1) for t in np.arange(0,1,0.25)]
     for ax in full_pattern_pca.pca_ts_plot[1].flatten()]
    full_pattern_pca.pca_ts_plot[0].set_size_inches(150, 10)
    full_pattern_pca.pca_ts_plot[0].set_layout_engine('tight')
    full_pattern_pca.pca_ts_plot[0].show()
    full_pattern_pca.pca_ts_plot[0].savefig(f'full_pattern_pca_ts_plot_aggregate_sessions.pdf')
    #plot 3d projections
    full_pattern_pca.plot_3d_pca_ts('by_rate',[-1,2],x_ser=x_ser,smoothing=10,pca_comps_2plot=[0,2,1],t_end=1,
                                    plot_out_event=True)
    full_pattern_pca.proj_3d_plot.savefig(f'full_pattern_pca_3d_plot_aggregate_sessions.pdf')
    # get [0,1,2] components can compute euc distance over time
    pca_ts_0_1_2 = {e: full_pattern_pca.projected_pca_ts_by_cond['by_rate'][e][[0, 1, 2]]
                    for e in full_pattern_pca.projected_pca_ts_by_cond['by_rate'].keys()}
    euc_dist =np.linalg.norm(pca_ts_0_1_2['A-0_rare']- pca_ts_0_1_2['A-0_frequent'],axis=0)
    euc_dist_plot = plt.subplots()
    # causally smooth euc dist
    euc_dist = np.convolve(euc_dist, np.ones(25)/25, mode='same')
    euc_dist_plot[1].plot(x_ser,euc_dist)
    format_axis(euc_dist_plot[1],vspan=[[t, t+0.15] for t in np.arange(0,1,0.25)],
                xlim=[-0.75,1.75],xticks=np.arange(0,1,0.25),xticklabels=np.arange(0,1,0.25))
    euc_dist_plot[1].set_title('Euclidean distance between rare and frequent')
    euc_dist_plot[1].set_xlabel('Time')
    euc_dist_plot[1].set_ylabel('Euclidean distance')
    # euc_dist_plot[0].set_layout_engine('tight')
    euc_dist_plot[0].show()
    # save fig
    euc_dist_plot[0].savefig(f'euc_dist_aggregate_sessions.pdf')




    active_units_by_pip_rare_freq = {pip: np.hstack([get_participation_rate(full_resps_by_cond_by_pip[pip][sess],
                                                                            full_patt_window,
                                                                            [0.1, 0.25], 2, max_func=np.max)
                                                     for sess in rare_freq_sess if any(e in sess for e in hipp_animals)
                                                     if any(e in sess for e in hipp_animals)])
                           for pip in pips_by_cond}
    rare_freq_active_map = np.array(list(active_units_by_pip_rare_freq.values()))
    rare_freq_active_map_by_cond = [rare_freq_active_map[np.arange(0, len(rare_freq_active_map), 2)+i ] for i in range(2)]
    [print(f'{ttest_ind(rare_freq_active_map_by_cond[0][i], rare_freq_active_map_by_cond[1][i])},'
           f' means {np.mean(rare_freq_active_map_by_cond[0][i])}, {np.mean(rare_freq_active_map_by_cond[1][i])}')
     for i in range(len(rare_freq_active_map_by_cond[0]))]
    participation_rate_boxplot = plt.subplots()
    participation_rate_boxplot[1].boxplot([e for e in rare_freq_active_map])
    participation_rate_boxplot[1].set_xticks(np.arange(0,8,2)+1.5)
    participation_rate_boxplot[1].set_xticklabels(list('ABCD'))
    participation_rate_boxplot[0].show()

    participation_rate_boxplot[0].savefig(aggr_figdir / 'participation_rate_boxplot.pdf')

    r_maps = [rare_freq_active_map.T[np.argsort(rare_freq_active_map.T[:, pip_i])[::-1]]
              for pip_i in [0]]
    r_sorted_map = plot_2d_array_with_subplots(np.vstack(r_maps), plot_cbar=True, cmap='Purples', interpolation='none',
                                               norm=matplotlib.colors.PowerNorm(1))
    r_sorted_map[1].set_xticks(np.arange(len(rare_freq_active_map)))
    r_sorted_map[1].set_yticklabels([])
    [r_sorted_map[1].axvline(i + 0.5, c='k', lw=0.5) for i in range(len(rare_freq_active_map))]
    [r_sorted_map[1].axhline(i * rare_freq_active_map.shape[0], c='k', lw=0.5) for i in
     range(len(r_maps))]
    r_sorted_map[1].set_xticklabels([e.split('-')[0] for e in list(active_units_by_pip_rare_freq.keys())])
    r_sorted_map[0].set_size_inches(4, 3)
    r_sorted_map[0].set_layout_engine('tight')
    r_sorted_map[0].show()

    # example units
    # eg_rare_units = np.argsort(rare_freq_active_map.T[:, 1])[::-1][:20]  # 1393
    eg_rare_units = [1393,865]
    for unit in eg_rare_units:
        eg_unit_plot = plt.subplots()
        [eg_unit_plot[1].plot(np.linspace(*full_patt_window,concat_rare_freq_hipp_only[f'A-0_{cond}'].shape[-1]),
                              concat_rare_freq_hipp_only[f'A-0_{cond}'][unit],label=cond,
                              c='darkblue' if cond == 'rare' else 'darkgreen',)
         for cond in ['rare', 'frequent']]
        # sem
        [eg_unit_plot[1].fill_between(np.linspace(*full_patt_window, concat_rare_freq_hipp_only[f'A-0_{cond}'].shape[-1]),
                         concat_rare_freq_hipp_only[f'A-0_{cond}'][unit] - concat_rare_freq_sem_hipp_only[f'A-0_{cond}'][unit],
                         concat_rare_freq_hipp_only[f'A-0_{cond}'][unit] + concat_rare_freq_sem_hipp_only[f'A-0_{cond}'][unit],
                         fc=c, alpha=0.2)
         for cond,c in zip(['rare', 'frequent'],['darkblue', 'darkgreen'])]
        format_axis(eg_unit_plot[1])
        eg_unit_plot[1].locator_params(axis='both', nbins=2, tight=True, integer=True)
        # eg_unit_plot[1].set_xlabel('time (s)')
        # eg_unit_plot[1].set_ylabel('response (a.u.)')
        # eg_unit_plot[1].legend()
        # eg_unit_plot[1].set_title(f'unit {unit}')
        eg_unit_plot[0].set_size_inches(2, 1.5)
        eg_unit_plot[0].set_layout_engine('constrained')
        eg_unit_plot[0].show()
        eg_unit_plot[0].savefig(aggr_figdir / f'eg_unit_{unit}_by{"_".join(conds)}.pdf')

    cross_cond_units = np.all([rare_freq_active_map.T[:, cond_i] > 0.5 for cond_i, _ in enumerate(['rare', 'frequent'])],
                              axis=0)
    mean_resps_by_cond = {f'A-0_{cond}': concat_rare_freq_hipp_only[f'A-0_{cond}'][cross_cond_units]
                           for cond_i, cond in enumerate(['rare', 'frequent'])}
    mean_resps_by_cond = {
        k: pd.DataFrame(resps,columns=x_ser).T.rolling(25).mean().T
        for k,resps in mean_resps_by_cond.items()}
    rare_v_freq_ts_plot = plt.subplots()
    [rare_v_freq_ts_plot[1].plot(np.linspace(*full_patt_window,concat_rare_freq_hipp_only[f'A-0_rare'].shape[-1]),
                                 mean_resps_by_cond[f'A-0_{cond}'].mean(axis=0),
                                 label=cond,c='darkblue' if cond == 'rare' else 'darkgreen',)
     for cond in ['rare', 'frequent']]
    [rare_v_freq_ts_plot[1].fill_between(np.linspace(*full_patt_window, mean_resps_by_cond[pip].shape[-1]),
                     mean_resps_by_cond[pip].mean(axis=0) - sem(mean_resps_by_cond[pip]),
                     mean_resps_by_cond[pip].mean(axis=0) + sem(mean_resps_by_cond[pip]),
                     fc=c, alpha=0.2) for pip, c in zip(pips_by_cond, ['darkblue', 'darkgreen'])]
    # rare_v_freq_ts_plot[1].set_title(f'{unit}')
    # rare_v_freq_ts_plot[1].locator_params(axis='y', nbins=3, tight=True)
    format_axis(rare_v_freq_ts_plot[1],vspan=[[t,t+0.15] for t in np.arange(0,1,0.25)], vlines=[0])
    rare_v_freq_ts_plot[1].set_ylabel('Firing rate (a.u.)')
    rare_v_freq_ts_plot[1].set_xlabel('Time from pattern onset (s)')
    # rare_v_freq_ts_plot[1].legend()
    rare_v_freq_ts_plot[0].set_size_inches(3, 2)
    rare_v_freq_ts_plot[0].show()
    rare_v_freq_ts_plot[0].savefig(aggr_figdir / f'rare_v_freq_by{"_".join(conds)}.pdf')

    diff_window_s = [0.0,1]
    diff_window_idx = [np.where(x_ser==t)[0][0] for t in diff_window_s]
    cond_diff_by_unit = mean_resps_by_cond['A-0_rare'].loc[:,diff_window_s[0]:diff_window_s[1]].max(axis=1) - \
                        mean_resps_by_cond['A-0_frequent'].loc[:,diff_window_s[0]:diff_window_s[1]].max(axis=1)
    all_mean_resps = np.vstack(list(mean_resps_by_cond.values()))
    shuffle_diff_by_unit = [np.subtract(*[e[:,diff_window_idx[0]:diff_window_idx[1]].max(axis=1)
                                          for e in np.array_split(np.random.permutation(all_mean_resps),2)])
                            for _ in range(1000)]

    shuffle_diff_by_unit = np.mean(shuffle_diff_by_unit, axis=0)

    #plot scatter
    cond_max_scatter = plt.subplots()
    cond_max_scatter[1].scatter(
        mean_resps_by_cond[f'A-0_frequent'].loc[:,diff_window_s[0]:diff_window_s[1]].max(axis=1),
        mean_resps_by_cond[f'A-0_rare'].loc[:,diff_window_s[0]:diff_window_s[1]].max(axis=1),
        alpha=0.5)
    format_axis(cond_max_scatter[1])
    cond_max_scatter[1].locator_params(axis='both', nbins=6)
    cond_max_scatter[1].set_xlabel('frequent mean response')
    cond_max_scatter[1].set_ylabel('rare mean response')
    # plot unity line
    cond_max_scatter[1].plot(cond_max_scatter[1].get_xlim(), cond_max_scatter[1].get_ylim(), ls='--', c='k')

    cond_max_scatter[0].set_size_inches(2, 2)
    cond_max_scatter[0].show()
    cond_max_scatter[0].savefig(aggr_figdir / f'unit_resps_scatter{"_".join(conds)}.pdf')

    # plot boxplot of cond diff
    cond_diff_by_unit_plot = plt.subplots()
    bins2use = np.histogram(cond_diff_by_unit.values,bins='fd',density=False)
    [cond_diff_by_unit_plot[1].hist(data, bins=bins2use[1],density=False,alpha=0.9,fc=c,ec='k',lw=0.05)
     for data,c in zip([cond_diff_by_unit.values, shuffle_diff_by_unit][:1],['#B28B84','grey'])]
    # cond_diff_by_unit_plot[1].boxplot([cond_diff_by_unit,shuffle_diff_by_unit],showmeans=False,
    #                                   bootstrap=1000,whis=[1,99],
    #                                   )
    format_axis(cond_diff_by_unit_plot[1], vlines=np.percentile(shuffle_diff_by_unit, [0, 100]).tolist(),lw=1.5,ls='-')
    cond_diff_by_unit_plot[1].set_ylabel('Frequency')
    cond_diff_by_unit_plot[1].set_xlabel('\u0394firing rate (rare - frequent)')

    cond_diff_by_unit_plot[0].set_size_inches(2, 2)
    cond_diff_by_unit_plot[0].show()
    cond_diff_by_unit_plot[0].savefig(aggr_figdir / f'cond_diff_by{"_".join(conds)}.pdf')
    print(f'1 sample ttest: {ttest_1samp(cond_diff_by_unit, 0)}, mean = {np.mean(cond_diff_by_unit)}')
    print(f'independent ttest from shuffle: {ttest_ind(cond_diff_by_unit, shuffle_diff_by_unit,equal_var=False,alternative="greater")}, '
          f'\n shuffle mean = {np.mean(shuffle_diff_by_unit)}')
    print(f'{(cond_diff_by_unit.values>np.percentile(shuffle_diff_by_unit, 100)).sum()/len(cond_diff_by_unit)} '
          f'units exceed 100th percentile')
    print(f'{(cond_diff_by_unit.values<np.percentile(shuffle_diff_by_unit, 0)).sum()/len(cond_diff_by_unit)} '
          f'units less than 0th percentile')



    dec_over_time_window = full_patt_window
    full_pattern_responses_4_ts_dec = {sess: {pip: resps_by_cond_by_pip[pip][sess] for pip in pips_by_cond}
                                       for sess in rare_freq_sess}
    window_size = 0.25
    resp_width = full_pattern_responses_4_ts_dec[rare_freq_sess[0]][pips_by_cond[0]].shape[-1]
    resp_x_ser = np.linspace(dec_over_time_window[0], dec_over_time_window[1], resp_width)

    pip_dec_df, pip_dec_dict = decode_over_sliding_t(full_pattern_responses_4_ts_dec,0.25,dec_over_time_window,
                                                     {p:i for i,p in enumerate(pips_by_cond)},pips_by_cond,hipp_animals)
    rare_freq_dec_ts_plot = plt.subplots()
    rare_freq_dec_ts_plot[1].plot(pip_dec_df.mean(axis=0), color='k')
    rare_freq_dec_ts_plot[1].fill_between(pip_dec_df.columns.tolist(),
                                          pip_dec_df.mean(axis=0) - pip_dec_df.sem(axis=0),
                                          pip_dec_df.mean(axis=0) + pip_dec_df.sem(axis=0),
                                          color='k', alpha=0.25
                                          )
    rare_freq_dec_ts_plot[1].set_xlabel('time (s)')
    rare_freq_dec_ts_plot[1].set_ylabel('decoding accuracy')
    rare_freq_dec_ts_plot[1].set_title('')
    format_axis(rare_freq_dec_ts_plot[1], hlines=[0.5], vspan=[[t, t + 0.15] for t in np.arange(0, 1, 0.25)])

    rare_freq_dec_ts_plot[0].set_layout_engine('tight')
    rare_freq_dec_ts_plot[0].set_size_inches((3, 2))
    rare_freq_dec_ts_plot[0].show()
    # rare_freq_dec_ts_plot[0].savefig(aggr_figdir / f'rare_freq_dec_ts{"_".join(conds)}.pdf')
