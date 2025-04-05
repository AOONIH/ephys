import argparse
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import ttest_ind, ttest_1samp, sem
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics.pairwise import cosine_similarity

from aggregate_ephys_funcs import *
from behviour_analysis_funcs import get_all_cond_filts
from ephys_analysis_funcs import posix_from_win, plot_2d_array_with_subplots, plot_psth, format_axis, plot_sorted_psth
from neural_similarity_funcs import plot_similarity_mat, compare_pip_sims_2way
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

    cond_filts = get_all_cond_filts()

    all_sess_info = session_topology.query('sess_order=="main" ')
    all_sess_td_df = load_aggregate_td_df(all_sess_info,home_dir,cond_filts['deviant_C'])
    sessions2use = all_sess_td_df.index.get_level_values('sess').unique()
    # sessnames = [Path(sess_info['sound_bin']).stem.replace('_SoundData', '') for _, sess_info in
    #              sessions2use.iterrows()]

    pkldir = ceph_dir / posix_from_win(args.pkldir)
    pkls = list(pkldir.glob('*pkl'))
    pkls2load = [pkl for pkl in pkls if Path(pkl).stem in sessions2use]
    sessions = load_aggregate_sessions(pkls2load,)

    # [sess for sess in sessions2use if not any([sess in Path(pkl).stem for pkl in pkls2load])]
    hipp_animals = ['DO79','DO80','DO81','DO82']

    [sessions.pop(sess) for sess in list(sessions.keys()) if 'A-1' not in list(sessions[sess].sound_event_dict.keys())]
    [sessions.pop(sess) for sess in list(sessions.keys()) if 'D-1' not in list(sessions[sess].sound_event_dict.keys())]
    # [sessions.pop(sess) for sess in list(sessions.keys()) if 'A-2' in list(sessions[sess].sound_event_dict.keys())]
    # [sessions.pop(sess) for sess in list(sessions.keys()) if not any(animal in sess for animal in hipp_animals)]
    [print(sess,sessions[sess].sound_event_dict.keys()) for sess in list(sessions.keys()) ]


    # update styles
    plt.style.use('figure_stylesheet.mplstyle')
    window = (-0.1, 0.25)

    event_responses = aggregate_event_reponses(sessions, events=None,  # [f'{pip}-0' for pip in 'ABCD']
                                               events2exclude=['trial_start',], window=window,
                                               pred_from_psth_kwargs={'use_unit_zscore': True, 'use_iti_zscore': False,
                                                                      'baseline': 0, 'mean': None, 'mean_axis': 0})
    concatenated_event_responses = {
        e: np.concatenate([event_responses[sessname][e].mean(axis=0) for sessname in event_responses
                           if any([animal in sessname for animal in hipp_animals])])
        for e in list(event_responses.values())[0].keys()}

    sessnames = list(event_responses.keys())
    event_features = aggregate_event_features(sessions, events=[f'{pip}-{pip_i}' for pip in 'ABCD' for pip_i in [0,1]],
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
        e: np.concatenate([full_pattern_responses[sessname][e][100::10][:10].mean(axis=0) for sessname in full_pattern_responses])
        if e == 'A-0' else np.concatenate([full_pattern_responses[sessname][e].mean(axis=0) for sessname in full_pattern_responses])
        for e in [e for e in concatenated_event_responses.keys() if 'A' in e]}

    events_by_property = {
        'ptype_i': {pip: 0 if int(pip.split('-')[1]) < 1 else 1 for pip in [f'{p}-{i}' for i in range(2) for p in 'ABCD']},
    }

    pip_desc = sessions[list(sessions.keys())[0]].pip_desc
    events_by_property['id'] = {pip: pip_desc[pip]['idx']
                                for pip in list(events_by_property['ptype_i'])
                                if pip.split('-')[0] in 'ABCD'}
    events_by_property['position'] = {pip: ord(pip.split('-')[0]) - ord('A') + 1
                                      for pip in list(events_by_property['ptype_i'])
                                      if pip.split('-')[0] in 'ABCD'}
    events_by_property['group'] = {pip: 0 if int(pip.split('-')[1]) <1 else 1 for pip in list(events_by_property['ptype_i'])}
    # events_by_property['id'] = {pip: pip if events_by_property['ptype_i'][pip] == 0 else
    #                             f'{"ABBA"[events_by_property["position"][pip]-1]}-{pip.split("-")[1]}'
    #                             for pip in list(events_by_property['ptype_i'])
    #                             if pip.split('-')[0] in 'ABCD'}

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
                                                                        events_by_property, ['ptype_i'],
                                                                        concatenate_flag=False)
    dict_4_pca = {'by_type': {'normal': concatenated_full_pattern_responses['A-0'],
                              'deviant': concatenated_full_pattern_responses['A-1']}}
    full_pattern_pca = PopPCA(dict_4_pca)
    # full_pattern_pca.eig_vals[2][0].show()
    full_pattern_pca.get_trial_averaged_pca(standardise=True)
    full_pattern_pca.get_projected_pca_ts(standardise=True)
    full_pattern_pca.plot_pca_ts([-1, 2], fig_kwargs={'figsize': (120, 8)}, plot_separately=False, n_comp_toplot=15,
                                 lss=['-', '--', '-', '--'], plt_cols=['C' + str(i) for i in [0, 0, 1, 1]]
                                 )
    # plot 3d projections
    x_ser = np.round(np.linspace(-1,2, concatenated_full_pattern_responses['A-0'].shape[-1]), 2)
    full_pattern_pca.plot_3d_pca_ts('by_type', [-1, 2], x_ser=x_ser, smoothing=10, pca_comps_2plot=[3,1,0], t_end=1,
                                    plot_out_event=False,
                                    scatter_times=[0.65],scatter_kwargs={'marker':'*','s':50,'c':'k'})
    # save 3d plot
    full_pattern_pca.proj_3d_plot.savefig(f'full_pattern_norm_dev_pca_3d_plot_aggregate_sessions.pdf')
    full_pattern_pca.plot_2d_pca_ts('by_type', [-1, 2], x_ser=x_ser, smoothing=10, pca_comps_2plot=[3,1], t_end=1,
                                    plot_out_event=False,
                                    scatter_times=[0.5],scatter_kwargs={'marker':'*','s':50,'c':'k'})


    # get [0,1,2] components can compute euc distance over time
    pca_ts_0_1_2 = {e: full_pattern_pca.projected_pca_ts_by_cond['by_type'][e][[3,1,0]]
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

    events = [e for e in concatenated_event_responses.keys() if 'A' in e]
    resps2use = full_pattern_responses

    # resps2use = full_pattern_responses
    abstraction_figdir = ceph_dir / 'Dammy' / 'figures' / 'normdev_aggregate_figs'
    if not abstraction_figdir.is_dir():
        abstraction_figdir.mkdir()


    # decode events
    pips_as_ints = {pip: pip_i for pip_i, pip in enumerate([f'{pip}-{pip_i}' for pip in 'ABCD' for pip_i in range(2)])}
    # train on A decode on B C D
    sessname = list(resps2use.keys())[0]
    decoders_dict = {}
    patt_is = [0,1]
    bad_dec_sess = set()
    for sessname in tqdm(list(resps2use.keys()), total=len(resps2use), desc='decoding across sessions'):
        if not any(e in sessname for e in ['DO79','DO81']):
            continue
        for p in 'ABCD':
            xys = [(event_responses[sessname][f'{p}-{pip_i}'][150:200][::3] if pip_i==0 else event_responses[sessname][f'{p}-{pip_i}'],
                    np.full_like(event_responses[sessname][f'{p}-{pip_i}'][:,0,0], pips_as_ints[f'{p}-{pip_i}']))
                   for pip_i in patt_is]
            ys = [np.full(xy[0].shape[0],pips_as_ints[f'{p}-{pip_i}']) for xy,pip_i in zip(xys,patt_is)]
            xs = np.vstack([xy[0][:,:,15:].mean(axis=-1) for xy in xys])
            # xs = [xy[0][:,:,15:].mean(axis=-1) for xy in xys]
            ys = np.hstack(ys)
            # if np.unique(ys).shape[0] < len(patt_is):
            #     continue
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
                      dec['data'].cm is not None and any([animal in dec_name for animal in hipp_animals])]
                     for pip in 'AD']
    all_cms_by_pip_arr = np.array(all_cms_by_pip)
    all_cms_by_pip_plot = plt.subplots(ncols=len(all_cms_by_pip))
    cmap_norm = TwoSlopeNorm(vmin=np.quantile(all_cms_by_pip_arr,0.1), vmax=np.quantile(all_cms_by_pip_arr,0.9),
                             vcenter=1/all_cms_by_pip_arr.shape[-1])
    [ax.invert_yaxis() for ax in all_cms_by_pip_plot[1]]
    all_cms_by_pip_plot[0].set_size_inches(10, 6)
    all_cms_by_pip_plot[0].show()
    # all_cms_by_pip_plot[0].savefig(abstraction_figdir / 'decoding_cms_across_pips_dev_ABBA1.pdf')
    for pip, pip_cm in zip('AD',all_cms_by_pip_arr):
        cm_plot_by_pip = plot_aggr_cm(pip_cm, im_kwargs={'norm': cmap_norm}, labels=[f'{pip}_{cond}'
                                                                                     for cond in ['norm','dev']],
                                           colorbar=True, cmap='bwr',figsize=(3,3))
        # cm_plot_by_pip[0].set_layout_engine('constrained')
        cm_plot_by_pip[0].show()
        cm_plot_by_pip[0].savefig(abstraction_figdir / f'decoding_normdev_across_pips_{pip}.pdf')

    # rolling window of norm dev decoding
    dec_over_time_window = [-1,1.5]
    full_pattern_responses_4_ts_dec = aggregate_event_reponses(sessions, events=[e for e in concatenated_event_responses.keys()
                                                                        if 'A' in e],
                                                      events2exclude=['trial_start'], window=dec_over_time_window,
                                                      pred_from_psth_kwargs={'use_unit_zscore': True,
                                                                             'use_iti_zscore': False,
                                                                             'baseline': 0, 'mean': None,
                                                                             'mean_axis': 0})
    window_size = 25
    resp_width = full_pattern_responses_4_ts_dec[sessname]['A-0'].shape[-1]
    resp_x_ser = np.linspace(*dec_over_time_window,resp_width)
    # decode events
    # train on A decode on B C D
    norm_dev_figdir = dev_ABBA1_figdir.parent / 'normdev_aggregate_figs'
    if not norm_dev_figdir.is_dir():
        norm_dev_figdir.mkdir()
    normdev_ts_dec_dict = {}
    patt_is = [0, 1]
    bad_dec_sess = set()
    for sessname in tqdm(list(full_pattern_responses_4_ts_dec.keys()), total=len(full_pattern_responses_4_ts_dec),
                         desc='decoding across sessions'):
        # if not any(e in sessname for e in ['DO79', 'DO81']):
        if not any(e in sessname for e in ['DO82']):
            continue

        for t in tqdm(range(resp_width - window_size), total=resp_width - window_size, desc='decoding across time'):
            xys = [(
                   full_pattern_responses_4_ts_dec[sessname][f'A-{pip_i}'][150:200][::3] if pip_i == 0 else full_pattern_responses_4_ts_dec[sessname][
                       f'A-{pip_i}'],
                   np.full_like(full_pattern_responses_4_ts_dec[sessname][f'A-{pip_i}'][:, 0, 0], pips_as_ints[f'A-{pip_i}']))
                   for pip_i in patt_is]
            ys = [np.full(xy[0].shape[0], pips_as_ints[f'A-{pip_i}']) for xy, pip_i in zip(xys, patt_is)]
            xs = np.vstack([xy[0][:, :, t:t + window_size].mean(axis=-1) for xy in xys])
            # xs = [xy[0][:,:,15:].mean(axis=-1) for xy in xys]
            ys = np.hstack(ys)
            # if np.unique(ys).shape[0] < len(patt_is):
            #     continue
            try:
                normdev_ts_dec_dict[f'{sessname}-{t}:{t+window_size}s'] = decode_responses(xs, ys, n_runs=100)
            except ValueError:
                print(f'{sessname} failed')
                bad_dec_sess.add(sessname)
                continue
    # [decoders_dict[dec_name]['data'].plot_confusion_matrix([f'{pip}-{pip_i}' for pip in 'D' for pip_i in patt_is])
    #  for dec_name in decoders_dict.keys()]
    sess2use = [dec_name.split('-')[0] for dec_name in normdev_ts_dec_dict.keys()]
    norm_dev_accs_ts_dict = {sessname:{t: np.mean(normdev_ts_dec_dict[f'{sessname}-{t}:{t+window_size}s']['data'].accuracy)
                                       for t in range(resp_width - window_size)}
                             for sessname in tqdm(list(resps2use.keys()),)
                             if sessname in sess2use}

    norm_dev_accs_ts_df = pd.DataFrame(norm_dev_accs_ts_dict).T
    norm_dev_accs_ts_df.columns = np.round(resp_x_ser[window_size:],2)

    # _ = decode_over_sliding_t(full_pattern_responses_4_ts_dec,0.25,dec_over_time_window,pips_as_ints,['A-0','A-1'],
    #                           animals_to_use=['DO82'])

    norm_dev_accs_ts_plot = plt.subplots()
    norm_dev_accs_ts_plot[1].plot(norm_dev_accs_ts_df.mean(axis=0),color='k')
    norm_dev_accs_ts_plot[1].fill_between(norm_dev_accs_ts_df.columns.tolist(),
                                          norm_dev_accs_ts_df.mean(axis=0)-norm_dev_accs_ts_df.sem(axis=0),
                                          norm_dev_accs_ts_df.mean(axis=0)+norm_dev_accs_ts_df.sem(axis=0),
                                          color='k',alpha=0.25
                                          )
    norm_dev_accs_ts_plot[1].set_xlabel('time (s)')
    norm_dev_accs_ts_plot[1].set_ylabel('decoding accuracy')
    norm_dev_accs_ts_plot[1].set_title('')
    format_axis(norm_dev_accs_ts_plot[1],hlines=[0.5],vspan=[[t,t+0.15] for t in np.arange(0,1,0.25)])
    norm_dev_accs_ts_plot[1].axvspan(0.5,0.5+0.15,fc='red',alpha=0.1)

    norm_dev_accs_ts_plot[0].set_layout_engine('tight')
    norm_dev_accs_ts_plot[0].set_size_inches((3,2))
    norm_dev_accs_ts_plot[0].show()
    norm_dev_accs_ts_plot[0].savefig( norm_dev_figdir/ 'norm_dev_accs_ts_82_only.pdf')

    dec_acc_plot = plt.subplots()
    accuracy_across_sessions_all_df = pd.concat([pd.DataFrame.from_dict({f'{dec_name}_{sffx}': dec[sffx].accuracy
                                                                         for dec_name, dec in decoders_dict.items()})
                                                 for sffx in ['data','shuffled']],axis=1)
    accuracy_across_sessions = accuracy_across_sessions_all_df.mean(axis=0)
    accuracy_across_sessions_by_pip = {f'{pip}_{sffx}': [acc for sess, acc in accuracy_across_sessions.items()
                                                         if f'{pip}s' in sess and sffx in sess]
                                       for pip in 'ABCD' for sffx in ['data','shuffled']}
    for pip in 'ABCD':
        dec_vs_shuffled_plot = plt.subplots()
        [[dec_vs_shuffled_plot[1].scatter(sffx,accs,c='k') for idx,accs in accuracy_across_sessions.items()
         if sffx in idx and f'{pip}s' in idx]
         for sffx in ['data','shuffled']]
        dec_vs_shuffled_plot[1].set_title(pip)
        dec_vs_shuffled_plot[0].show()

    #
    # [print(ttest_1samp(acc, 1/all_cms_by_pip_arr.shape[-1], alternative='greater')) for acc in accuracy_across_sessions]
    # print(ttest_ind(accuracy_across_sessions[0][0],accuracy_across_sessions[1][0], alternative='two-sided'))
    # print(ttest_ind(accuracy_across_sessions[0][1],accuracy_across_sessions[1][1], alternative='two-sided'))
    #


    dec_acc_plot[1].boxplot(accuracy_across_sessions_by_pip.values(), labels=accuracy_across_sessions_by_pip.keys(),
                             patch_artist=False,
                             showmeans=True, meanprops=dict(mfc='k'), showfliers=False)

    dec_acc_plot[1].axhline(1/all_cms_by_pip_arr.shape[-1], color='k',ls='--',lw=1)

    dec_acc_plot[0].set_layout_engine('tight')
    dec_acc_plot[0].show()
    # dec_acc_plot[0].savefig(abstraction_figdir / 'decoding_across_sessions_dev_ABBA1.pdf')

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
        # decoder_acc_plot[0].savefig(abstraction_figdir / f'{lbl}_decoding_accuracy_dev_ABBA1.pdf')

        # ttest on fold accuracy
        ttest = ttest_ind(*[np.array(dec.fold_accuracy).flatten() for dec in decoder.values()],
                          alternative='greater', equal_var=True)
        print(f'{lbl} ttest: {ttest}')

    # decode normal vs deviant full pattern

    sessnames_normdev = list(full_pattern_responses)
    norm_dev_decoder_dict = {}
    patt_is = [0, 1]
    bad_dec_sess = set()
    p = 'A'
    for sessname in tqdm(sessnames_normdev, total=len(sessnames_normdev), desc='decoding across sessions'):
        # if not any(e in sessname for e in ['DO79', 'DO81']):
        if not any(e in sessname for e in ['DO82']):
            continue
        xys = [(
               full_pattern_responses[sessname][f'{p}-{pip_i}'][100:250][::5] if pip_i == 0 else
               full_pattern_responses[sessname][f'{p}-{pip_i}'],
               np.full_like(event_responses[sessname][f'{p}-{pip_i}'][:, 0, 0], pips_as_ints[f'{p}-{pip_i}']))
               for pip_i in patt_is]
        ys = [np.full(xy[0].shape[0], pips_as_ints[f'{p}-{pip_i}']) for xy, pip_i in zip(xys, patt_is)]
        xs = np.vstack([xy[0][:, :, 80:].mean(axis=-1) for xy in xys])
        ys = np.hstack(ys)
        if np.unique(ys).shape[0] < len(patt_is):
            continue
        try:
            norm_dev_decoder_dict[f'{sessname}-{p}s'] = decode_responses(xs, ys, n_runs=1000)
        except ValueError:
            print(f'{sessname}-{p}s failed')
            bad_dec_sess.add(sessname)
            continue

    norm_dev_accs_across_sessions_df = pd.concat([pd.DataFrame.from_dict({f'{dec_name}_{sffx}': dec[sffx].accuracy
                                                                         for dec_name, dec in norm_dev_decoder_dict.items()})
                                                 for sffx in ['data', 'shuffled']], axis=1)
    norm_dev_accs_across_sessions = norm_dev_accs_across_sessions_df.mean(axis=0)
    norm_dev_accs_across_sessions_by_pip = {f'{pip}_{sffx}': [acc for sess, acc in norm_dev_accs_across_sessions.items()
                                                         if f'{pip}s' in sess and sffx in sess]
                                            for pip in 'A' for sffx in ['data', 'shuffled']}
    # boxplot of accuracy across sessions

    norm_dev_accs_across_sessions_plot = plt.subplots()
    norm_dev_accs_across_sessions_plot[1].boxplot(norm_dev_accs_across_sessions_by_pip.values(),
                                                  labels=['data', 'shuffled'],
                                                  patch_artist=False, widths=0.3,
                                                  showmeans=False, meanprops=dict(mfc='k'), showfliers=False)
    norm_dev_accs_across_sessions_plot[1].axhline(1/all_cms_by_pip_arr.shape[-1], color='k',ls='--',lw=1)
    format_axis(norm_dev_accs_across_sessions_plot[1])
    norm_dev_accs_across_sessions_plot[0].set_size_inches(1.75, 2)
    norm_dev_accs_across_sessions_plot[0].set_layout_engine('tight')
    norm_dev_accs_across_sessions_plot[0].show()
    norm_dev_accs_across_sessions_plot[0].savefig(norm_dev_figdir / 'norm_dev_accs_across_sessions_DO82_only.pdf')

    # ttest
    ttest = ttest_ind(norm_dev_accs_across_sessions_by_pip['A_data'], norm_dev_accs_across_sessions_by_pip['A_shuffled'],
                      alternative='greater', equal_var=True)
    print(f'normal vs dev full pattern ttest: {ttest}')


    # cosine sims across sessions
    event_names = list(pips_as_ints.keys())
    prop = 'position'
    non_prop = 'group'
    prop_sim_diff_by_sess = {}
    prop_sim_by_sess = {}
    for sessname in tqdm(list(event_responses.keys()), total=len(event_responses), desc='decoding across sessions'):
        event_resps_by_pip = [event_responses[sessname][e] for e in event_names]
        sim_mat = cosine_similarity([event_resps[:,:,-10:].mean(axis=-1).mean(axis=0)
                                     for event_resps in event_resps_by_pip])

        sim_mat_plot = plot_similarity_mat(sim_mat,event_names,cmap='Reds')
        sim_mat_plot[1].set_title(sessname)
        # sim_mat_plot[0].show()
        print(f'{sessname} mean sim',sim_mat[~np.eye(sim_mat.shape[0],dtype=bool)].reshape(sim_mat.shape[0],-1).mean())
        within_prop_idxs = [[events_by_property[prop][ee] == events_by_property[prop][e] for ee in event_names]
                            for e in event_names]
        within_prop_sim = [sim_mat[ei][e_idxs] for ei, e_idxs in enumerate(within_prop_idxs)]
        within_prop_sim_means = {e:within_prop_sim[ei][within_prop_sim[ei] != 1].mean()
                                 for ei, e in enumerate(event_names)}
        # non_prop_sim_idxs = [[events_by_property[non_prop][ee] == events_by_property[non_prop][e] for ee in event_names]
        #                      for e in event_names]
        non_prop_sim_idxs = [np.invert(within_prop_idxs[ei]) for ei in range(len(within_prop_idxs))]
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
                                  showfliers=True, showmeans=True, meanprops=dict(mfc='k'))
    prop_sim_diff_plot[1].set_title(f'Similarity difference of {prop} vs {non_prop}')
    prop_sim_diff_plot[1].axhline(0, color='k', ls='--')
    prop_sim_diff_plot[0].set_layout_engine('tight')
    prop_sim_diff_plot[0].show()

    # c pip only
    pip_C_sim_diff = {}
    p = 'C'
    pi = 1
    not_p = 'B'
    dev_C_pip = f'{p}-{pi}'
    same_pos = [f'{p}-{pi -1}',f'{p}-{pi +1}']
    same_pip = [f'{not_p}-{pi + 1}']
    sess2use = [sessname for sessname in event_responses.keys() if any([e in sessname for e in ['DO79','DO81']])]
    sim_mat_plots = plt.subplots(int(np.ceil(len(sess2use)/3)),3)
    for s_ax, sessname in zip(sim_mat_plots[1].flatten(),tqdm(sess2use,
                                      total=len(event_responses), desc='decoding across sessions')):
        if not any([e in sessname for e in ['DO79','DO81']]):
            continue
        if sessname not in pip_C_sim_diff:
            pip_C_sim_diff[sessname] = {}

        event_resps_by_pip = [event_responses[sessname][e] for e in [dev_C_pip] + same_pos + same_pip]
        sim_mat = cosine_similarity([event_resps[:,:,-10:].mean(axis=-1).mean(axis=0)
                                     for event_resps in event_resps_by_pip])
        self_sims = [compare_pip_sims_2way([e_resps[:, :, -10:]], mean_flag=True, n_shuffles=50)[0].mean(axis=0)[0, 1]
                     for e_resps in event_resps_by_pip]
        for ei,e in enumerate([dev_C_pip] + same_pos + same_pip):
            sim_mat[ei,ei] = self_sims[ei]
        # plot_similarity_mat(sim_mat,[f'dev {p}-{pi}', f'norm {p}', f'norm {same_pip[0].split("-")[0]}'],
        plot_similarity_mat(sim_mat,sum([[dev_C_pip], same_pos, same_pip],[]),
                            cmap='Reds', plot=(sim_mat_plots[0],s_ax),)
        s_ax.set_title(sessname)
        s_ax.set_yticklabels([])
        # sim_mat_plot[0].show()
        pip_C_sim_diff[sessname][f'{p}-{pi}'] = sim_mat[0, 1] - sim_mat[0, 2]

    pip_C_sim_diff_plot = plt.subplots()
    pip_C_sim_diff_plot[1].boxplot([[pip_C_sim_diff[sess][e] for sess in pip_C_sim_diff.keys()]
                                   for e in [f'{pp}-{pi}' for pp in p]], labels=[f'{pp}-{pi}' for pp in p],
                                  showfliers=True, showmeans=True, meanprops=dict(mfc='k'))
    pip_C_sim_diff_plot[1].set_title(f'Sim between deviant {p} with same postion or same pip')
    pip_C_sim_diff_plot[1].set_ylabel(f'Similarity difference \n sim(dev {p}, same pos) - sim(dev {p}, same pip)')
    pip_C_sim_diff_plot[1].axhline(0, color='k', ls='--')
    pip_C_sim_diff_plot[0].set_layout_engine('tight')
    pip_C_sim_diff_plot[0].show()
    # pip_C_sim_diff_plot[0].savefig(abstraction_figdir / f'{p}_normdev_pips_sim_diff.pdf')

    sim_mat_plots[0].set_size_inches(12,4*sim_mat_plots[1].shape[0])
    sim_mat_plots[0].set_layout_engine('tight')
    sim_mat_plots[0].show()
    sim_mat_plots[0].savefig(norm_dev_figdir / f'{p}_normdev_pips_compared_sim_mat_w_self_sim_diag.pdf')
    # ttest
    [print(f'{e}: {ttest_1samp([pip_C_sim_diff[sess][e] for sess in pip_C_sim_diff.keys()], 0, alternative="greater")}') for e in [f'{p}-1' for p in 'C']]

    # prop_sim_diff_plot[0].savefig(abstraction_figdir / f'{prop}_vs_{non_prop}_sim_diff_plot.pdf')

    pip_sim_diff = [[prop_sim_diff_by_sess[sess][e] for sess in prop_sim_diff_by_sess.keys()]
                                   for e in [f'{p}-1' for p in 'ABCD']]
    [print(f'{e}: {ttest_1samp(pip_sim_diff[ei], 0, alternative="greater")}') for ei, e in enumerate([f'{p}-1' for p in 'ABCD'])]

    # some psth plots
    psth_figdir = abstraction_figdir.parent/'psth_plots_aggr_sessions'
    if not psth_figdir.is_dir():
        psth_figdir.mkdir()
    full_pattern_responses_4_psth = aggregate_event_reponses(sessions, events=[e for e in concatenated_event_responses.keys()
                                                                        if 'A' in e],
                                                      events2exclude=['trial_start'], window=[-0.25, 1],
                                                      pred_from_psth_kwargs={'use_unit_zscore': True,
                                                                             'use_iti_zscore': False,
                                                                             'baseline': 0, 'mean': None,
                                                                             'mean_axis': 0})

    concat_full_patt_resps = {
        e: np.concatenate([full_pattern_responses_4_psth[sessname][e].mean(axis=0)
                           for sessname in full_pattern_responses_4_psth])
        for e in [e for e in concatenated_event_responses.keys() if 'A' in e]}

    psth_figdir = abstraction_figdir.parent/'psth_plots_aggr_sessions'
    for pip in [e for e in concatenated_event_responses.keys() if 'A' in e]:
        resp_mat = concat_full_patt_resps[pip]
        resp_mat_sorted = resp_mat[concat_full_patt_resps['A-0'][:,25:50].max(axis=1).argsort()[::-1]]
        cmap_norm = TwoSlopeNorm(vcenter=0, vmin=-5, vmax=5)
        all_resps_psth = plot_psth(resp_mat_sorted,pip,[-0.25,1],aspect=0.1,cmap='bwr',norm=cmap_norm,)
        all_resps_psth[0].set_layout_engine('tight')
        format_axis(all_resps_psth[1],vlines=np.arange(0,1,0.25).tolist(),ylabel='Unit #',
                    xlabel=f'Time from {pip} onset (s)')
        all_resps_psth[0].show()

        # all_resps_psth[0].savefig(psth_figdir / f'{pip}_all_resps_psth_abstraction_sessions.pdf')

        # format axis

    psth_figdir = abstraction_figdir.parent / 'psth_plots_aggr_sessions'
    for sorting in ['own','cross']:
        for pip in [e for e in concatenated_event_responses.keys() if 'A' in e]:
            for animal in hipp_animals:
                cmap_norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
                all_resps_psth = plot_sorted_psth(full_pattern_responses_4_psth,pip,pip if sorting=='own' else 'A-0',
                                                  window=[-0.25,1],sort_window=[0,1],
                                                  sessname_filter=animal,im_kwargs=dict(norm=cmap_norm,cmap='bwr'))
                all_resps_psth[0].set_layout_engine('tight')
                format_axis(all_resps_psth[1][1],vlines=np.arange(0,1,0.25).tolist(),ylabel='Unit #',
                            xlabel=f'Time from pattern onset (s)')
                format_axis(all_resps_psth[1][0],vlines=[0])
                all_resps_psth[1][0].set_ylim(-0.15,0.2)
                all_resps_psth[1][0].locator_params(axis='y', nbins=2)
                if pip == 'A-0':
                    all_resps_psth[1][1].set_ylabel('Unit #')
                all_resps_psth[0].set_layout_engine('tight')
                all_resps_psth[0].set_size_inches(3,3)
                all_resps_psth[0].show()

                all_resps_psth[0].savefig(psth_figdir / f'{pip}_{animal}_{pip}_resps_psth_aggr_normdev_{sorting}_sort_sessions.pdf')


    # example session plots
    eg_sess = 'DO81_240724a'
    for eg_sess in list(full_pattern_responses_4_psth.keys()):
        eg_resp = full_pattern_responses[eg_sess]['A-0'].mean(axis=0)
        # cmap_norm = TwoSlopeNorm(vcenter=0, vmin=np.percentile(eg_resp, 0), vmax=np.percentile(eg_resp, 95))
        eg_sess_psth = plot_sorted_psth(full_pattern_responses, 'A-0','A-1', [-0.25, 1],[0.1,1],
                                        sessname_filter=eg_sess, plot_ts=True,
                                        im_kwargs= dict(cmap='Reds', vmin=np.percentile(eg_resp, 0),vmax=np.percentile(eg_resp, 99)))
        eg_sess_psth[0].suptitle(eg_sess)
        eg_sess_psth[0].show()

    for eg_sess in list(full_pattern_responses_4_psth.keys()):
        eg_resp = event_responses[eg_sess]['X'].mean(axis=0)
        # cmap_norm = TwoSlopeNorm(vcenter=0, vmin=np.percentile(eg_resp, 0), vmax=np.percentile(eg_resp, 95))
        eg_sess_psth = plot_sorted_psth(event_responses, 'X','X', [-0.1, .25],[0.1,0.25],
                                        sessname_filter=eg_sess, plot_ts=True,
                                        im_kwargs= dict(cmap='Reds', vmin=np.percentile(eg_resp, 0),vmax=np.percentile(eg_resp, 99)))
        eg_sess_psth[0].suptitle(eg_sess)
        eg_sess_psth[0].show()

    # for conds

    rare_freq_sess = np.intersect1d(list(full_pattern_responses_4_ts_dec.keys()),
                                    list(full_pattern_responses_4_ts_dec.keys())).tolist()
    pips_by_cond = ['A-0', 'A-1']
    full_patt_window = dec_over_time_window
    concat_normdev_hipp_only = {
        e: np.concatenate([full_pattern_responses_4_ts_dec[sessname][e].mean(axis=0) for sessname in rare_freq_sess
                           if any([animal in sessname for animal in hipp_animals])])
        for e in pips_by_cond}
    concat_normdev_sem_hipp_only = {
        e: np.concatenate([sem(full_pattern_responses_4_ts_dec[sessname][e]) for sessname in rare_freq_sess
                           if any([animal in sessname for animal in hipp_animals])])
        for e in pips_by_cond}
    x_ser = np.round(np.linspace(*full_patt_window, concat_normdev_hipp_only[f'A-0'].shape[-1]), 2)

    active_units_by_pip_normdev = {pip: np.hstack([get_participation_rate(full_pattern_responses_4_ts_dec[sess][pip],
                                                                          full_patt_window,
                                                                          [0.1, 1], 2, max_func=np.max)
                                                   for sess in rare_freq_sess if
                                                   any(e in sess for e in hipp_animals)
                                                   if any(e in sess for e in hipp_animals)])
                                   for pip in pips_by_cond}
    normdev_active_map = np.array(list(active_units_by_pip_normdev.values()))
    normdev_active_map_by_cond = [normdev_active_map[np.arange(0, len(normdev_active_map), 2) + i] for i in
                                  range(len(pips_by_cond))]
    [print(f'{ttest_ind(normdev_active_map_by_cond[0][i], normdev_active_map_by_cond[1][i])},'
           f' means {np.mean(normdev_active_map_by_cond[0][i])}, {np.mean(normdev_active_map_by_cond[1][i])}')
     for i in range(len(normdev_active_map_by_cond[0]))]
    participation_rate_boxplot = plt.subplots()
    participation_rate_boxplot[1].boxplot([e for e in normdev_active_map])
    # participation_rate_boxplot[1].set_xticks(np.arange(0, len(pips_by_cond*2), 2) + 1.5)
    # participation_rate_boxplot[1].set_xticklabels(list('ABCD'))
    participation_rate_boxplot[0].show()

    participation_rate_boxplot[0].savefig(norm_dev_figdir / 'participation_rate_boxplot.pdf')

    r_maps = [normdev_active_map.T[np.argsort(normdev_active_map.T[:, pip_i])[::-1]]
              for pip_i in [0]]
    r_sorted_map = plot_2d_array_with_subplots(np.vstack(r_maps), plot_cbar=True, cmap='Purples',
                                               interpolation='none',
                                               norm=matplotlib.colors.PowerNorm(1))
    r_sorted_map[1].set_xticks(np.arange(len(normdev_active_map)))
    r_sorted_map[1].set_yticklabels([])
    [r_sorted_map[1].axvline(i + 0.5, c='k', lw=0.5) for i in range(len(normdev_active_map))]
    [r_sorted_map[1].axhline(i * normdev_active_map.shape[0], c='k', lw=0.5) for i in
     range(len(r_maps))]
    r_sorted_map[1].set_xticklabels([e.split('-')[0] for e in list(active_units_by_pip_normdev.keys())])
    r_sorted_map[0].set_size_inches(4, 3)
    r_sorted_map[0].set_layout_engine('tight')
    r_sorted_map[0].show()

    # example units
    eg_rare_units = np.argsort(normdev_active_map.T[:, 1])[::-1][:20]  # 1393
    # eg_rare_units = [1393, 865]
    for unit in eg_rare_units:
        eg_unit_plot = plt.subplots()
        [eg_unit_plot[1].plot(resp_x_ser, concat_normdev_hipp_only[pip][unit], label=pip,c=c)
         for pip,c in zip(pips_by_cond,['darkblue','darkred'])]
        # sem
        [eg_unit_plot[1].fill_between(
            resp_x_ser,
            concat_normdev_hipp_only[pip][unit] - concat_normdev_sem_hipp_only[pip][unit],
            concat_normdev_hipp_only[pip][unit] + concat_normdev_sem_hipp_only[pip][unit],
            fc=c, alpha=0.2)
         for pip, c in zip(pips_by_cond, ['darkblue', 'darkred'])]
        format_axis(eg_unit_plot[1])
        eg_unit_plot[1].locator_params(axis='both', nbins=2, tight=True, integer=True)
        # eg_unit_plot[1].set_xlabel('time (s)')
        # eg_unit_plot[1].set_ylabel('response (a.u.)')
        # eg_unit_plot[1].legend()
        # eg_unit_plot[1].set_title(f'unit {unit}')
        eg_unit_plot[0].set_size_inches(2, 1.5)
        eg_unit_plot[0].set_layout_engine('constrained')
        eg_unit_plot[0].show()
        eg_unit_plot[0].savefig(norm_dev_figdir / f'eg_unit_{unit}_by{"_".join(pips_by_cond)}.pdf')

    cross_cond_units = np.all(
        [normdev_active_map.T[:, cond_i] > 0.5 for cond_i, _ in enumerate(pips_by_cond)],
        axis=0)
    mean_resps_by_cond = {pip: concat_normdev_hipp_only[pip][cross_cond_units]
                          for cond_i, pip in enumerate(pips_by_cond)}
    mean_resps_by_cond = {
        k: pd.DataFrame(resps, columns=x_ser).T.rolling(25).mean().T
        for k, resps in mean_resps_by_cond.items()}
    normdev_ts_plot = plt.subplots()
    [normdev_ts_plot[1].plot(resp_x_ser,mean_resps_by_cond[pip].mean(axis=0),label=pip, c=c, )
     for pip,c in zip(pips_by_cond,['darkblue','darkred'])]
    [normdev_ts_plot[1].fill_between(resp_x_ser,
                                     mean_resps_by_cond[pip].mean(axis=0) - sem(mean_resps_by_cond[pip]),
                                     mean_resps_by_cond[pip].mean(axis=0) + sem(mean_resps_by_cond[pip]),
                                     fc=c, alpha=0.2) for pip, c in
     zip(pips_by_cond, ['darkblue', 'darkred'])]
    # rare_v_freq_ts_plot[1].set_title(f'{unit}')
    # rare_v_freq_ts_plot[1].locator_params(axis='y', nbins=3, tight=True)
    format_axis(normdev_ts_plot[1], vspan=[[t, t + 0.15] for t in np.arange(0, 1, 0.25)], vlines=[0])
    normdev_ts_plot[1].axvspan(0.5,0.5 + 0.15, color='r', alpha=0.1, )
    normdev_ts_plot[1].set_ylabel('Firing rate (a.u.)')
    normdev_ts_plot[1].set_xlabel('Time from pattern onset (s)')
    # rare_v_freq_ts_plot[1].legend()
    normdev_ts_plot[0].set_size_inches(3, 2)
    normdev_ts_plot[0].show()
    normdev_ts_plot[0].savefig(norm_dev_figdir / f'norm_dev_ts.pdf')

    diff_window_s = [0.5, 1.5]
    diff_window_idx = [np.where(x_ser == t)[0][0] for t in diff_window_s]
    cond_diff_by_unit = mean_resps_by_cond['A-1'].loc[:, diff_window_s[0]:diff_window_s[1]].max(axis=1) - \
                        mean_resps_by_cond['A-0'].loc[:, diff_window_s[0]:diff_window_s[1]].max(axis=1)
    all_mean_resps = np.vstack(list(mean_resps_by_cond.values()))
    shuffle_diff_by_unit = [np.subtract(*[e[:, diff_window_idx[0]:diff_window_idx[1]].max(axis=1)
                                          for e in np.array_split(np.random.permutation(all_mean_resps), 2)])
                            for _ in range(1000)]

    shuffle_diff_by_unit = np.mean(shuffle_diff_by_unit, axis=0)

    # plot scatter
    cond_max_scatter = plt.subplots()
    cond_max_scatter[1].scatter(
        mean_resps_by_cond[f'A-1'].loc[:, diff_window_s[0]:diff_window_s[1]].max(axis=1),
        mean_resps_by_cond[f'A-0'].loc[:, diff_window_s[0]:diff_window_s[1]].max(axis=1),
        alpha=0.5)
    format_axis(cond_max_scatter[1])
    # cond_max_scatter[1].locator_params(axis='both', nbins=6)
    cond_max_scatter[1].set_xlabel('deviant mean response')
    cond_max_scatter[1].set_ylabel('normal mean response')
    # plot unity line
    cond_max_scatter[1].plot(cond_max_scatter[1].get_xlim(), cond_max_scatter[1].get_ylim(), ls='--', c='k')

    cond_max_scatter[0].set_size_inches(2, 2)
    cond_max_scatter[0].show()
    cond_max_scatter[0].savefig(norm_dev_figdir / f'unit_resps_scatter{"_".join(pips_by_cond)}.pdf')

    # plot boxplot of cond diff
    cond_diff_by_unit_plot = plt.subplots()
    bins2use = np.histogram(cond_diff_by_unit.values, bins='fd', density=False)
    [cond_diff_by_unit_plot[1].hist(data, bins=bins2use[1], density=False, alpha=0.9, fc=c, ec='k', lw=0.05)
     for data, c in zip([cond_diff_by_unit.values, shuffle_diff_by_unit][:1], ['#B28B84', 'grey'])]
    # cond_diff_by_unit_plot[1].boxplot([cond_diff_by_unit,shuffle_diff_by_unit],showmeans=False,
    #                                   bootstrap=1000,whis=[1,99],
    #                                   )
    format_axis(cond_diff_by_unit_plot[1], vlines=np.percentile(shuffle_diff_by_unit, [0, 100]).tolist(), lw=1.5,
                ls='-')
    cond_diff_by_unit_plot[1].set_ylabel('Frequency')
    cond_diff_by_unit_plot[1].set_xlabel('\u0394firing rate (deviant - normal)')

    cond_diff_by_unit_plot[0].set_size_inches(2, 2)
    cond_diff_by_unit_plot[0].show()
    cond_diff_by_unit_plot[0].savefig(norm_dev_figdir / f'cond_diff_by{"_".join(pips_by_cond)}.pdf')
    print(f'1 sample ttest: {ttest_1samp(cond_diff_by_unit, 0)}, mean = {np.mean(cond_diff_by_unit)}')
    print(
        f'independent ttest from shuffle: {ttest_ind(cond_diff_by_unit, shuffle_diff_by_unit, equal_var=False, alternative="greater")}, '
        f'\n shuffle mean = {np.mean(shuffle_diff_by_unit)}')
    print(f'{(cond_diff_by_unit.values > np.percentile(shuffle_diff_by_unit, 100)).sum() / len(cond_diff_by_unit)} '
          f'units exceed 100th percentile')
    print(f'{(cond_diff_by_unit.values < np.percentile(shuffle_diff_by_unit, 0)).sum() / len(cond_diff_by_unit)} '
          f'units less than 0th percentile')


    # rare_freq_dec_ts_plot[0].savefig(aggr_figdir / f'rare_freq_dec_ts{"_".join(conds)}.pdf')

    # sim using full population
    concatenated_event_responses_hipp_only = {
        e: np.concatenate([event_responses[sessname][e].mean(axis=0) for sessname in event_responses
                           if any([animal in sessname for animal in ['DO79', 'DO81']])],)
        for e in list(event_responses.values())[0].keys()}
    pips_2_comp = ['D-1', 'D-0', 'D-2', ]
    dev_comps = {}
    for pip in 'ABCD':
        pips_2_comp = [f'{pip}-1', f'{pip}-0', f'{pip}-2', ]
        resp_vectors = [concatenated_event_responses_hipp_only[e][:,-10:].mean(axis=1) for e in pips_2_comp]
        sim_dev_to_norms = cosine_similarity(resp_vectors)
        sim_mat_plot = plot_similarity_mat(sim_dev_to_norms,pips_2_comp,'Greys',im_kwargs=dict(vmax=1,vmin=0.6,)),
        sim_mat_plot[0][0].show()
        sim_mat_plot[0][0].savefig(dev_ABBA1_figdir / f"sim_mat{'_'.join(pips_2_comp)}_grays.pdf")
        dev_comps[pip] = sim_dev_to_norms[0, 1:]
    resp_vectors = [concatenated_event_responses_hipp_only[e][:,-10:].mean(axis=1) for e in ['D-1','D-0','C-0']]
    sim_dev_to_norms = cosine_similarity(resp_vectors)

    dev_comp_plot = plt.subplots()
    for pip_i, (pip,pip_sims) in enumerate(dev_comps.items()):
        [dev_comp_plot[1].scatter(pip_i+offset, sim, label=lbl,c=c,s=50)
         for offset,sim,lbl,c in zip([-0.1,0.1],pip_sims,['ABCD(0)','ABBA(1)',],['blue','red'])]
    # format_axis(dev_comp_plot[1],vlines=list(range(len(dev_comps))),ls='--',lw=0.2)
    dev_comp_plot[1].set_ylabel('Cosine similarity')
    # dev_comp_plot[1].set_xlabel('unit')
    # dev_comp_plot[1].legend()
    dev_comp_plot[1].set_yticks([0.6,0.7])
    dev_comp_plot[1].set_yticklabels([0.6,0.7])
    dev_comp_plot[1].set_xticks(list(range(len(dev_comps))))
    dev_comp_plot[1].set_xticklabels(list(dev_comps.keys()))
    dev_comp_plot[0].set_layout_engine('constrained')
    dev_comp_plot[0].set_size_inches(2, 2)
    dev_comp_plot[0].show()
    dev_comp_plot[0].savefig(norm_dev_figdir / f"dev_comp_scatter.pdf")

