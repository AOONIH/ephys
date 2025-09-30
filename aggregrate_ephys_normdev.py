import argparse
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
from behviour_analysis_funcs import get_all_cond_filts, get_n_since_last, get_prate_block_num, get_cumsum_columns, \
    add_datetimecol, get_lick_in_patt_trials, get_earlyX_trials, get_last_pattern
from io_utils import posix_from_win
from plot_funcs import plot_2d_array_with_subplots, plot_psth, format_axis
from neural_similarity_funcs import plot_similarity_mat
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

    cond_filters = get_all_cond_filts()

    all_sess_info = session_topology.query('sess_order=="main" & date >= 240219')
    all_sess_td_df = load_aggregate_td_df(all_sess_info,home_dir,'Stage==4')
    sessions2use = all_sess_td_df.index.get_level_values('sess').unique()
    # sessnames = [Path(sess_info['sound_bin']).stem.replace('_SoundData', '') for _, sess_info in
    #              sessions2use.iterrows()]

    pkldir = ceph_dir / posix_from_win(args.pkldir)
    pkls = list(pkldir.glob('*pkl'))
    pkls2load = [pkl for pkl in pkls if Path(pkl).stem in sessions2use]
    sessions = load_aggregate_sessions(pkls2load,)


    # [sess for sess in sessions2use if not any([sess in Path(pkl).stem for pkl in pkls2load])]

    [sessions.pop(sess) for sess in list(sessions.keys()) if 'A-0' not in list(sessions[sess].sound_event_dict.keys())]
    # [sessions.pop(sess) for sess in list(sessions.keys()) if 'D-1' not in list(sessions[sess].sound_event_dict.keys())]
    [sessions.pop(sess) for sess in list(sessions.keys()) if 'A-2' in list(sessions[sess].sound_event_dict.keys())]
    [print(sess,sessions[sess].sound_event_dict.keys()) for sess in list(sessions.keys()) ]

    # update styles
    plt.style.use('figure_stylesheet.mplstyle')

    aggr_figdir = ceph_dir / 'Dammy' / 'figures' / 'normdev_aggr_plots'
    if not aggr_figdir.is_dir():
        aggr_figdir.mkdir()

    window = (-0.1, 0.25)
    event_responses = aggregate_event_responses(sessions, events=None,  # [f'{pip}-0' for pip in 'ABCD']
                                                events2exclude=['trial_start',], window=window,
                                                pred_from_psth_kwargs={'use_unit_zscore': True, 'use_iti_zscore': False,
                                                                      'baseline': 0, 'mean': None, 'mean_axis': 0})
    concatenated_event_responses = {
        e: np.concatenate([event_responses[sessname][e].mean(axis=0) for sessname in event_responses])
        for e in list(event_responses.values())[0].keys()}

    sessnames = list(event_responses.keys())
    event_features = aggregate_event_features(sessions, events=[f'{pip}-0' for pip in 'ABCD'],
                                              events2exclude=['trial_start'])

    # some psth plots
    psth_figdir = aggr_figdir.parent / 'psth_plots_aggr_sessions_normdev'
    if not psth_figdir.is_dir():
        psth_figdir.mkdir()
    full_pattern_responses_4_psth = aggregate_event_responses(sessions, events=[e for e in concatenated_event_responses.keys()
                                                                                if 'A' in e],
                                                              events2exclude=['trial_start'], window=[-0.25, 1],
                                                              pred_from_psth_kwargs={'use_unit_zscore': False,
                                                                             'use_iti_zscore': False,
                                                                             'baseline': 0.25, 'mean': None,
                                                                             'mean_axis': 0})
    concat_full_patt_resps = {
        e: np.concatenate([full_pattern_responses_4_psth[sessname][e].mean(axis=0)
                           for sessname in full_pattern_responses_4_psth])
        for e in [e for e in concatenated_event_responses.keys() if 'A' in e]}

    event_responses_4_psth = aggregate_event_responses(sessions, events=None,  # [f'{pip}-0' for pip in 'ABCD']
                                                       events2exclude=['trial_start', ], window=window,
                                                       pred_from_psth_kwargs={'use_unit_zscore': False, 'use_iti_zscore': False,
                                                                      'baseline': 0.1, 'mean': None, 'mean_axis': 0})
    concatenated_event_responses_4_psth = {
        e: np.concatenate([event_responses_4_psth[sessname][e].mean(axis=0) for sessname in event_responses_4_psth])
        for e in list(event_responses_4_psth.values())[0].keys()
    }


    psth_figdir = aggr_figdir.parent / 'psth_plots_aggr_sessions'
    for pip in [e for e in concatenated_event_responses.keys() if 'A' in e]:
        all_resps_psth = plt.subplots(2,sharex=True, gridspec_kw={'height_ratios': [1, 9],'hspace': 0.1})
        resp_mat = concat_full_patt_resps[pip]
        resp_mat_sorted = resp_mat[concat_full_patt_resps['A-0'][:,25:50].max(axis=1).argsort()[::-1]]
        cmap_norm = TwoSlopeNorm(vcenter=0, vmin=-5, vmax=5)
        plot_psth(resp_mat_sorted,pip,[-0.25,1],aspect=0.1,cmap='bwr',norm=cmap_norm,
                  plot=(all_resps_psth[0],all_resps_psth[1][1]))
        all_resps_psth[0].set_layout_engine('tight')
        format_axis(all_resps_psth[1][1],vlines=np.arange(0,1,0.25).tolist(),ylabel='Unit #',
                    xlabel=f'Time from {pip} onset (s)')
        psth_x_ser = np.linspace(-0.25, 1, resp_mat.shape[1])
        all_resps_psth[1][0].plot(psth_x_ser, resp_mat.mean(axis=0), color='k', lw=2)
        all_resps_psth[1][0].fill_between(psth_x_ser,
                                          resp_mat.mean(axis=0) - resp_mat.std(axis=0) / np.sqrt(resp_mat.shape[0]),
                                          resp_mat.mean(axis=0) + resp_mat.std(axis=0) / np.sqrt(resp_mat.shape[0]),
                                          color='k', alpha=0.1)
        format_axis(all_resps_psth[1][0],vlines=[0])
        all_resps_psth[1][0].locator_params(axis='y', nbins=2)
        all_resps_psth[0].show()

        all_resps_psth[0].savefig(psth_figdir / f'{pip}_all_resps_psth_aggr_fam_sessions.svg')


    pips_2_plot = ['base','X'][:]
    for pip in pips_2_plot:
        psth_plot = plt.subplots(2,sharex=True, gridspec_kw={'height_ratios': [1, 9],'hspace': 0.1})
        resp_mat = concatenated_event_responses_4_psth[pip]
        resp_mat_sorted = resp_mat[resp_mat.max(axis=1).argsort()[::-1]]
        low,high = np.quantile(resp_mat,0.05),np.quantile(resp_mat,0.95)
        cmap_norm = TwoSlopeNorm(vcenter=0, vmin=np.floor(low), vmax=np.ceil(high))
        plot_psth(resp_mat_sorted,pip,[-0.1,0.25],aspect=0.1,cmap='bwr',norm=cmap_norm,
                  plot=(psth_plot[0],psth_plot[1][1]))
        psth_plot[0].set_layout_engine('tight')
        format_axis(psth_plot[1][1],vlines=[0],ylabel='Unit #',
                    xlabel=f'Time from {pip} onset (s)')
        psth_x_ser = np.linspace(-0.1, 0.25, resp_mat.shape[1])
        psth_plot[1][0].plot(psth_x_ser,resp_mat.mean(axis=0),color='k',lw=2)
        psth_plot[1][0].fill_between(psth_x_ser,
                                     resp_mat.mean(axis=0) - resp_mat.std(axis=0) / np.sqrt(resp_mat.shape[0]),
                                     resp_mat.mean(axis=0) + resp_mat.std(axis=0) / np.sqrt(resp_mat.shape[0]),
                                     color='k', alpha=0.1)
        format_axis(psth_plot[1][0],vlines=[0])
        psth_plot[1][0].locator_params(axis='y', nbins=2)
        psth_plot[0].show()

        psth_plot[0].savefig(psth_figdir / f'{pip}_all_resps_psth_aggr_fam_sessions.svg')



    # do some decoding
    event_responses_by_features_by_sess = {
        e: [event_responses[sessname][e] for sessname in event_responses]
        for e in list(event_responses.values())[0].keys()}

    full_pattern_responses = aggregate_event_responses(sessions, events=[e for e in concatenated_event_responses.keys()
                                                                         if 'A' in e],
                                                       events2exclude=['trial_start'], window=[-0.25, 1],
                                                       pred_from_psth_kwargs={'use_unit_zscore': False,
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


    # # decode events
    # resps2use = event_responses
    # pips_as_ints = {pip: pip_i for pip_i, pip in enumerate([f'{pip}-{pip_i}' for pip in 'ABCD' for pip_i in range(1)])}
    # # train on A decode on B C D
    # sessname = list(resps2use.keys())[0]
    # decoders_dict = {}
    # # patt_is = [1, 2]
    # bad_dec_sess = set()
    # for sessname in tqdm(list(resps2use.keys()), total=len(resps2use), desc='decoding across sessions'):
    #     xys = [(event_responses[sessname][f'{pip}'],
    #             np.full_like(event_responses[sessname][pip][:,0,0], pips_as_ints[pip]))
    #            for pip in [f'{p}-0' for p in 'ABCD']]
    #     xs = np.vstack([xy[0][:,:,-10:].mean(axis=-1) for xy in xys])
    #     ys = np.hstack([xy[1] for xy in xys])
    #     try:
    #         decoders_dict[f'{sessname}-allvall'] = decode_responses(xs, ys,n_runs=100)
    #     except ValueError:
    #         print(f'{sessname}-allvall failed')
    #         bad_dec_sess.add(sessname)
    #         continue
    # [decoders_dict[dec_name]['data'].plot_confusion_matrix([f'{p}-{0}' for p in 'ABCD'],)
    #  for dec_name in decoders_dict.keys()]
    # # [decoders_dict[dec_name]['data'].cm_plot[0].show()
    # #  for dec_name in decoders_dict.keys()]
    #
    # [decoders_dict.pop(dec_name) for dec_name in [k for k in decoders_dict.keys() for sess in bad_dec_sess if
    #                                               k.startswith(sess)]
    #  if dec_name in decoders_dict.keys()]
    # all_cms_by_pip =[[dec['data'].cm for dec_name, dec in decoders_dict.items() if dec['data'].cm is not None]]
    # all_cms_by_pip_arr = np.squeeze(np.array(all_cms_by_pip))
    # all_cms_by_pip_plot = plt.subplots(ncols=len(all_cms_by_pip))
    # cmap_norm = TwoSlopeNorm(vmin=np.quantile(all_cms_by_pip_arr,0.1), vmax=np.quantile(all_cms_by_pip_arr,0.9),
    #                          vcenter=1/all_cms_by_pip_arr.shape[-1])
    # [plot_2d_array_with_subplots(np.mean(pip_cms,axis=0),plot=(all_cms_by_pip_plot[0],all_cms_by_pip_plot[1]),
    #                              cmap='bwr', norm=cmap_norm)
    #  for pip_i, pip_cms in enumerate(all_cms_by_pip)]
    # all_cms_by_pip_plot[1].invert_yaxis()
    # all_cms_by_pip_plot[0].set_size_inches(10, 6)
    # all_cms_by_pip_plot[0].show()
    # all_cms_by_pip_plot[0].savefig(aggr_figdir / 'decoding_cms_all_v_all.svg')
    # dec_acc_plot = plt.subplots()
    # accuracy_across_sessions = [np.diagonal(pip_cm,axis1=1, axis2=2) for pip_cm in all_cms_by_pip_arr]
    # accuracy_across_sessions_dfs = [pd.DataFrame(pip_accs,index=list([sess for sess in resps2use.keys()
    #                                                                   if sess not in bad_dec_sess]))
    #                                 for pip_accs in accuracy_across_sessions]
    #
    # # barplot for each pip
    # pip_vs_pip_acc_plot = plt.subplots()
    # pip_vs_pip_acc_plot[1].boxplot([acc for acc in np.diagonal(all_cms_by_pip_arr,axis1=1,axis2=2).T],
    #                                labels=list('ABCD'))
    # pip_vs_pip_acc_plot[0].show()
    # pip_vs_pip_acc_plot[0].savefig(aggr_figdir / 'decoding_across_pips.svg')
    #
    # accuracy_across_sessions_df = pd.concat(accuracy_across_sessions_dfs, axis=1)
    # [print(ttest_1samp(acc, 1/all_cms_by_pip_arr.shape[-1], alternative='greater')) for acc in accuracy_across_sessions]
    # print(ttest_ind(accuracy_across_sessions[0][0],accuracy_across_sessions[1][0], alternative='two-sided'))
    # print(ttest_ind(accuracy_across_sessions[0][1],accuracy_across_sessions[1][1], alternative='two-sided'))



        # all_resps_psth[0].savefig(psth_figdir / f'{pip}_all_resps_psth_abstraction_sessions.svg')

        # format axis
    # td_df stuff
    for sessname in sessions:
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


    full_pattern_responses = aggregate_event_responses(sessions, events=[e for e in concatenated_event_responses.keys()
                                                                         if 'A' in e],
                                                       events2exclude=['trial_start'], window=[-0.25, 1],
                                                       pred_from_psth_kwargs={'use_unit_zscore': True,
                                                                             'use_iti_zscore': False,
                                                                             'baseline': 0, 'mean': None,
                                                                             'mean_axis': 0})

    # decode normal from deviant
    decode_patt_base_dict = {}
    pips = ['A-0','A-1']
    for sessname in tqdm(full_pattern_responses.keys(), desc='decoding', total=len(full_pattern_responses.keys())):
        xs_list = [full_pattern_responses[sessname]['A-0'][100:150][::2], full_pattern_responses[sessname]['A-1']]
        xs = np.vstack([x.mean(axis=-1) for x in xs_list])
        ys = np.hstack([np.full(x.shape[0], ci) for ci, x in enumerate(xs_list)])
        decode_patt_base_dict[f'{sessname}_patt_v_base'] = decode_responses(xs, ys, n_runs=100)
    # plot accuracy
    patt_v_base_accuracy = np.array([decode_patt_base_dict[dec_name]['data'].accuracy
                                         for dec_name in decode_patt_base_dict.keys()])
    patt_v_base_shuff_accuracy = np.array([decode_patt_base_dict[dec_name]['shuffled'].accuracy
                                           for dec_name in decode_patt_base_dict.keys()])
    patt_v_base_accuracy_plot = plt.subplots()
    patt_v_base_accuracy_plot[1].boxplot([patt_v_base_accuracy.mean(axis=1),
                                              patt_v_base_shuff_accuracy.mean(axis=1)],labels=['data','shuffle'],
                                             showmeans=True, meanprops=dict(mfc='k'),)
    patt_v_base_accuracy_plot[1].set_ylabel('Accuracy')
    format_axis(patt_v_base_accuracy_plot[1],hlines=[0.5])
    patt_v_base_accuracy_plot[0].show()
    patt_v_base_accuracy_plot[0].savefig(aggr_figdir / 'norm_v_dev_dec_accuracy.svg')
    # ttest
    ttest = ttest_ind(patt_v_base_accuracy.mean(axis=1), patt_v_base_shuff_accuracy.mean(axis=1),
                      alternative='greater', equal_var=True)
    print(f'patt_v_base ttest: {ttest}')


    # decode by condition
    decode_by_cond_dict = {}
    pip = 'A-0'
    # window = (-0.1, 0.25)
    # event_responses = aggregate_event_reponses(sessions, events=None,  # [f'{pip}-0' for pip in 'ABCD']
    #                                            events2exclude=['trial_start','base', 'X'], window=window,
    #                                            pred_from_psth_kwargs={'use_unit_zscore': True, 'use_iti_zscore': False,
    #                                                                   'baseline': 0, 'mean': None, 'mean_axis': 0})
    resps_by_cond = {}
    conds = ['normal_exp', 'deviant_C']
    for cond in conds:
        resps_by_cond[cond] = group_ephys_resps('A-0', full_pattern_responses, event_features_all, cond_filters[cond])[0]

    for sessname in tqdm(resps_by_cond[conds[0]].keys(),desc='decoding', total=len(resps_by_cond[conds[0]].keys())):
        xs_list = [resps_by_cond[cond][sessname] for cond in conds]
        xs = np.vstack([x[:,:].mean(axis=-1) for x in xs_list])
        ys = np.hstack([np.full(x.shape[0],ci) for ci,x in enumerate(xs_list)])
        decode_by_cond_dict[f'{sessname}-{"_v_".join(conds)}'] = decode_responses(xs, ys, n_runs=100,
                                                                                  dec_kwargs={'n_jobs': 8})

    rec_dec_decoder_accuracy = np.array([decode_by_cond_dict[dec_name]['data'].accuracy
                                         for dec_name in decode_by_cond_dict.keys() if 'recent_v_distant' in dec_name])
    rec_dec_decoder_shuff_accuracy = np.array([decode_by_cond_dict[dec_name]['shuffled'].accuracy
                                         for dec_name in decode_by_cond_dict.keys() if 'recent_v_distant' in dec_name])

    rec_dec_decoder_accuracy_plot = plt.subplots()
    rec_dec_decoder_accuracy_plot[1].boxplot([rec_dec_decoder_accuracy.mean(axis=1),
                                              rec_dec_decoder_shuff_accuracy.mean(axis=1)],labels=['data','shuffle'],
                                             showmeans=True, meanprops=dict(mfc='k'),)
    # [rec_dec_decoder_accuracy_plot[1].hist(dec, alpha=0.2,fc=c,ec='k', bins=np.arange(0.45,0.85,0.05))
    #  for dec,c in zip([rec_dec_decoder_accuracy.mean(axis=1), rec_dec_decoder_shuff_accuracy.mean(axis=1)],['orange','grey'])]
    # rec_dec_decoder_accuracy_plot[1].set_xlabel('Accuracy')
    # rec_dec_decoder_accuracy_plot[1].set_ylabel('Number of sessions')
    # rec_dec_decoder_accuracy_plot[1].axvline(np.mean(rec_dec_decoder_shuff_accuracy.mean(axis=1)), color='k', ls='--')
    # rec_dec_decoder_accuracy_plot[1].axvline(np.percentile(rec_dec_decoder_accuracy.mean(axis=1),2.5), color='k', ls='--')
    rec_dec_decoder_accuracy_plot[1].set_title(f'Accuracy of recent vs distant')
    rec_dec_decoder_accuracy_plot[0].set_layout_engine('tight')
    rec_dec_decoder_accuracy_plot[0].show()
    # decoder_acc_plot[0].savefig(abstraction_figdir / f'{lbl}_decoding_accuracy_dev_ABBA1.svg')

    # ttest on fold accuracy
    ttest = ttest_ind( rec_dec_decoder_accuracy.mean(axis=1), rec_dec_decoder_shuff_accuracy.mean(axis=1),
                      alternative='greater', equal_var=True)
    print(f'Recent vs. distant ttest: {ttest}')