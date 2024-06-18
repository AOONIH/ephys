import matplotlib.pyplot as plt
import numpy as np

from ephys_analysis_funcs import *
import platform
import argparse
import yaml
from scipy.stats import pearsonr, ttest_1samp, ttest_ind
from scipy.signal import savgol_filter
from sklearn.metrics.pairwise import cosine_similarity

from neural_similarity_funcs import *
from postprocessing_utils import get_sorting_dirs
from datetime import datetime
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('sess_date')
    parser.add_argument('--sorter_dirname',default='from_concat',required=False)
    parser.add_argument('--sess_top_filts', default='')

    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])

    plt.ioff()

    # try: gen_metadata(ceph_dir/posix_from_win(r'X:\Dammy\ephys\session_topology.csv'),ceph_dir,ceph_dir/'Dammy'/'harpbins')
    sess_topology_path = ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology.csv')
    # try: gen_metadata(sess_topology_path,ceph_dir,
    #                   col_name='beh_bin',harp_bin_dir='')
    # except OSError: pass
    # gen_metadata(sess_topology_path, ceph_dir,
    #              col_name='beh_bin', harp_bin_dir='')
    session_topology = pd.read_csv(sess_topology_path)
    # win_rec_dir = r'X:\Dammy\ephys\DO79_2024-01-17_15-20-19_001\Record Node 101\experiment1\recording1'
    name,date = args.sess_date.split('_')
    date = int(date)
    all_sess_info = session_topology.query('name==@name & date==@date').reset_index(drop=True)
    if args.sess_top_filts:
        all_sess_info = all_sess_info.query(args.sess_top_filts)

    dir1_name, dir2_name = 'sorting_no_si_drift', 'kilosort2_5_ks_drift'
    ephys_dir = ceph_dir / 'Dammy' / 'ephys'
    date_str = datetime.strptime(str(date), '%y%m%d').strftime('%Y-%m-%d')
    print(f'{name}_{date_str}')

    if all_sess_info.shape[0] == 1:
        sorter_dirname = 'sorter_output'
    else:
        sorter_dirname = args.sorter_dirname
    sorter_dirname = 'si_output'
    sort_dirs = get_sorting_dirs(ephys_dir, f'{name}_{date_str}',dir1_name, dir2_name, sorter_dirname)
    sort_dirs = [e for ei,e in enumerate(sort_dirs) if ei in all_sess_info.index]

    ephys_figdir = ceph_dir/'Dammy'/'figures'/'run_240613_single_sorted_sessions'
    if not ephys_figdir.is_dir():
        ephys_figdir.mkdir()

    sessions = {}
    psth_window = [-2, 3]
    main_sess = all_sess_info.query('sess_order=="main"')
    main_sess_td_name = Path(main_sess['sound_bin'].iloc[0].replace('_SoundData', '_TrialData')).with_suffix('.csv').name
    # get main sess pattern
    home_dir = Path(config[f'home_dir_{sys_os}'])
    main_patterns = get_main_sess_patterns(name, date, main_sess_td_name, home_dir)
    normal_patterns = [pattern for pattern in main_patterns if np.all(np.diff(pattern)>0)]
    non_normal_patterns = [pattern for pattern in main_patterns if not np.all(np.diff(pattern)>0)]
    if non_normal_patterns:
        main_patterns = list(sum(zip(normal_patterns,non_normal_patterns),()))
    else:
        main_patterns = normal_patterns
    print(f'{main_patterns=}')
    n_patts_per_rule = int(len(main_patterns)/ len(normal_patterns))

    main_pattern = main_patterns[0]
    # sort_dirs =

    plot_psth_decode = True
    decode_over_time = False

    for (_,sess_info),spike_dir in zip(all_sess_info.iterrows(), sort_dirs):
        if sess_info['sess_order'] != 'main':
            continue

        recording_dir = next((ceph_dir/posix_from_win(sess_info['ephys_dir'],config['ceph_dir_linux'])).rglob('continuous')).parent
        # print((ceph_dir/posix_from_win(sess_info['ephys_dir'])))
        print(f'analysing {recording_dir.name}')

        spike_cluster_path = r'spike_clusters.npy'
        spike_times_path = r'spike_times.npy'

        with open(recording_dir / 'metadata.json', 'r') as jsonfile:
            recording_meta = json.load(jsonfile)
        start_time = recording_meta['trigger_time']
        sessname = Path(sess_info['sound_bin']).stem
        sessname = sessname.replace('_SoundData','')

        sessions[sessname] = Session(sessname, ceph_dir)

        # sess_td_path = next(home_dir.rglob(f'*{sessname}_TrialData.csv'))
        # sess_td_path = sess_info['trialdata_path']

        sound_bin_path = Path(sess_info['sound_bin'])
        beh_bin_path = Path(sess_info['beh_bin'])

        if sess_info['sess_order'] == 'main':  # load trial_data

            sessions[sessname].load_trial_data(get_main_sess_td_df(name,date,main_sess_td_name,home_dir)[1])
            # normal = sessions[sessname].td_df[sessions[sessname].td_df['Tone_Position'] == 0]['PatternID'].iloc[0]
            normal = main_pattern
            # normal = [int(pip) for pip in normal.split(';')]
            if -1 in sessions[sessname].td_df['Pattern_Type'].unique():
                new_normal = sessions[sessname].td_df[sessions[sessname].td_df['Pattern_Type'] == -1]['PatternID'].iloc[0]
                # new_normal = [int(pip) for pip in new_normal.split(';')]
            else:
                new_normal = None

        else:
            # normal = [int(pip) for pip in main_pattern.split(';')]
            normal= main_pattern
            new_normal = None
        if not sessions[sessname].sound_event_dict:
            sessions[sessname].init_spike_obj(spike_times_path, spike_cluster_path, start_time, parent_dir=spike_dir)
            _parts = sessname.split('_')
            '_'.join([_parts[0],'SoundData',_parts[1]])
            labels = ['A-0', 'B-0', 'C-0', 'D-0', 'X', 'base','newA']

            abs_writes_path = str(sound_bin_path.with_stem(f'{sound_bin_path.stem}_write_indices'))
            sessions[sessname].init_sound_event_dict(ceph_dir/posix_from_win(abs_writes_path).with_suffix('.csv'),
                                                     patterns=main_patterns)
            sessions[sessname].get_event_free_zscore()

            sessions[sessname].get_sound_psth(psth_window=psth_window,zscore_flag=True,baseline_dur=0,redo_psth=True,
                                              use_iti_zscore=False)

            n_shuffles = 1000
            by_pip_predictors = {}
            for pi, pip2use in enumerate('ABCD'):
                by_pip_predictors[pip2use] = {event_lbl: get_predictor_from_psth(sessions[sessname], event_lbl,
                                                                                 psth_window, [0,0.25],
                                                                                 mean=None)
                                              for event_lbl in sessions[sessname].sound_event_dict
                                              if any(char in event_lbl for char in pip2use)}
            # get self sims for all pips
            assert check_unique_across_dim([list(e.values()) for e in by_pip_predictors.values()]), 'overlapping pips'
            pip_desc, pip_lbls, pip_names = get_pip_info(sessions[sessname].sound_event_dict, normal_patterns,
                                                         n_patts_per_rule)
            n_shuffles = 1000
            # assert not check_unique_across_dim([np.random.permutation(len(by_pip_predictors['A']['A-0']))
            #                                     for _ in range(1000)]), 'Permute shuffling across unit or time axis'
            self_sims = [[compute_self_similarity(responses[shuffled], cv_folds=5) for shuffled in
                          [np.random.permutation(len(responses)) for _ in range(1000)]] for responses in
                         sum([list(e.values()) for e in by_pip_predictors.values()], [])]
            self_sim_means = np.squeeze([np.mean(e, axis=1) for e in self_sims])
            self_sim_names, self_sims_idx = get_reordered_idx(pip_desc, pip_lbls, ['group'])

            self_sims_plot = plt.subplots(figsize=(18, 8))
            all_pip_labels = [e.split(' ')[-1] for e in self_sim_names]
            self_sims_plot[1].boxplot(self_sim_means[self_sims_idx].tolist(), labels=self_sim_names)
            self_sims_plot[1].set_title('Within pip self similarity', fontsize=20)
            self_sims_plot[1].set_ylabel('Self similarity', fontsize=18)
            self_sims_plot[0].set_layout_engine('tight')
            self_sims_plot[0].show()
            self_sims_plot[0].savefig(ephys_figdir / f'pip_self_sim_{sessname}.svg')

            event_psth_dict = {e: by_pip_predictors[e] for e in sum(list(by_pip_predictors.values()), [])}
            compared_pips_plot = plt.subplots(4,figsize=(6, 18))
            for pi,pip in enumerate(['A', 'B', 'C', 'D']):
                compared_pips = compare_pip_sims_2way([event_psth_dict[f'{pip}-0'], event_psth_dict[f'{pip}-1']])

                mean_comped_sims = [np.squeeze(pip_sims)[:, 0, 1] for pip_sims in np.array_split(compared_pips[0], 2)]
                mean_comped_sims.append(np.squeeze(compared_pips[1][:, 0, 1]))
                compared_pips_plot[1][pi].boxplot(mean_comped_sims,
                                              labels=[f'{pip}-0 self', f'{pip}-1 self', f'{pip}-0 vs {pip}-1'], )
                compared_pips_plot[1][pi].set_ylim([0, 1])
                compared_pips_plot[1][pi].set_ylabel('cosine similarity')
                compared_pips_plot[0].show()
                compared_pips_plot[0].savefig(ephys_figdir/f'{pip}_compared_{sessname}.svg')


            # save_session psth
            if plot_psth_decode:
                sess_psth_dir = ceph_dir / 'Dammy' / 'ephys' / 'session_data' / sessname
                if not sess_psth_dir.is_dir():
                    sess_psth_dir.mkdir(parents=True)
                x_ser = np.linspace(psth_window[0], psth_window[1],
                                    sessions[sessname].sound_event_dict['A-0'].psth[0].shape[-1])
                # np.save(sess_psth_dir / f'psth_times.npy', x_ser)
                # [np.save(sess_psth_dir / f'{key}_psth.npy', get_predictor_from_psth(sessions[sessname], key, psth_window,
                #                                                                     psth_window, mean=None,
                #                                                                     use_unit_zscore=False,
                #                                                                     use_iti_zscore=True))
                #  for key in tqdm(sessions[sessname].sound_event_dict,total=len(sessions[sessname].sound_event_dict),
                #                  desc='saving psth')]

                # sessions[sessname].get_sound_psth(psth_window=psth_window,use_iti_zscore=True, redo_psth_plot=True,)

                [sessions[sessname].sound_event_dict['A-0'].psth_plot[1].axvline(t, c='white', ls='--') for t in
                 np.arange(0, 1, 0.25) if sess_info['sess_order'] == 'main']

                psth_ts_plot = plt.subplots()
                psth_ts_plot[1].plot(sessions[sessname].sound_event_dict['A-0'].psth[1].columns.to_series().dt.total_seconds(),
                                     sessions[sessname].sound_event_dict['A-0'].psth[1].mean(axis=0),
                                     c='k',lw=3)
                # psth_ts_plot[1].set_ylim(-0.17,0.04)
                psth_ts_plot[1].set(frame_on=False)
                psth_ts_plot[1].set_xticklabels([])
                psth_ts_plot[1].set_yticklabels(psth_ts_plot[1].get_yticklabels())
                psth_ts_plot[0].set_size_inches(6.4,1)
                psth_ts_plot[1].axvline(0,c='k',ls='--')
                psth_ts_plot[0].savefig(ephys_figdir/f'A_psth_ts_{sessname}.svg')

                plot_2d_array_with_subplots(sessions[sessname].sound_event_dict['D-0'].psth[1].loc[sessions[sessname].sound_event_dict['A-0'].psth[2]])
                sessions[sessname].save_psth(figdir=ephys_figdir)

                # plot all on 1
                # X and A
                if 'X' in list(sessions[sessname].sound_event_dict.keys()):
                    psth_XA_plot = plt.subplots(ncols=2,figsize=(4.5,3.5),sharey='all')
                    for ei, e in enumerate(['X','A-0']):
                        psth_mat = get_predictor_from_psth(sessions[sessname], e, psth_window,[-0.5,1],mean=np.mean,mean_axis=0)
                        plot_psth(psth_mat,f'Time from {e} onset', [-0.5,1],plot_cbar=(True if ei==1 else False),
                                  plot=(psth_XA_plot[0],psth_XA_plot[1][ei]))
                        if e =='A-0':
                            [psth_XA_plot[1][ei].axvline(t, c='white', ls='--') for t in np.arange(0, 1, 0.25)]
                        if ei==0:
                            psth_XA_plot[1][ei].set_ylabel('units',fontsize=18)
                        else:
                            psth_XA_plot[1][ei].set_ylabel('')
                        psth_XA_plot[1][ei].set_xticks([0,1])
                        psth_XA_plot[1][ei].set_yticks([])
                        psth_XA_plot[1][ei].set_xlabel(f'',fontsize=18)
                        psth_XA_plot[1][ei].set_title(f'{e}',fontsize=18)
                        psth_XA_plot[1][ei].tick_params(axis='both', which='major', labelsize=18)
                        psth_XA_plot[1][ei].locator_params(axis='both', nbins=3)
                    psth_XA_plot[0].tight_layout(pad=0)
                    psth_XA_plot[0].show()

                if sess_info['sess_order'] != 'main':
                    psth_ABCD_plot = plt.subplots(ncols=4,figsize=(9,3.5),sharey='all')
                    for ei, e in enumerate(['A-0','B-0','C-0','D-0']):
                        psth_mat = get_predictor_from_psth(sessions[sessname], e, psth_window,[-0.5,1],mean=np.mean,mean_axis=0)
                        plot_psth(psth_mat,f'Time from {e} onset', [-0.5,1],vmin=-1,vmax=4.5,
                                  plot_cbar=(True if ei==len('ABCD')-1 else False),
                                  plot=(psth_ABCD_plot[0],psth_ABCD_plot[1][ei]))
                        psth_ABCD_plot[1][ei].axvline(0, c='white', ls='--')
                        if ei == 0:
                            psth_ABCD_plot[1][ei].set_ylabel('units',fontsize=18)
                        else:
                            psth_ABCD_plot[1][ei].set_ylabel('')
                        psth_ABCD_plot[1][ei].set_xticks([0,1])
                        psth_ABCD_plot[1][ei].set_yticks([])
                        psth_ABCD_plot[1][ei].set_xlabel(f'',fontsize=18)
                        psth_ABCD_plot[1][ei].set_title(f'{e}',fontsize=18)
                        psth_ABCD_plot[1][ei].tick_params(axis='both', which='major', labelsize=18)
                        psth_ABCD_plot[1][ei].locator_params(axis='both', nbins=3)
                    psth_ABCD_plot[0].tight_layout(pad=0)
                    psth_ABCD_plot[0].show()

                print('plotted XA')
                window = [0, 0.25]
                if 'X' in list(sessions[sessname].sound_event_dict.keys()):
                    _predictor_list = [get_predictor_from_psth(sessions[sessname], key, psth_window, window,mean=np.mean)
                                       for key in ['X', 'base']]
                    _feature_list = [np.full(mat.shape[0], i) for i, mat in enumerate(_predictor_list)]

                    sessions[sessname].init_decoder('X_to_base', np.vstack(_predictor_list), np.hstack(_feature_list))
                    sessions[sessname].run_decoder('X_to_base', ['data','shuffle'], dec_kwargs={'cv_folds':0},
                                                   plot_flag=True)
                    sessions[sessname].decoders['X_to_base'].accuracy_plot[0].show()
                    sessions[sessname].decoders['X_to_base'].accuracy_plot[0].savefig(ephys_figdir/f'X_to_base_{sessname}.svg')
                # decoder to base
                dec_kwargs = {'cv_folds': 10}
                for pip in ['A-0','B-0','C-0','D-0']:
                    _predictor_list = [get_predictor_from_psth(sessions[sessname], key, psth_window, window,mean=np.mean)
                                       for key in [pip, 'base']]
                    # _predictor_list[-1] = subset_base_preds
                    _feature_list = [np.full(mat.shape[0], i) for i, mat in enumerate(_predictor_list)]

                    sessions[sessname].init_decoder(f'{pip}_to_base', np.vstack(_predictor_list), np.hstack(_feature_list))
                    sessions[sessname].run_decoder(f'{pip}_to_base', ['data','shuffle'], dec_kwargs={'cv_folds':0})
                    # sessions[sessname].decoders[f'{pip}_to_base'].accuracy_plot[0].set_constrained_layout('constrained')
                    # sessions[sessname].decoders[f'{pip}_to_base'].accuracy_plot[0].savefig(ceph_dir/'Dammy'/'figures'/'ephys'/f'{pip}_to_base_accr.svg',)

                tone2base_all_plot = plt.subplots()
                # for pi, pip in enumerate(['A-0','B-0','C-0','D-0']):
                #     metric2plot = sessions[sessname].decoders[f'{pip}_to_base'].fold_accuracy
                #     plot_decoder_accuracy(metric2plot, pip, fig=tone2base_all_plot[0], ax=tone2base_all_plot[1],
                #                           start_loc=pi, n_features=2)
                # metric2plot = sessions[sessname].decoders[f'{pip}_to_base'].fold_accuracy
                # plot_decoder_accuracy(metric2plot, pip, fig=tone2base_all_plot[0], ax=tone2base_all_plot[1],
                #                       start_loc=pi, n_features=2)
                # plot shuffle
                sessions[sessname].init_decoder(f'A_to_base_shuffle', np.vstack([_predictor_list[0],_predictor_list[-1]]),
                                                np.hstack([_feature_list[0],_feature_list[-1]]))
                sessions[sessname].run_decoder(f'A_to_base_shuffle', ['data', 'shuffle'], dec_kwargs={'cv_folds': 0,'shuffle':True})
                plot_decoder_accuracy(sessions[sessname].decoders[f'A_to_base_shuffle'].fold_accuracy, 'A_shuffle',
                                      fig=tone2base_all_plot[0], ax=tone2base_all_plot[1], plt_kwargs={'c':'k'},
                                      start_loc=4, n_features=2)

                # tone2base_all_plot[1].set_xticks(np.arange(len('ABCD')+1))
                # tone2base_all_plot[1].set_xticklabels(list('ABCD')+['A_shuffle'])
                tone2base_all_plot[1].get_legend().remove()
                tone2base_all_plot[0].show()
                tone2base_all_plot[0].set_constrained_layout('constrained')
                tone2base_all_plot[0].savefig(ephys_figdir/f'ABCD_to_base_perf_{sessname}.svg')

                window = [0, 0.25]
                _predictor_list = [get_predictor_from_psth(sessions[sessname], key, psth_window, window, mean=None)
                                   for key in ['A-0','B-0','C-0','D-0','base']]
                _feature_list = [np.full(mat.shape[0], i) for i, mat in enumerate(_predictor_list)]

                # compute cosine similarity
                pip_sim_over_trials = [cosine_similarity([trial_resp[:,-1] for trial_resp in pip_responses])
                                       for pip_responses in _predictor_list]
                pip_sim_over_trials_plot = plt.subplots(len(_predictor_list),figsize=(8,32))
                [plot_similarity_mat(sim, ''*sim.shape[0], plot=(pip_sim_over_trials_plot[0],ax), cmap='Reds')
                 for sim,ax in zip(pip_sim_over_trials,pip_sim_over_trials_plot[1])]
                pip_sim_over_trials_plot[0].set_layout_engine('compressed',w_pad=0.5)
                pip_sim_over_trials_plot[0].savefig(ephys_figdir/f'pip_sim_over_trials_{sessname}.svg')
                mean_pip_sim = [np.mean(sim[~np.eye(sim.shape[0],dtype=bool)]) for sim in pip_sim_over_trials]
                pip_sim_over_pips = cosine_similarity([pred[:,:,-1].mean(axis=0) for pred in _predictor_list])
                # pip_sim_over_pips = cosine_similarity([pred.mean(axis=0).mean(axis=-1) for pred in _predictor_list])
                pip_sim_pip_plot = plot_similarity_mat(pip_sim_over_pips, np.arange(len(_predictor_list)),cmap='Reds')
                pip_sim_pip_plot[0].savefig(ephys_figdir/f'pip_sim_over_pips_{sessname}.svg')

                # pearson_corr_all = np.zeros((len(_predictor_list),len(_predictor_list)))
                # for i,ii in enumerate(_predictor_list):
                #     midpoint = int(ii.shape[0]/2)
                #     for j, jj in enumerate(_predictor_list):
                #         pearson_corr_all[i,j] = pearsonr(np.array_split(ii,2)[0].mean(axis=0),
                #                                          np.array_split(jj,2)[1].mean(axis=0))[0]
                #
                # pearson_plot = plot_2d_array_with_subplots(pearson_corr_all,cbar_height=20)
                # pearson_plot[1].invert_yaxis()
                # pearson_plot[1].set_xticklabels(['','A-0','B-0','C-0','D-0','base'])
                # # pearson_plot[1].set_xticklabels(['','A-0','B-0','C-0','D-0',])
                # pearson_plot[1].set_xlabel('second half')
                # pearson_plot[1].set_yticklabels(['','A-0','B-0','C-0','D-0','base'])
                # # pearson_plot[1].set_yticklabels(['','A-0','B-0','C-0','D-0'])
                # pearson_plot[1].set_ylabel('first half')
                # pearson_plot[2].ax.set_ylabel("Pearson's correlation",rotation=270,labelpad=12)
                # pearson_plot[0].show()
                # pearson_plot[0].savefig(ephys_figdir/f'pearson_no_base_corr_matrix_{sessname}.svg',)

                # run all decoder with base for tseries
                _predictor_list = [get_predictor_from_psth(sessions[sessname], key, psth_window, window, mean=np.mean)
                                   for key in ['A-0', 'B-0', 'C-0', 'D-0', 'base']]
                _feature_list = [np.full(mat.shape[0], i) for i, mat in enumerate(_predictor_list)]

                all_dec_lbls = ['A-0', 'B-0', 'C-0', 'D-0', 'base']
                all_dec_name = 'all_vs_all'

                sessions[sessname].init_decoder(all_dec_name, np.vstack(_predictor_list), np.hstack(_feature_list),
                                                model_name='logistic')
                dec_kwargs = {'cv_folds': 10}
                sessions[sessname].run_decoder(all_dec_name, ['data', 'shuffle'], dec_kwargs=dec_kwargs)

                sessions[sessname].decoders[all_dec_name].plot_confusion_matrix(all_dec_lbls, include_values=False,cmap='copper')
                sessions[sessname].decoders[all_dec_name].cm_plot[0].set_size_inches(4,3.5)
                sessions[sessname].decoders[all_dec_name].cm_plot[1].tick_params(axis='both', which='major', labelsize=16)
                # sessions[sessname].decoders['all_vs_all'].cm_plot[0].set_constrained_layout('constrained')
                sessions[sessname].decoders[all_dec_name].cm_plot[0].savefig(ephys_figdir/
                                                                             f'ABCD_base_cm_{sessname}.svg')

                sessions[sessname].decoders[all_dec_name].plot_decoder_accuracy(all_dec_lbls, )
                sessions[sessname].decoders[all_dec_name].accuracy_plot[0].savefig(ephys_figdir/
                                                                                   f'all_vs_all_accuracy_{sessname}.svg')

        home_dir = Path(config[f'home_dir_{sys_os}'])
        if sess_info['sess_order'] == 'main':
            if 3 in sessions[sessname].td_df['Stage'].values:
                # sessions[sessname].load_trial_data(f'{sess_td_path}.csv',home_dir,
                #                                    rf'H:\data\Dammy\{sessname.split("_")[0]}\TrialData')
                idx_bool = sessions[sessname].td_df['local_rate'] <= 0.2
                idx_bool2 = sessions[sessname].td_df['local_rate'] >= 0.8
                recent_pattern_trials = sessions[sessname].td_df[idx_bool].index.to_numpy()
                distant_pattern_trials = sessions[sessname].td_df[idx_bool2].index.to_numpy()

                cumsum_plot = plt.subplots()
                cumsum_plot[1].plot(np.cumsum(idx_bool),label='freq')
                cumsum_plot[1].plot(np.cumsum(idx_bool2),label='rare')
                cumsum_plot[1].legend()
                cumsum_plot[1].set_xlabel('trial number')
                cumsum_plot[1].set_title('distribution of rare vs freq trials over session')
                cumsum_plot[0].set_constrained_layout('constrained')
                cumsum_plot[0].savefig(ephys_figdir/f'02_08_local_rate_cumsum_plot_{sessname}.svg')

                rec_dist_decoder_plot = plt.subplots()
                # dec_events = ['A-0','B-0','C-0','D-0','A_shuffle','A_halves']
                dec_events = ['A-0','A-0_shuffle']
                for pi,pip in enumerate(dec_events):
                    pip_id = pip.split('_')[0]

                    pip_predictor = get_predictor_from_psth(sessions[sessname], pip_id, psth_window, [0, 1],mean=np.mean)
                    recent_idx_bool = np.isin(sessions[sessname].sound_event_dict[pip_id].trial_nums-1,recent_pattern_trials)
                    distant_idx_bool = np.isin(sessions[sessname].sound_event_dict[pip_id].trial_nums-1,distant_pattern_trials)
                    if 'badpred' in pip:
                        rec_dist_predictors = pip_predictor[recent_idx_bool[:len(pip_predictor)]], pip_predictor[recent_idx_bool[:len(pip_predictor)]]
                    elif 'halves' in pip:
                        rec_dist_predictors = np.array_split(pip_predictor,2)
                    else:
                        rec_dist_predictors = pip_predictor[recent_idx_bool[:len(pip_predictor)]], pip_predictor[distant_idx_bool[:len(pip_predictor)]]
                    print('len 0.1/0.9 predictors',len(rec_dist_predictors[0]),len(rec_dist_predictors[1]))
                    rec_dist_features = [np.full(e.shape[0], ei) for ei, e in enumerate(rec_dist_predictors)]

                    plt_kwargs = {}
                    if 'shuffle' in pip:
                        # print('shuffle')
                        dec_kwargs['shuffle'] = True
                        plt_kwargs['c'] = 'k'

                    sessions[sessname].init_decoder(f'rec_dist_{pip}', np.vstack(rec_dist_predictors), np.hstack(rec_dist_features))
                    sessions[sessname].run_decoder(f'rec_dist_{pip}', ['data', 'shuffle'], dec_kwargs=dec_kwargs)
                    plot_decoder_accuracy([sessions[sessname].decoders[f'rec_dist_{pip}'].accuracy], [pip],
                                          ax=rec_dist_decoder_plot[1], start_loc=pi,plt_kwargs=plt_kwargs)

                # ttest
                ctrl_idx = 1
                for pi, pip in enumerate(dec_events[:4]):
                    ttest_res = ttest_ind(np.array(sessions[sessname].decoders[f'rec_dist_{pip}'].accuracy).flatten(),
                                          np.array(sessions[sessname].decoders[f'rec_dist_{dec_events[ctrl_idx]}'].accuracy).flatten(),
                                          equal_var=False)
                    p_val = ttest_res[1]
                    print(p_val)
                    rec_dist_decoder_plot[1].text(pi+0.5, 1+0.05, f'{p_val:.1e}', ha='center')
                    rec_dist_decoder_plot[1].plot([pi, ctrl_idx], [1+pi*0.015, 1+pi*0.015], ls='-', c='darkgrey')
                    if p_val <= 0.01:
                        rec_dist_decoder_plot[1].scatter(pi+0.5, 1+0.075, marker='x', s=20, c='k')

                rec_dist_decoder_plot[1].set_xticks(np.arange(len(dec_events)))
                rec_dist_decoder_plot[1].set_xticklabels(dec_events)
                rec_dist_decoder_plot[1].get_legend().remove()
                rec_dist_decoder_plot[0].set_constrained_layout('constrained')

                rec_dist_decoder_plot[0].savefig(ephys_figdir/f'02_vs_08_local_bin5_acc_{sessname}.svg')

                # do as boxplot
                rare_freq_boxplot = plt.subplots()
                perfs = [np.array(sessions[sessname].decoders[f'rec_dist_{pip}'].accuracy) for pip in dec_events[:-1]]
                # [rare_freq_boxplot[1].boxplot(perf,pi) for pi, perf in enumerate(perfs)]
                rare_freq_boxplot[1].boxplot(perfs,bootstrap=100000,labels=dec_events[:-2]+['shuffle'],)
                rare_freq_boxplot[1].set_ylim(0.25,0.8)
                rare_freq_boxplot[1].set_yticks([0,.25,0.5,0.75])
                rare_freq_boxplot[1].axhline(0.5,c='k',ls='--')
                rare_freq_boxplot[1].set_yticklabels([0,.25,0.5,0.75])
                # rare_freq_boxplot[1].set_xticks(np.arange(len(dec_events[:-1]))+1)
                # rare_freq_boxplot[1].set_xticklabels()
                rare_freq_boxplot[1].set_xlabel('event', fontsize=14)
                rare_freq_boxplot[1].set_ylabel('accuracy', fontsize=14)
                rare_freq_boxplot[1].set_title('Event rate decoding from population response to event',fontsize=14)
                rare_freq_boxplot[1].tick_params(axis='both', which='major', labelsize=14)
                # rare_freq_boxplot[0].set_constrained_layout('constrained')
                rare_freq_boxplot[0].set_size_inches(3.5,2.4)
                rare_freq_boxplot[0].savefig(ephys_figdir/f'rare_freq_decoding_boxplot_patt_window_{sessname}.svg')
                rare_freq_boxplot[0].show()

                # psth rare vs freq
                recent_idx_bool = np.isin(sessions[sessname].sound_event_dict['A-0'].trial_nums - 1, recent_pattern_trials)
                distant_idx_bool = np.isin(sessions[sessname].sound_event_dict['A-0'].trial_nums - 1, distant_pattern_trials)
                A_psth = get_predictor_from_psth(sessions[sessname], 'A-0', psth_window, [-0, .25], mean=None)
                x_ser = np.linspace(-2, 3, A_psth.shape[-1])
                rare_A = A_psth[distant_idx_bool].mean(axis=0)
                freq_A = A_psth[recent_idx_bool].mean(axis=0)
                rare_freq_psth = plot_psth_ts((rare_A-freq_A),x_ser,'Time from A',
                                              'mean difference in firing rate (rare-frequent)',c='k')
                plot_ts_var(x_ser,(rare_A-freq_A),'k',rare_freq_psth[1])

                rare_freq_psth[0].savefig(ephys_figdir/f'rare_vs_freq_psth_ts_{sessname}.svg')
                np.save(ceph_dir/'Dammy'/'ephys_pkls'/f'rare_vs_freq_array_{sessname}.npy',rare_A-freq_A)

                # bootstrap data
                n_resamples = 9999
                # for n in n_resamples:
                #     pass
                rare_freq_diff_plot = plot_psth((rare_A-freq_A),'Time from A',[-2,3],cmap='bwr',
                                                cbar_label='zscored firing rate( rare - frequent)')
                rare_freq_diff_plot[0].savefig(ephys_figdir/f'rare_vs_freq_psth_{sessname}.svg')

                rare_freq_schem_plot = plt.subplots()
                switches = sessions[sessname].td_df['PatternPresentation_Rate'][sessions[sessname].td_df['PatternPresentation_Rate'].diff() != 0]
                for si,switch in enumerate(switches):
                    if si==len(switches)-1:
                        end = sessions[sessname].td_df.shape[0]
                    else:
                        end = switches.iloc[[si+1]].index[0]
                    rare_freq_schem_plot[1].axvspan(switches.iloc[[si]].index[0],end,fc=f'C{1 if switch==0.9 else 0}',
                                                    alpha=0.3)
                rare_freq_schem_plot[1].plot((1-sessions[sessname].td_df['local_rate']),c='k',)
                rare_freq_schem_plot[1].tick_params(axis='both', which='major', labelsize=14)
                rare_freq_schem_plot[0].show()
                rare_freq_schem_plot[0].savefig(ephys_figdir/f'rate_switches_{sessname}.svg')

                # trial by trial pearsonr
                # pearson_corr_all = np.zeros((len(A_psth), len(A_psth)))
                pearsonr_list = [np.array([pearsonr(aa.mean(axis=1),a.mean(axis=1))[0] for a in A_psth]) for aa in A_psth]
                pearson_corr_all = np.vstack(pearsonr_list)

                pearson_corr_sorted = np.vstack([pearson_corr_all[distant_idx_bool],pearson_corr_all[~distant_idx_bool]])
                pearson_plot = plot_2d_array_with_subplots(pearson_corr_all, cbar_height=20,cmap='cividis',)
                r2f_switch = ((sessions[sessname].td_df['Tone_Position']==0).cumsum()
                                  [sessions[sessname].td_df.PatternPresentation_Rate.diff()<0])
                f2r_switch = ((sessions[sessname].td_df['Tone_Position']==0).cumsum()
                                  [sessions[sessname].td_df.PatternPresentation_Rate.diff()>0])
                # pearson_plot[1].axvline(recent_idx_bool.sum(), c='k', ls='--')
                # pearson_plot[1].axhline(recent_idx_bool.sum(), c='k', ls='--')
                [pearson_plot[1].axvline(i,c='w',ls='--') for i in r2f_switch.values if i not in [0,1]]
                [pearson_plot[1].axhline(i,c='lightcoral',ls='--') for i in f2r_switch.values if i not in [0,1]]

                pearson_plot[1].invert_yaxis()
                # pearson_plot[1].set_xticklabels(['','A-0','B-0','C-0','D-0',])
                # pearson_plot[1].set_yticklabels(['','A-0','B-0','C-0','D-0'])
                pearson_plot[2].ax.set_ylabel("Pearson's correlation", rotation=270, labelpad=12)
                pearson_plot[1].tick_params(axis='both', which='major', labelsize=18)
                pearson_plot[2].ax.tick_params(axis='y', which='major', labelsize=14)
                pearson_plot[0].set_size_inches(3.5,3)
                pearson_plot[0].show()
                pearson_plot[0].savefig(ephys_figdir / f'pearson_tofirst_A_{sessname}.svg', )

        if sess_info['sess_order'] == 'main' and 4 in sessions[sessname].td_df['Stage'].values:
            new_window = [-1, 2]
            pip_predictor = get_predictor_from_psth(sessions[sessname], 'A-0', psth_window, new_window, mean=None)

            # new_norm_predictors =
            x_ser = np.linspace(new_window[0],new_window[1],pip_predictor.shape[-1])
            norm_trial_nums = sessions[sessname].td_df.query('(Pattern_Type == 0) & Tone_Position == 0 & '
                                                             'local_rate == 0.0 & Session_Block == 3 & N_TonesPlayed == 4').index.to_numpy()
            dev_trial_nums = sessions[sessname].td_df.query('(Pattern_Type == 1) & Tone_Position == 0').index.to_numpy()
            newnorm_trial_nums = sessions[sessname].td_df.query('(Pattern_Type == -1) & Tone_Position == 0').index.to_numpy()
            predictors_dict = {}
            features_dict = {}
            pred_names = ['normal','deviant','new_norm']
            colors = ['saddlebrown','chocolate','darkslategreen']
            psth_ts_plot=plt.subplots()
            for di, (name, trial_nums, color) in enumerate(zip(pred_names[:-1],[norm_trial_nums,dev_trial_nums,newnorm_trial_nums],colors)):
                idx_bool = np.isin(sessions[sessname].sound_event_dict['A-0'].trial_nums - 1,
                                   trial_nums)
                predictors_dict[name] = pip_predictor[idx_bool]
                features_dict[name] = np.full(predictors_dict[name].shape[0],di)
                if di<2:
                    smoothed_data = savgol_filter(predictors_dict[name].mean(axis=0),10,2,axis=1)
                    psth_ts_plot = plot_psth_ts(smoothed_data,x_ser,label=name,plot=psth_ts_plot,c=color)
                    plot_ts_var(x_ser,smoothed_data,color,psth_ts_plot[1])
                    psth_plot = plot_psth(smoothed_data,'Time since pattern onset (s)',window)
                    psth_ts_plot[1].legend()
                    [psth_ts_plot[1].axvspan(t,t+0.15,fc='k',alpha=0.1) for t in np.arange(0,1,0.25)]
                # psth_plot[0].show()
            psth_ts_plot[0].show()
            psth_ts_plot[1].tick_params(axis='both', which='major', labelsize=16)
            psth_ts_plot[1].locator_params(axis='both', nbins=4)
            psth_ts_plot[0].set_size_inches(3.5,3)
            psth_ts_plot[0].savefig(ephys_figdir/f'norm_dev_psth_ts_{sessname}.svg')

            new_norm_pred = get_predictor_from_psth(sessions[sessname], 'A2', psth_window, new_window, mean=None)
            predictors_dict['new_norm'] = new_norm_pred
            features_dict['new_norm'] = np.full_like(new_norm_pred.shape[0],2)

            for dev_type in ['deviant','new_norm']:
                norm_dec_diff_mat = plot_psth(predictors_dict[dev_type].mean(axis=0)-predictors_dict['normal'].mean(axis=0),
                                              'Pattern',new_window,cmap='bwr')
                norm_dec_diff_mat[1].axvline(0.5,c='w',ls='--')
                norm_dec_diff_mat[0].show()

                all_dev_plot = plt.subplots(4,5,figsize=(30,20))
                for i, (trial_response,ax) in enumerate(zip(predictors_dict[dev_type],all_dev_plot[1].flatten())):
                    norm_dec_diff_mat = plot_psth(trial_response - predictors_dict['normal'].mean(axis=0),
                                                  'Pattern', new_window, cmap='bwr',plot=(all_dev_plot[0],ax),
                                                  vmin=-10,vmax=30)
                    ax.axvline(0.5, c='w', ls='--')
                all_dev_plot[0].show()

        # stage5 analysis on representation of rising vs non-rising tones
        if sess_info['sess_order'] == 'main' and 5 in sessions[sessname].td_df['Stage'].values:
            # get event idx
            window = [0, 0.25]
            pip_predictors = {event_lbl: get_predictor_from_psth(sessions[sessname], event_lbl, psth_window, window,
                                                           mean=None,)
                              for event_lbl in sessions[sessname].sound_event_dict
                              if any(char in event_lbl for char in 'ABCD')}
            pip_idxs = {event_lbl: sessions[sessname].sound_event_dict[event_lbl].idx
                        for event_lbl in sessions[sessname].sound_event_dict
                        if any(char in event_lbl for char in 'ABCD')}
            pip_desc,pip_lbls,pip_names = get_pip_info(sessions[sessname].sound_event_dict,normal_patterns,
                                                       n_patts_per_rule)
            similarity = cosine_similarity([pred[:,:,-1].mean(axis=0) for pred in pip_predictors.values()])

            sort_keys = ['name','group',]
            plot_names = [pip_desc[i]['desc']
                          for i in [p for p in sorted(pip_desc,key=lambda x: [pip_desc[x][sort_key]
                                                                              for sort_key in sort_keys])]]
            plot_order = [pip_lbls.index(i)
                          for i in [p for p in sorted(pip_desc,key=lambda x: [pip_desc[x][sort_key]
                                                                              for sort_key in sort_keys])]]

            similarity_plot = plot_similarity_mat(similarity, plot_names,cmap='Reds',reorder_idx=plot_order)
            similarity_plot[0].set_size_inches(15, 13)
            similarity_plot[0].show()
            similarity_plot[0].savefig(ephys_figdir/f'pip_similarity_{sessname}.svg')

            # time series of similarity
            x_ser = np.linspace(window[0],window[1],pip_predictors['A-0'].shape[-1])
            similarity_over_time = [cosine_similarity([pred[:,:,t].mean(axis=0) for pred in pip_predictors.values()])
                                    for t in range(x_ser.shape[0])]
            similarity_over_time_arr = np.array(similarity_over_time)[:,:,plot_order]
            sim_over_time_tsplot = plt.subplots(similarity_over_time_arr.shape[2])
            for i in range(similarity_over_time_arr.shape[2]):
                sim_over_time_tsplot[1][i].plot(x_ser,
                                                (similarity_over_time_arr[:,0,i]),label=plot_names[i])
                sim_over_time_tsplot[1][i].set_xticks([])
                box = sim_over_time_tsplot[1][i].get_position()
                # sim_over_time_tsplot[1][i].set_position([box.x0, box.y0, box.width * 0.8, box.height])
                sim_over_time_tsplot[1][i].legend(loc='upper right',bbox_to_anchor=(1, 0.85))
            sim_over_time_tsplot[1][-1].set_xticks(x_ser)
            sim_over_time_tsplot[1][-1].locator_params(axis='x', nbins=6)
            sim_over_time_tsplot[1][-1].set_xlabel('Time from pip onset (s)')
            comp_pip = plot_names[0].replace("\n"," ")
            sim_over_time_tsplot[0].suptitle(f'Similarity to {comp_pip} over time',y=0.9)
            sim_over_time_tsplot[0].set_size_inches(9, 18)
            sim_over_time_tsplot[0].show()
            sim_over_time_tsplot[0].savefig(ephys_figdir/f'pip_similarity_over_time_{sessname}.svg')

            # time series of similarity
            # pip_positions = [1]
            sim_by_property_plot = plt.subplots(ncols=len('ABCD'),figsize=(32,16),sharey='all')
            for p in range(len('ABCD')):
                pips_by_property = {}
                for prop in ['idx','ptype']:
                    pips_by_property[prop] = get_list_pips_by_property(pip_desc,prop,[p+1])
                for prop in pips_by_property:
                    pop_rate_mats_to_pip = [[pip_predictors[pip] for pip in pips] for pips in pips_by_property[prop]]
                    sim_over_time_arrs = [get_sim_mat_over_time(pop_rate_mats_to_comp)
                                          for pop_rate_mats_to_comp in pop_rate_mats_to_pip]
                    [sim_by_property_plot[1][p].plot(x_ser,sim_over_time_arrs[i].mean(axis=1).mean(1),label=f'{prop} {i}')
                     for i in range(len(sim_over_time_arrs))]

                sim_by_property_plot[1][p].legend()
                sim_by_property_plot[1][p].set_xlabel('Time from pip onset (s)')
                sim_by_property_plot[1][p].set_ylabel('Similarity')
                sim_by_property_plot[1][p].set_title(f'Pip {p+1}')

            sim_by_property_plot[0].show()
            sim_by_property_plot[0].savefig(ephys_figdir/f'pip_similarity_by_property_{sessname}.svg')

            # similarity for indv pips

            pip_plot_lbls = [f'{"ABCD" if pi % n_patts_per_rule == 0 else "ABBA"} ({pi//n_patts_per_rule})'
                             for pi in range(len(main_patterns))]
            permute = True

            min_sim = np.round(np.quantile(similarity,0.1),1)
            pip_by_pip_sim_plot = plt.subplots(1,ncols=len('ABCD'),squeeze=False)
            preds_shuffled = {}

            for permute, plots in zip([False],pip_by_pip_sim_plot[1]):

                for pi, pip2use in enumerate('ABCD'):
                    single_pip_predictor = by_pip_predictors[pip2use]
                    if permute:
                        n_shuffled_splits = [permute_pip_preds(single_pip_predictor) for _ in range(n_shuffles)]
                        preds_shuffled[pip2use] = n_shuffled_splits
                        preds2use = [np.mean([e[i] for e in n_shuffled_splits],axis=0)
                                     for i in tqdm(range(len(single_pip_predictor)), total=len(single_pip_predictor),
                                                   desc='averaging shuffles')]
                    else:
                        preds2use = list(single_pip_predictor.values())
                    similarity_to_pip = cosine_similarity([pred[:, :, -1].mean(axis=0)
                                                           for pred in preds2use])
                    reordered_names, reordered_idxs = get_reordered_idx(pip_desc,pip_lbls,['ptype_i'],
                                                                        subset=[p for p in pip_lbls if pip2use in p])
                    reordered_idxs = [int(idx/4) for idx in reordered_idxs]
                    pip_plot_lbls = [e.split(' ')[-1] for e in reordered_names]
                    plot_similarity_mat(similarity_to_pip, pip_plot_lbls,
                                        reorder_idx=reordered_idxs,
                                        cmap='Reds', plot=(pip_by_pip_sim_plot[0],plots[pi]),
                                        im_kwargs=dict(vmin=min_sim,vmax=1),
                                        plot_cbar=True if pi == len('ABCD')-1 else False)
                    # plot sim scaled by self sim
                    if pip_by_pip_sim_plot[1].shape[0] < 2:
                        continue
                    pip_self_sims = self_sim_means.mean(axis=1)[pi*len(reordered_names):(pi+1)*len(reordered_names)]
                    sim_rescaled = similarity_to_pip.copy()
                    np.fill_diagonal(sim_rescaled,pip_self_sims)
                    sim_rescaled = sim_rescaled/pip_self_sims.reshape(-1,1)
                    plot_similarity_mat(sim_rescaled,
                                        pip_plot_lbls, reorder_idx=reordered_idxs,
                                        cmap='Reds', plot=(pip_by_pip_sim_plot[0],
                                                           pip_by_pip_sim_plot[1][-1][pi]),
                                        # im_kwargs=dict(vmin=0, vmax=1),
                                        plot_cbar=True if pi == len('ABCD') - 1 else False)
                    # pip_by_pip_sim_plot[1][pi].set_xticks()
                    # pip_by_pip_sim_plot[1][pi].set_yticks([])
                    plots[pi].set_title(f'pip {pi}')
            [plot.set_ylabel('') for pi, plot in enumerate(pip_by_pip_sim_plot[1].flatten()) if pi%len('ABCD')>0 ]
            # [plot.set_xlabel('') for plot in pip_by_pip_sim_plot[1][-1]]
            pip_by_pip_sim_plot[0].set_size_inches(20,5*pip_by_pip_sim_plot[1].shape[0])
            pip_by_pip_sim_plot[0].set_layout_engine('tight')
            pip_by_pip_sim_plot[0].show()
            sim_figdir = ephys_figdir.parent/'by_pip_similarity_by_rule_single_sorted_sessions'
            if not sim_figdir.is_dir():
                sim_figdir.mkdir()
            pip_by_pip_sim_plot[0].savefig( sim_figdir/ f'by_pip_similarity_{sessname}{"_permute" if permute else ""}_reordered.svg')

                # # ttest
                # all_pip_tests = {}
                # for pip in 'ABCD':
                #     null_dist_sims = np.array([cosine_similarity([pred[:, :, -1].mean(axis=0) for pred in shuffled]) for shuffled in preds_shuffled[pip]])
                #     null_dist_sims = null_dist_sims - null_dist_sims[:,0,0].reshape(-1,1,1)
                #     data_sims = cosine_similarity([pred[:, :, -1].mean(axis=0) for pred in by_pip_predictors[pip].values()])
                #     data_sims = data_sims - data_sims[0:,0]
                #     [print(f'{null_dist_sims[i,j].mean():.2f} {data_sims[i,j]:.2f} {ttest_1samp(null_dist_sims[:,i,j], data_sims[i,j])[1]:.10E}')
                #      for i in range(data_sims.shape[0]) for j in range(data_sims.shape[1])]
                #
                #     t_res = [ttest_1samp(null_dist_sims[:,i,j], data_sims[i,j])[1] for i in range(data_sims.shape[0])
                #              for j in range(data_sims.shape[1])]
                #     all_pip_tests[pip] = np.array(t_res).reshape(data_sims.shape[0],data_sims.shape[1])

                # print(t_res[1])

                # compare similarity of pips by half
                # similarity_by_half = [cosine_similarity([np.array_split(pred,2,axis=0)[i][:,:,-1].mean(axis=0)
                #                                          for pred in pip_predictors.values()])
                #                       for i in range(2)]
                # similarity_by_half_plot = plt.subplots(ncols=2,figsize=(20,10))
                # _ = [plot_similarity_mat(similarity_by_half[i], pip_names,cmap='Reds',
                #                          plot=(similarity_by_half_plot[0],similarity_by_half_plot[1][i]))
                #      for i, half in enumerate(similarity_by_half)]
                # similarity_by_half_plot[0].set_size_inches(30,13)
                # similarity_by_half_plot[0].show()
        # decoding accuracy over time
        if decode_over_time:
            psth_window = psth_window
            window = [-1,2]
            predictions_ts_array = sessions[sessname].map_preds2sound_ts('A-0',['all_vs_all'],
                                                                         psth_window,window,)
            predictions_ts_array = np.squeeze(predictions_ts_array)
            x_ser = np.linspace(window[0],window[1],predictions_ts_array.shape[2])
            # x_ser = np.linspace(0,1,predictions_ts_array.shape[2])

            prediction_ts_plot = plt.subplots(sharex='all',sharey='all')
            cmap = plt.get_cmap('PuBuGn')
            # colors = list(cmap(np.linspace(0, 1, len(all_dec_lbls))+0.25))
            colors =['k','slategray','midnightblue','darkslategray']
            for li, lbl in enumerate(['A-0','B-0','C-0','D-0']):
                lbl_pred = predictions_ts_array==li
                prediction_ts_plot[1].plot(x_ser,savgol_filter(lbl_pred.mean(axis=0).mean(axis=0),51,2),c=colors[li], label=lbl)


                # prediction_ts_plot[1].plot(x_ser,lbl_pred.mean(axis=0).mean(axis=0), label=lbl)
                # prediction_ts_plot[1][li].axis('off')
                # for dec_i, (dec_preds, dec_name) in enumerate(zip(predictions_ts_array,list('ABCD'))):
            [prediction_ts_plot[1].axvspan(t,t+0.15,fc='k',alpha=0.1)
             for t in np.arange(0,1,.25) if sess_info['sess_order'] == 'main']
            prediction_ts_plot[1].set_ylabel('prediction rate')
            # prediction_ts_plot[1].set_ylim(0,.7)
            prediction_ts_plot[1].set_xlim(-0.5,1.5)
            prediction_ts_plot[1].set_xlabel('Time from A onset (s)',fontsize=16)
            # prediction_ts_plot[1].set_yticks(np.linspace(0,0.3,4))
            # prediction_ts_plot[1].set_yticklabels(np.arange(0,0.4,4),fontsize=14)
            # prediction_ts_plot[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
            prediction_ts_plot[1].set_ylabel('prediction rate',fontsize=18)
            prediction_ts_plot[1].tick_params(axis='both', which='major', labelsize=16)
            prediction_ts_plot[1].locator_params(axis='both', nbins=4)
            prediction_ts_plot[1].legend(loc=1)
            prediction_ts_plot[0].set_constrained_layout('constrained')
            prediction_ts_plot[0].set_size_inches(4,3.5)
            prediction_ts_plot[0].show()
            prediction_ts_plot[0].savefig(ephys_figdir/f'decoding_acc_ts_no_base_{sessname}.svg')

        # pred_to_peak_plot = plt.subplots()
        # for li,lbl in enumerate(['A-0:,'B-0','C-0','D-0
        #     signal = savgol_filter((predictions_ts_array==li).mean(axis=0).mean(axis=0),51,2)
    #     # signal = signal-signal[:70].mean()
        #     algo = rpt.KernelCPD()
        #     algo.fit(signal)
        #     # t_change_point=[x_ser[t-1] for t in algo.predict(n_bkps=3)][0]
        #     t_change_point=[x_ser[t] for t in np.where(signal>0.2)[0]][0]
        #     # t_change_point=[x_ser[t-1] for t in np.where(signal>signal.max()*0.5)[0]][0]
        #     pred_to_peak_plot[1].scatter(li, np.subtract(t_change_point, li*0.25),c=colors[li])
        #     prediction_ts_plot[1].axvline(t_change_point,c=colors[li])
        # # pred_to_peak_plot[1].set_ylim(0,0.2)
        # # pred_to_peak_plot[1].plot
        # pred_to_peak_plot[0].show()
        # prediction_ts_plot[0].show()

        sessions[sessname].pickle_obj(ceph_dir/'Dammy'/'ephys_concat_pkls')
    cross_sess_decode_flag = False
    if cross_sess_decode_flag:
        sessnames = list(sessions.keys())
        # sess_decoder2use = (all_sess_info[all_sess_info['trialdata_path'].str.contains('TrialData')]
        #                     ['sound_bin_stem'].values[0].replace('SoundData_',''))
        # sess_decoder2use = sessnames[1]
        window = [-1,2]
        for sessname in sessnames:
            sess_decoder2use = sessname
            for si, sessname in enumerate(sessnames):
                for pip in ['A-0','B-0','C-0','D-0']:
                    if sessname == sess_decoder2use:
                        continue
                    model2use = np.array(sessions[sess_decoder2use].decoders['all_vs_all'].models).flatten()
                    pip_predictor = get_predictor_from_psth(sessions[sessname], pip, psth_window, window, mean=None)
                    cross_sess_preds = np.array([predict_1d(model2use,ts)
                                                 for ts in tqdm(pip_predictor,total=len(pip_predictor),desc=f'predicting {sessname}')])
                    x_ser = np.linspace(window[0],window[1],cross_sess_preds.shape[2])
                    cross_sess_preds_tsplot = plt.subplots()
                    for li, lbl in enumerate(['A-0','B-0','C-0','D-0']):
                        lbl_pred = cross_sess_preds==li
                        cross_sess_preds_tsplot[1].plot(x_ser,savgol_filter(lbl_pred.mean(axis=0).mean(axis=0),51,2), label=lbl)
                    cross_sess_preds_tsplot[1].legend()
                    cross_sess_preds_tsplot[1].axvline(0,c='k',ls='--')
                    [cross_sess_preds_tsplot[1].axvspan(t, t + 0.15, fc='k', alpha=0.1) for t in np.arange(0, 1, .25)
                     if all_sess_info.iloc[si]['sess_order'] == 'main']

                    cross_sess_preds_tsplot[0].savefig(ephys_figdir/f'cross_sess_preds_{sess_decoder2use}_on_{sessname}_for_{pip}.svg')




