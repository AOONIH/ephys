import warnings

import plot_funcs
from behviour_analysis_funcs import get_all_cond_filts, get_main_sess_patterns, get_main_sess_td_df
from decoding_funcs import predict_1d
from ephys_analysis_funcs import *
import platform
import argparse
import yaml
from scipy.stats import pearsonr, ttest_ind
from scipy.signal import savgol_filter

from io_utils import posix_from_win, load_sess_pkl
from plot_funcs import plot_decoder_accuracy, plot_psth, plot_psth_ts, plot_ts_var
from regression_funcs import run_glm, run_regression
from neural_similarity_funcs import *
from postprocessing_utils import get_sorting_dirs
from datetime import datetime
from matplotlib.colors import TwoSlopeNorm
import pandas as pd

from sess_dataclasses import Session, get_predictor_from_psth

if __name__ == "__main__":
    print('args')
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('sess_date')
    parser.add_argument('--sorter_dirname',default='from_concat',required=False)
    parser.add_argument('--sess_top_tag', default='')
    parser.add_argument('--sess_top_filts', default='')
    parser.add_argument('--synth_data',default=0,type=int)
    parser.add_argument('--rel_sorting_path',default='')
    parser.add_argument('--ow',default=1,type=int)

    args = parser.parse_args()
    print(f'{args = }')
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])
    pkl_dir = ceph_dir / 'Dammy' / 'ephys_concat_pkls'

    plt.ioff()
    # try: gen_metadata(ceph_dir/posix_from_win(r'X:\Dammy\ephys\session_topology.csv'),ceph_dir,ceph_dir/'Dammy'/'harpbins')
    sess_topology_path = ceph_dir/posix_from_win(rf'X:\Dammy\Xdetection_mouse_hf_test\session_topology_{args.sess_top_tag}.csv')
    # try: gen_metadata(sess_topology_path,ceph_dir,
    #                   col_name='beh_bin',harp_bin_dir='')
    # except OSError: pass
    gen_metadata(sess_topology_path, ceph_dir,
                 col_name='beh_bin', harp_bin_dir='')
    session_topology = pd.read_csv(sess_topology_path)
    # win_rec_dir = r'X:\Dammy\ephys\DO79_2024-01-17_15-20-19_001\Record Node 101\experiment1\recording1'
    name,date = args.sess_date.split('_')
    date = int(date)
    all_sess_info = session_topology.query('name==@name & date==@date').reset_index(drop=True)
    if args.sess_top_filts:
        all_sess_info = all_sess_info.query(args.sess_top_filts)

    if args.rel_sorting_path:
        dir1_name, dir2_name = Path(args.rel_sorting_path).parts
    else:
        dir1_name, dir2_name = 'sorting_no_si_drift', 'kilosort2_5_ks_drift'

    ephys_dir = ceph_dir / 'Dammy' / 'ephys'
    date_str = datetime.strptime(str(date), '%y%m%d').strftime('%Y-%m-%d')
    print(f'{name}_{date_str}')

    # if all_sess_info.shape[0] == 1:
    #     sorter_dirname = 'sorter_output'
    # else:
    sorter_dirname = args.sorter_dirname
    # sort_dirs = get_sorting_dirs(ephys_dir, f'{name}_{date_str}',dir1_name, dir2_name, sorter_dirname)
    # sort_dirs = [e for ei,e in enumerate(sort_dirs) if ei in all_sess_info.index]

    ephys_figdir = ceph_dir/'Dammy'/'figures'/f'sim_analysis_based_{dir1_name}_{dir2_name}'
    if args.synth_data:
        if 'synth' not in ephys_figdir.stem:
            ephys_figdir = ephys_figdir.with_stem(f'{ephys_figdir.stem}_synth_data')
    if not ephys_figdir.is_dir():
        ephys_figdir.mkdir()

    sessions = {}
    psth_window = [-2, 3]
    main_sess = session_topology.query('sess_order=="main" & name==@name & date==@date')
    if 'tdata_file' in main_sess.columns:
        main_sess_td_name = posix_from_win(main_sess['tdata_file'].iloc[0],ceph_linux_dir=config['home_dir_linux'])
    else:
        main_sess_td_name = Path(main_sess['sound_bin'].iloc[0].replace('_SoundData', '_TrialData')).with_suffix('.csv').name
    # get main sess pattern
    home_dir = Path(config[f'home_dir_{sys_os}'])
    try:
        main_patterns = get_main_sess_patterns(name,date, main_sess_td_name=main_sess_td_name, home_dir=home_dir)
    except FileNotFoundError:
        main_patterns = [[0,0,0,0]]

    main_pattern = main_patterns[0]
    # sort_dirs =

    plot_psth_decode = True
    decode_over_time = False

    cond_filters = get_all_cond_filts()
    print(f'{main_sess_td_name =}')

    for (_,sess_info) in all_sess_info.iterrows():
        # if sess_info['sess_order'] != 'main':
        #     continue
        # if 'good_units' not in spike_dir.parent.name:
        #     pass
        spike_dir = ceph_dir/posix_from_win(sess_info['ephys_dir'])/args.rel_sorting_path/args.sorter_dirname
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

        sess_pkl_path = pkl_dir / f'{sessname}.pkl'
        print(f'looking for {sess_pkl_path}')
        if sess_pkl_path.is_file() and not args.ow:
            print(f'found {sess_pkl_path.name}, loading')
            sessions[sessname] = load_sess_pkl(sess_pkl_path)
            print(f'loaded {sess_pkl_path.name}')
        else:
            sessions[sessname] = Session(sessname, ceph_dir)

        # sess_td_path = next(home_dir.rglob(f'*{sessname}_TrialData.csv'))
        # sess_td_path = sess_info['trialdata_path']

        sound_bin_path = Path(sess_info['sound_bin'])
        beh_bin_path = Path(sess_info['beh_bin'])

        # if sess_info['sess_order'] == 'main':  # load trial_data


        sessions[sessname].load_trial_data(get_main_sess_td_df(name,date,main_sess_td_name,home_dir)[1])
        # normal = sessions[sessname].td_df[sessions[sessname].td_df['Tone_Position'] == 0]['PatternID'].iloc[0]

        if sessions[sessname].td_df['Stage'].iloc[0] <=2 and sess_info['sess_order'] == 'main':
            print('stage 2, nothing to analyse')
            continue
        main_patterns = get_main_sess_patterns(td_df=sessions[sessname].td_df)
        main_patterns = sorted(main_patterns, key=lambda x: (x[0], np.any(np.diff(x)<=0)))
        print(f'{main_patterns=}')
        patts_by_rule = get_patts_by_rule(main_patterns)
        n_patts_per_rule = max(len(e) for e in patts_by_rule.values())
        print(f'{main_patterns=}')
        if sessions[sessname].td_df['Stage'].iloc[0] ==3:
            normal_patterns = main_patterns
            if len(normal_patterns) > 1 and sessions[sessname].td_df['Stage'].iloc[0] == 3:
                warnings.warn(f'{sessname} has more than one normal pattern for stage 3')
        elif sessions[sessname].td_df['Stage'].iloc[0] == 4:
            normal_patterns = get_main_sess_patterns(
                td_df=sessions[sessname].td_df.query(cond_filters['normal_exp']))
        elif sessions[sessname].td_df['Stage'].iloc[0] == 5:
            normal_patterns = [e for i, e in enumerate(main_patterns) if i % 2 == 0] # write function for getting rule (sub start ad cnt unique)
        else:
            normal_patterns = main_patterns

        normal = main_pattern
        # normal = [int(pip) for pip in normal.split(';')]
        if -1 in sessions[sessname].td_df['Pattern_Type'].unique():
            new_normal = sessions[sessname].td_df[sessions[sessname].td_df['Pattern_Type'] == -1]['PatternID'].iloc[0]
            # new_normal = [int(pip) for pip in new_normal.split(';')]
        else:
            new_normal = None

        # else:
        #     # normal = [int(pip) for pip in main_pattern.split(';')]
        #     normal= main_pattern
        #     new_normal = None
        beh_write_data_path = beh_bin_path.with_stem(f'{beh_bin_path.stem}_write_data').with_suffix('.csv')
        beh_write_data_path = ceph_dir / posix_from_win(str(beh_write_data_path))

        sessions[sessname].init_spike_obj(spike_times_path, spike_cluster_path, start_time, parent_dir=spike_dir,
                                          beh_write_data_path=beh_write_data_path,rec_dir=recording_dir)
        if not sessions[sessname].sound_event_dict or args.ow:
            _parts = sessname.split('_')
            '_'.join([_parts[0],'SoundData',_parts[1]])

            abs_writes_path = str(sound_bin_path.with_stem(f'{sound_bin_path.stem}_write_indices'))
            sessions[sessname].init_sound_event_dict(ceph_dir/posix_from_win(abs_writes_path).with_suffix('.csv'),
                                                     patterns=main_patterns,normal_patterns=normal_patterns)
        sessions[sessname].get_event_free_zscore()
        synth_data_flag = args.synth_data if args.synth_data else None
        n_units = len(sessions[sessname].spike_obj.units)
        if synth_data_flag:
            sessions[sessname].spike_obj.unit_means = (np.zeros(n_units), np.ones(n_units))

        pip_idxs = {event_lbl: sessions[sessname].sound_event_dict[event_lbl].idx
                    for event_lbl in sessions[sessname].sound_event_dict 
                    if any(char in event_lbl for char in 'ABCD')}
        pip_desc, pip_lbls, pip_names = get_pip_info(sessions[sessname].sound_event_dict,
                                                     main_patterns if sess_info['sess_order'] == 'main' else list(pip_idxs.values()),
                                                     n_patts_per_rule)
        sessions[sessname].pip_desc = pip_desc
        # generate patterned unit rates
        # sessions[sessname].get_grouped_rates_by_property(pip_desc,'ptype',0.1)
        sessions[sessname].get_sound_psth(psth_window=psth_window, zscore_flag=False, baseline_dur=0, redo_psth=False,
                                          use_iti_zscore=False, synth_data=synth_data_flag)
        sessions[sessname].pickle_obj(pkl_dir)
        # exit()

        n_shuffles = 1000
        by_pip_predictors = {}
        for pi, pip2use in enumerate('ABCD'):
            by_pip_predictors[pip2use] = {event_lbl: get_predictor_from_psth(sessions[sessname], event_lbl,
                                                                             psth_window, [0,0.25],
                                                                             mean=None)
                                          for event_lbl in sessions[sessname].sound_event_dict
                                          if any(char in event_lbl for char in pip2use)}
        # get self sims for all pips
        # assert check_unique_across_dim([list(e.values()) for e in by_pip_predictors.values()]), 'overlapping pips'
        n_shuffles = 1000
        # assert not check_unique_across_dim([np.random.permutation(len(by_pip_predictors['A']['A-0']))
        #                                     for _ in range(1000)]), 'Permute shuffling across unit or time axis'
        self_sims = [[compute_self_similarity(responses[shuffled], cv_folds=5) for shuffled in
                      [np.random.permutation(len(responses)) for _ in range(1000)]] for responses in
                     sum([list(e.values()) for e in by_pip_predictors.values()], [])]
        self_sim_means = np.squeeze([np.mean(e, axis=1) for e in self_sims])
        self_sim_names, self_sims_idx = get_reordered_idx(pip_desc, ['group'])

        self_sims_plot = plt.subplots(figsize=(18, 8))
        all_pip_labels = [e.split(' ')[-1] for e in self_sim_names]
        self_sims_plot[1].boxplot(self_sim_means[self_sims_idx].tolist(), labels=self_sim_names)
        self_sims_plot[1].set_title('Within pip self similarity', fontsize=20)
        self_sims_plot[1].set_ylabel('Self similarity', fontsize=18)
        self_sims_plot[0].set_layout_engine('tight')
        self_sims_plot[0].show()
        self_sims_plot[0].savefig(ephys_figdir / f'pip_self_sim_{sessname}.pdf')

        # event_psth_dict = {e: by_pip_predictors[e] for e in sum(list([e.valuesby_pip_predictors.values()]), [])}
        event_psth_dict = {k:v for e_key in by_pip_predictors for (k, v) in by_pip_predictors[e_key].items()}

        compared_pips_plot = plt.subplots(4,figsize=(6, 18))
        for pi,pip in enumerate(['A', 'B', 'C', 'D']):
            event_names = [p for p in event_psth_dict if pip in p]
            compared_pips = compare_pip_sims_2way([event_psth_dict[e] for e in event_names])

            mean_comped_sims = [np.squeeze(pip_sims)[:, 0, 1] for pip_sims in np.array_split(compared_pips[0], (len(event_names)))]
            mean_comped_sims.append(np.squeeze(compared_pips[1][:, 0, 1])) if len(event_names) > 1 else None
            compared_pips_plot[1][pi].boxplot(mean_comped_sims,
                                              labels=(event_names+['vs '.join(event_names)] if len(event_names) > 1
                                                      else event_names))
            compared_pips_plot[1][pi].set_ylim([0, 1])
            compared_pips_plot[1][pi].set_ylabel('cosine similarity')
        compared_pips_plot[0].show()
        compared_pips_plot[0].savefig(ephys_figdir/f'pips_compared_{sessname}.pdf')

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
            psth_ts_plot[0].savefig(ephys_figdir/f'A_psth_ts_{sessname}.pdf')

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
                preds_sim_over_pips = [get_predictor_from_psth(sessions[sessname], key, psth_window, window, mean=np.mean)
                                       for key in ['X', 'base']]
                feats_sim_over_pips = [np.full(mat.shape[0], i) for i, mat in enumerate(preds_sim_over_pips)]

                sessions[sessname].init_decoder('X_to_base', np.vstack(preds_sim_over_pips), np.hstack(feats_sim_over_pips))
                sessions[sessname].run_decoder('X_to_base', ['data','shuffle'], dec_kwargs={'cv_folds': 0},
                                               plot_flag=True,)
                # sessions[sessname].decoders['X_to_base'].accuracy_plot[0].show()
                sessions[sessname].decoders['X_to_base'].accuracy_plot[0].savefig(ephys_figdir/f'X_to_base_{sessname}.pdf')
            # decoder to base
            dec_kwargs = {'cv_folds': 10}
            for pip in ['A-0','B-0','C-0','D-0']:
                preds_all_vs_all = [get_predictor_from_psth(sessions[sessname], key, psth_window, window, mean=np.mean)
                                    for key in [pip, 'base']]
                # _predictor_list[-1] = subset_base_preds
                feats_all_vs_all = [np.full(mat.shape[0], i) for i, mat in enumerate(preds_all_vs_all)]

                sessions[sessname].init_decoder(f'{pip}_to_base', np.vstack(preds_all_vs_all), np.hstack(feats_all_vs_all))
                sessions[sessname].run_decoder(f'{pip}_to_base', ['data','shuffle'], dec_kwargs={'cv_folds':0})
                # sessions[sessname].decoders[f'{pip}_to_base'].accuracy_plot[0].set_constrained_layout('constrained')
                # sessions[sessname].decoders[f'{pip}_to_base'].accuracy_plot[0].savefig(ceph_dir/'Dammy'/'figures'/'ephys'/f'{pip}_to_base_accr.pdf',)

            tone2base_all_plot = plt.subplots()
            # for pi, pip in enumerate(['A-0','B-0','C-0','D-0']):
            #     metric2plot = sessions[sessname].decoders[f'{pip}_to_base'].fold_accuracy
            #     plot_decoder_accuracy(metric2plot, pip, fig=tone2base_all_plot[0], ax=tone2base_all_plot[1],
            #                           start_loc=pi, n_features=2)
            # metric2plot = sessions[sessname].decoders[f'{pip}_to_base'].fold_accuracy
            # plot_decoder_accuracy(metric2plot, pip, fig=tone2base_all_plot[0], ax=tone2base_all_plot[1],
            #                       start_loc=pi, n_features=2)
            # plot shuffle
            sessions[sessname].init_decoder(f'A_to_base_shuffle', np.vstack([preds_all_vs_all[0], preds_all_vs_all[-1]]),
                                            np.hstack([feats_all_vs_all[0], feats_all_vs_all[-1]]))
            sessions[sessname].run_decoder(f'A_to_base_shuffle', ['data', 'shuffle'], dec_kwargs={'cv_folds': 0,'shuffle':True})
            plot_decoder_accuracy(sessions[sessname].decoders[f'A_to_base_shuffle'].fold_accuracy, 'A_shuffle',
                                  fig=tone2base_all_plot[0], ax=tone2base_all_plot[1], plt_kwargs={'c':'k'},
                                  start_loc=4, n_features=2)

            # tone2base_all_plot[1].set_xticks(np.arange(len('ABCD')+1))
            # tone2base_all_plot[1].set_xticklabels(list('ABCD')+['A_shuffle'])
            tone2base_all_plot[1].get_legend().remove()
            # tone2base_all_plot[0].show()
            tone2base_all_plot[0].set_constrained_layout('constrained')
            tone2base_all_plot[0].savefig(ephys_figdir/f'ABCD_to_base_perf_{sessname}.pdf')

            window = [0, 0.25]
            preds_sim_over_pips = [get_predictor_from_psth(sessions[sessname], key, psth_window, window, mean=None)
                                   for key in ['A-0','B-0','C-0','D-0','base']]

            # compute cosine similarity
            pip_sim_over_trials = [cosine_similarity([trial_resp[:,-1] for trial_resp in pip_responses])
                                   for pip_responses in preds_sim_over_pips]
            pip_sim_over_trials_plot = plt.subplots(len(preds_sim_over_pips), figsize=(8, 32))
            [plot_similarity_mat(sim, ''*sim.shape[0], plot=(pip_sim_over_trials_plot[0],ax), cmap='Reds')
             for sim,ax in zip(pip_sim_over_trials,pip_sim_over_trials_plot[1])]
            pip_sim_over_trials_plot[0].set_layout_engine('compressed',w_pad=0.5)
            pip_sim_over_trials_plot[0].savefig(ephys_figdir/f'pip_sim_over_trials_{sessname}.pdf')
            mean_pip_sim = [np.mean(sim[~np.eye(sim.shape[0],dtype=bool)]) for sim in pip_sim_over_trials]
            pip_sim_over_pips = cosine_similarity([pred[:,:,-1].mean(axis=0) for pred in preds_sim_over_pips])
            # pip_sim_over_pips = cosine_similarity([pred.mean(axis=0).mean(axis=-1) for pred in _predictor_list])
            pip_sim_pip_plot = plot_similarity_mat(pip_sim_over_pips, np.arange(len(preds_sim_over_pips)), cmap='Reds')
            pip_sim_pip_plot[0].savefig(ephys_figdir/f'pip_sim_over_pips_{sessname}.pdf')

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
            # pearson_plot[0].savefig(ephys_figdir/f'pearson_no_base_corr_matrix_{sessname}.pdf',)

            # run all decoder with base for tseries
            preds_all_vs_all = [get_predictor_from_psth(sessions[sessname], key, psth_window, window, mean=np.mean)
                                for key in ['A-0', 'B-0', 'C-0', 'D-0', 'base']]
            feats_all_vs_all = [np.full(mat.shape[0], i) for i, mat in enumerate(preds_all_vs_all)]

            all_dec_lbls = ['A-0', 'B-0', 'C-0', 'D-0', 'base']
            all_dec_name = 'all_vs_all'

            sessions[sessname].init_decoder(all_dec_name, np.vstack(preds_all_vs_all), np.hstack(feats_all_vs_all),
                                            model_name='logistic')
            dec_kwargs = {'cv_folds': 10}
            sessions[sessname].run_decoder(all_dec_name, ['data', 'shuffle'], dec_kwargs=dec_kwargs)

            sessions[sessname].decoders[all_dec_name].plot_confusion_matrix(all_dec_lbls, include_values=False,cmap='copper')
            sessions[sessname].decoders[all_dec_name].cm_plot[0].set_size_inches(4,3.5)
            sessions[sessname].decoders[all_dec_name].cm_plot[1].tick_params(axis='both', which='major', labelsize=16)
            # sessions[sessname].decoders['all_vs_all'].cm_plot[0].set_constrained_layout('constrained')
            sessions[sessname].decoders[all_dec_name].cm_plot[0].savefig(ephys_figdir/
                                                                         f'ABCD_base_cm_{sessname}.pdf')

            plot_funcs.plot_decoder_accuracy(all_dec_lbls, )
            sessions[sessname].decoders[all_dec_name].accuracy_plot[0].savefig(ephys_figdir/
                                                                               f'all_vs_all_accuracy_{sessname}.pdf')

        home_dir = Path(config[f'home_dir_{sys_os}'])
        if sess_info['sess_order'] == 'main':
            if 3 in sessions[sessname].td_df['Stage'].values:
                # sessions[sessname].load_trial_data(f'{sess_td_path}.csv',home_dir,
                #                                    rf'H:\data\Dammy\{sessname.split("_")[0]}\TrialData')
                idx_bool = sessions[sessname].td_df['local_rate'] <= 0.2
                idx_bool2 = sessions[sessname].td_df['local_rate'] >= 0.8
                recent_pattern_trials = sessions[sessname].td_df[idx_bool].index.get_level_values('trial_num').to_numpy()
                distant_pattern_trials = sessions[sessname].td_df[idx_bool2].index.get_level_values('trial_num').to_numpy()

                # cumsum_plot = plt.subplots()
                # cumsum_plot[1].plot(np.cumsum(idx_bool),label='freq')
                # cumsum_plot[1].plot(np.cumsum(idx_bool2),label='rare')
                # cumsum_plot[1].legend()
                # cumsum_plot[1].set_xlabel('trial number')
                # cumsum_plot[1].set_title('distribution of rare vs freq trials over session')
                # cumsum_plot[0].set_constrained_layout('constrained')
                # cumsum_plot[0].savefig(ephys_figdir/f'02_08_local_rate_cumsum_plot_{sessname}.pdf')

                rec_dist_decoder_plot = plt.subplots()
                # dec_events = ['A-0','B-0','C-0','D-0','A_shuffle','A_halves']
                dec_events = ['A-0','X','A-0_shuffle']
                for pi,pip in enumerate(dec_events):
                    pip_id = pip.split('_')[0]

                    preds_norm_dev = get_predictor_from_psth(sessions[sessname], pip_id, psth_window, [0, 1], mean=np.mean)
                    recent_idx_bool = np.isin(sessions[sessname].sound_event_dict[pip_id].trial_nums,recent_pattern_trials)
                    distant_idx_bool = np.isin(sessions[sessname].sound_event_dict[pip_id].trial_nums,distant_pattern_trials)
                    if 'badpred' in pip:
                        rec_dist_predictors = preds_norm_dev[recent_idx_bool[:len(preds_norm_dev)]], preds_norm_dev[recent_idx_bool[:len(preds_norm_dev)]]
                    elif 'halves' in pip:
                        rec_dist_predictors = np.array_split(preds_norm_dev,2)
                    else:
                        rec_dist_predictors = preds_norm_dev[recent_idx_bool[:len(preds_norm_dev)]], preds_norm_dev[distant_idx_bool[:len(preds_norm_dev)]]
                    print('len 0.1/0.9 predictors',len(rec_dist_predictors[0]),len(rec_dist_predictors[1]))
                    rec_dist_features = [np.full(e.shape[0], ei) for ei, e in enumerate(rec_dist_predictors)]

                    plt_kwargs = {}
                    dec_kwargs = {'cv_folds': 5}
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

                rec_dist_decoder_plot[0].savefig(ephys_figdir/f'02_vs_08_local_bin5_acc_{sessname}.pdf')

                # do as boxplot
                rare_freq_boxplot = plt.subplots()
                perfs = [np.array(sessions[sessname].decoders[f'rec_dist_{pip}'].accuracy) for pip in dec_events]
                # [rare_freq_boxplot[1].boxplot(perf,pi) for pi, perf in enumerate(perfs)]
                rare_freq_boxplot[1].boxplot(perfs,bootstrap=100000,labels=dec_events,)
                rare_freq_boxplot[1].set_ylim(0.25,0.85)
                rare_freq_boxplot[1].set_yticks([0,.25,0.5,1.05])
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
                rare_freq_boxplot[0].savefig(ephys_figdir/f'rare_freq_decoding_boxplot_patt_window_{sessname}.pdf')
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

                rare_freq_psth[0].savefig(ephys_figdir/f'rare_vs_freq_psth_ts_{sessname}.pdf')
                np.save(ceph_dir/'Dammy'/'ephys_pkls'/f'rare_vs_freq_array_{sessname}.npy',rare_A-freq_A)

                # bootstrap data
                n_resamples = 9999
                # for n in n_resamples:
                #     pass
                rare_freq_diff_plot = plot_psth((rare_A-freq_A),'Time from A',[-2,3],cmap='bwr',
                                                cbar_label='zscored firing rate( rare - frequent)')
                rare_freq_diff_plot[0].savefig(ephys_figdir/f'rare_vs_freq_psth_{sessname}.pdf')

                # rare_freq_schem_plot = plt.subplots()
                # switches = sessions[sessname].td_df['PatternPresentation_Rate'][sessions[sessname].td_df['PatternPresentation_Rate'].diff() != 0]
                # for si,switch in enumerate(switches):
                #     if si==len(switches)-1:
                #         end = sessions[sessname].td_df.shape[0]
                #     else:
                #         end = switches.iloc[[si+1]].index.to_frame()['trial_num'][0]
                #     rare_freq_schem_plot[1].axvspan(switches.iloc[[si]].index.to_frame()['trial_num'][0],end,fc=f'C{1 if switch==0.9 else 0}',
                #                                     alpha=0.3)
                # rare_freq_schem_plot[1].plot((1-sessions[sessname].td_df['local_rate']),c='k',)
                # rare_freq_schem_plot[1].tick_params(axis='both', which='major', labelsize=14)
                # rare_freq_schem_plot[0].show()
                # rare_freq_schem_plot[0].savefig(ephys_figdir/f'rate_switches_{sessname}.pdf')
                #
                # # trial by trial pearsonr
                # # pearson_corr_all = np.zeros((len(A_psth), len(A_psth)))
                # pearsonr_list = [np.array([pearsonr(aa.mean(axis=1),a.mean(axis=1))[0] for a in A_psth]) for aa in A_psth]
                # pearson_corr_all = np.vstack(pearsonr_list)
                #
                # pearson_corr_sorted = np.vstack([pearson_corr_all[distant_idx_bool],pearson_corr_all[~distant_idx_bool]])
                # pearson_plot = plot_2d_array_with_subplots(pearson_corr_all, cbar_height=20,cmap='cividis',)
                # r2f_switch = ((sessions[sessname].td_df['Tone_Position']==0).cumsum()
                #                   [sessions[sessname].td_df.PatternPresentation_Rate.diff()<0])
                # f2r_switch = ((sessions[sessname].td_df['Tone_Position']==0).cumsum()
                #                   [sessions[sessname].td_df.PatternPresentation_Rate.diff()>0])
                # # pearson_plot[1].axvline(recent_idx_bool.sum(), c='k', ls='--')
                # # pearson_plot[1].axhline(recent_idx_bool.sum(), c='k', ls='--')
                # [pearson_plot[1].axvline(i,c='w',ls='--') for i in r2f_switch.values if i not in [0,1]]
                # [pearson_plot[1].axhline(i,c='lightcoral',ls='--') for i in f2r_switch.values if i not in [0,1]]
                #
                # pearson_plot[1].invert_yaxis()
                # # pearson_plot[1].set_xticklabels(['','A-0','B-0','C-0','D-0',])
                # # pearson_plot[1].set_yticklabels(['','A-0','B-0','C-0','D-0'])
                # pearson_plot[2].ax.set_ylabel("Pearson's correlation", rotation=270, labelpad=12)
                # pearson_plot[1].tick_params(axis='both', which='major', labelsize=18)
                # pearson_plot[2].ax.tick_params(axis='y', which='major', labelsize=14)
                # pearson_plot[0].set_size_inches(3.5,3)
                # pearson_plot[0].show()
                # pearson_plot[0].savefig(ephys_figdir / f'pearson_tofirst_A_{sessname}.pdf', )

        if sess_info['sess_order'] == 'main' and 4 in sessions[sessname].td_df['Stage'].values:
            new_window = [-1, 2]
            preds_norm_dev = get_predictor_from_psth(sessions[sessname], 'A-0', psth_window, new_window, mean=None,baseline=1)
            normal_responses = get_predictor_from_psth(sessions[sessname], 'A-0', psth_window, new_window, mean=None,baseline=1)
            deviant_responses = get_predictor_from_psth(sessions[sessname], 'A-1', psth_window, new_window, mean=None,baseline=1)
            dev_ABBA1_responses = get_predictor_from_psth(sessions[sessname], 'A-2', psth_window, new_window, mean=None,baseline=1)

            # new_norm_predictors =
            x_ser = np.linspace(new_window[0], new_window[1], preds_norm_dev.shape[-1])
            # norm_trial_nums = sessions[sessname].td_df.query('(Pattern_Type == 0) & Tone_Position == 0 & '
            #                                                  'local_rate == 0.0 & Session_Block == 3 & N_TonesPlayed == 4').index.to_numpy()
            # dev_trial_nums = sessions[sessname].td_df.query('(Pattern_Type == 1) & Tone_Position == 0').index.to_numpy()
            # newnorm_trial_nums = sessions[sessname].td_df.query('(Pattern_Type == -1) & Tone_Position == 0').index.to_numpy()
            predictors_dict = {}
            features_dict = {}
            pred_names = ['normal','deviant','dev_ABBA1']
            colors = ['saddlebrown','chocolate','darkgreen']
            psth_ts_plot=plt.subplots()
            for di, (name, responses, color) in enumerate(zip(pred_names[:],[normal_responses[-20:],deviant_responses,dev_ABBA1_responses],
                                                              colors)):
                # idx_bool = np.isin(sessions[sessname].sound_event_dict['A-0'].trial_nums - 1,
                #                    trial_nums)
                predictors_dict[name] = responses
                features_dict[name] = np.full(predictors_dict[name].shape[0],di)
                if di<3:
                    smoothed_data = savgol_filter(predictors_dict[name].mean(axis=0),10,2,axis=1)
                    psth_ts_plot = plot_psth_ts(smoothed_data,x_ser,label=name,plot=psth_ts_plot,c=color)
                    plot_ts_var(x_ser,smoothed_data,color,psth_ts_plot[1])
                    # two slope color bar
                    cmap_norm = TwoSlopeNorm(vmin=smoothed_data.min(), vmax=smoothed_data.max(), vcenter=0)
                    psth_plot = plot_psth(smoothed_data,'Time since pattern onset (s)',new_window,cmap='bwr',
                                          norm=cmap_norm)
                    psth_plot[1].set_title(name)
                    psth_ts_plot[1].legend()
                    [psth_ts_plot[1].axvspan(t,t+0.15,fc='k',alpha=0.1) for t in np.arange(0,1,0.25)]
                psth_plot[0].show()
            psth_ts_plot[1].tick_params(axis='both', which='major', labelsize=16)
            psth_ts_plot[1].locator_params(axis='both', nbins=4)
            psth_ts_plot[1].locator_params(axis='both', nbins=4)
            psth_ts_plot[0].set_size_inches(3.5,3)
            psth_ts_plot[0].show()
            psth_ts_plot[0].savefig(ephys_figdir/f'norm_dev_psth_ts_{sessname}.pdf')

            for dev_type in pred_names[1:]:
                response_diff_mat = predictors_dict[dev_type].mean(axis=0)-predictors_dict['normal'][-20:].mean(axis=0)
                cmap_norm = TwoSlopeNorm(vmin=response_diff_mat.min(), vmax=1.5, vcenter=0)
                norm_dev_diff_mat = plot_psth(response_diff_mat,'Pattern',new_window,cmap='bwr',norm=cmap_norm)
                norm_dev_diff_mat[1].axvline(0,c='k',ls='--')
                norm_dev_diff_mat[1].axvline(0.5,c='k',ls='--')
                norm_dev_diff_mat[1].set_title(f'{dev_type} - {pred_names[0]}')
                norm_dev_diff_mat[0].show()
                norm_dev_diff_mat[0].savefig(ephys_figdir/f'norm_dev_diff_mat_{sessname}_{dev_type}.pdf')

            response_diff_mat = predictors_dict['dev_ABBA1'].mean(axis=0) - predictors_dict['deviant'].mean(
                axis=0)
            cmap_norm = TwoSlopeNorm(vmin=response_diff_mat.min(), vmax=1.5, vcenter=0)
            norm_dev_diff_mat = plot_psth(response_diff_mat, 'Pattern', new_window, cmap='bwr', norm=cmap_norm)
            norm_dev_diff_mat[1].axvline(0, c='k', ls='--')
            norm_dev_diff_mat[1].axvline(0.5, c='k', ls='--')
            norm_dev_diff_mat[1].set_title(f'dev_ABBA1 - deviant')
            norm_dev_diff_mat[0].show()
            norm_dev_diff_mat[0].savefig(ephys_figdir / f'norm_dev_diff_mat_{sessname}_dev_ABBA1_deviant.pdf')

                # all_dev_plot = plt.subplots(4,5,figsize=(30,20))
                # for i, (trial_response,ax) in enumerate(zip(predictors_dict[dev_type],all_dev_plot[1].flatten())):
                #     norm_dec_diff_mat = plot_psth(trial_response - predictors_dict['normal'].mean(axis=0),
                #                                   'Pattern', new_window, cmap='bwr',plot=(all_dev_plot[0],ax),
                #                                   norm=cmap_norm)
                #     ax.axvline(0.5, c='w', ls='--')
                # all_dev_plot[0].show()

            # decode responses


        # stage5 analysis on representation of rising vs non-rising tones
        if sess_info['sess_order'] == 'main' and 5 in sessions[sessname].td_df['Stage'].values:
            # get event idx
            window = [0, 0.25]
            t4sim = -1
            preds_pips4sim = {event_lbl: get_predictor_from_psth(sessions[sessname], event_lbl, psth_window, window,
                                                                 mean=None,baseline=0.25)
                              for event_lbl in sessions[sessname].sound_event_dict
                              if any(char in event_lbl for char in 'ABCD')}

            similarity = cosine_similarity([pred[:,:,t4sim].mean(axis=0) for pred in preds_pips4sim.values()])
            # similarity = cosine_similarity([pred[:,:,-10:].mean(axis=-1).mean(axis=0) for pred in pip_predictors.values()])

            for grouping in [['group', 'name'], ['position'], ['idx'], ['ptype_i']]:
                similarity_plot = plot_sim_by_grouping(similarity, grouping, pip_desc, 'Reds',
                                                                 im_kwargs=dict(vmin=similarity.min(),vmax=1))
                grouping_name = '_'.join(grouping)
                similarity_plot[1].set_title(f'{sessname} {grouping_name}')
                similarity_plot[0].set_size_inches(15, 13)
                similarity_plot[0].show()
                similarity_plot[0].savefig(ephys_figdir/f'pip_similarity_{sessname}_{grouping_name}.pdf')

            pearson_sim = [[pearsonr(ii[:,:,t4sim].mean(axis=0),jj[:,:,t4sim].mean(axis=0))[0]
                            for jj in preds_pips4sim.values()]
                           for ii in preds_pips4sim.values()]
            pearson_sim = np.array(pearson_sim)
            # compare pearson to cosine
            for grouping in [['group', 'name'], ['position'], ['idx'], ['ptype_i']]:
                sim_comp_plot = plt.subplots(2)
                for sim_i, (sim_type_mat,sim_type) in enumerate(zip([pearson_sim, similarity], ['pearson', 'cosine'])):
                    plot_sim_by_grouping(sim_type_mat, grouping, pip_desc, 'Reds',
                                         plot=(sim_comp_plot[0],sim_comp_plot[1][sim_i]),
                                         im_kwargs=dict(vmin=sim_type_mat.min(),vmax=1),)
                    sim_comp_plot[1][sim_i].set_title(sim_type)
                grouping_name = '_'.join(grouping)
                sim_comp_plot[0].set_size_inches(15, 12*2)
                sim_comp_plot[0].suptitle(f'{sessname} {grouping_name}')
                sim_comp_plot[0].show()
                sim_comp_plot[0].savefig(ephys_figdir/f'pip_sim_pearson_vs_cosine_{sessname}_{grouping_name}.pdf')


            sort_keys = ['name','group',]
            plot_order,plot_names = get_reordered_idx(pip_desc, sort_keys)
            # time series of similarity
            x_ser = np.linspace(window[0], window[1], preds_pips4sim['A-0'].shape[-1])
            # similarity_over_time = [cosine_similarity([pred[:,:,t].mean(axis=0) for pred in pip_predictors.values()])
            #                         for t in range(x_ser.shape[0])]
            # similarity_over_time_arr = np.array(similarity_over_time)[:,:,plot_order]
            # sim_over_time_tsplot = plt.subplots(similarity_over_time_arr.shape[2])
            # for i in range(similarity_over_time_arr.shape[2]):
            #     sim_over_time_tsplot[1][i].plot(x_ser,
            #                                     (similarity_over_time_arr[:,0,i]),label=plot_names[i])
            #     sim_over_time_tsplot[1][i].set_xticks([])
            #     box = sim_over_time_tsplot[1][i].get_position()
            #     # sim_over_time_tsplot[1][i].set_position([box.x0, box.y0, box.width * 0.8, box.height])
            #     sim_over_time_tsplot[1][i].legend(loc='upper right',bbox_to_anchor=(1, 0.85))
            # sim_over_time_tsplot[1][-1].set_xticks(x_ser)
            # sim_over_time_tsplot[1][-1].locator_params(axis='x', nbins=6)
            # sim_over_time_tsplot[1][-1].set_xlabel('Time from pip onset (s)')
            # comp_pip = plot_names[0].replace("\n"," ")
            # sim_over_time_tsplot[0].suptitle(f'Similarity to {comp_pip} over time',y=0.9)
            # sim_over_time_tsplot[0].set_size_inches(9, 18)
            # sim_over_time_tsplot[0].show()
            # sim_over_time_tsplot[0].savefig(ephys_figdir/f'pip_similarity_over_time_{sessname}.pdf')

            # time series of similarity
            # pip_positions = [1]
            sim_by_property_plot = plt.subplots(ncols=len('ABCD'),figsize=(32,16),sharey='all')
            for p,pp in enumerate('ABCD'):
                pips_by_property = {}
                for prop in ['idx','ptype']:
                    pips_by_property[prop] = get_list_pips_by_property(pip_desc,prop,[p+1])
                for prop in pips_by_property:
                    pop_rate_mats_to_pip = [[preds_pips4sim[pip] for pip in pips] for pips in pips_by_property[prop]]
                    sim_over_time_arrs = [get_sim_mat_over_time(pop_rate_mats_to_comp)
                                          for pop_rate_mats_to_comp in pop_rate_mats_to_pip]
                    [sim_by_property_plot[1][p].plot(x_ser,sim_over_time_arrs[i].mean(axis=1).mean(1),label=f'{prop} {i}')
                     for i in range(len(sim_over_time_arrs))]

                sim_by_property_plot[1][p].legend()
                sim_by_property_plot[1][p].set_xlabel('Time from pip onset (s)')
                sim_by_property_plot[1][p].set_ylabel('Similarity')
                sim_by_property_plot[1][p].set_title(f'Pip {p+1}')

            sim_by_property_plot[0].show()
            sim_by_property_plot[0].savefig(ephys_figdir/f'pip_similarity_by_property_{sessname}.pdf')

            # similarity for indv pips

            pip_plot_lbls = [f'{"ABCD" if pi % n_patts_per_rule == 0 else "ABBA"} ({pi//n_patts_per_rule})'
                             for pi in range(len(main_patterns))]
            permute = True

            min_sim = np.round(np.quantile(similarity,0.1),1)
            pip_by_pip_sim_plot = plt.subplots(1,ncols=len('ABCD'),squeeze=False)
            preds_shuffled = {}

            # for permute, plots in zip([False],pip_by_pip_sim_plot[1]):
                # for pi, pip2use in enumerate('ABCD'):
                #     single_pip_predictor = by_pip_predictors[pip2use]
                #     if permute:
                #         n_shuffled_splits = [permute_pip_preds(single_pip_predictor) for _ in range(n_shuffles)]
                #         preds_shuffled[pip2use] = n_shuffled_splits
                #         preds2use = [np.mean([e[i] for e in n_shuffled_splits],axis=0)
                #                      for i in tqdm(range(len(single_pip_predictor)), total=len(single_pip_predictor),
                #                                    desc='averaging shuffles')]
                #     else:
                #         preds2use = list(single_pip_predictor.values())
                #     similarity_to_pip = cosine_similarity([pred[:, :, t4sim].mean(axis=0)
                #                                            for pred in preds2use])
                #     reordered_names, reordered_idxs = get_reordered_idx(pip_desc,['ptype_i'],
                #                                                         subset=[p for p in pip_lbls if pip2use in p])
                #     reordered_idxs = [int(idx/4) for idx in reordered_idxs]
                #     pip_plot_lbls = [e.split(' ')[-1] for e in reordered_names]
                #     plot_similarity_mat(similarity_to_pip, pip_plot_lbls,
                #                         reorder_idx=reordered_idxs,
                #                         cmap='Reds', plot=(pip_by_pip_sim_plot[0],plots[pi]),
                #                         im_kwargs=dict(vmin=min_sim,vmax=1),
                #                         plot_cbar=True if pi == len('ABCD')-1 else False)
            plot_sim_by_pip(preds_pips4sim, similarity, pip_by_pip_sim_plot[0], pip_by_pip_sim_plot[1][0],
                            pip_desc, cmap='Reds', im_kwargs=dict(vmin=min_sim,vmax=1))

            [plot.set_ylabel('') for pi, plot in enumerate(pip_by_pip_sim_plot[1].flatten()) if pi%len('ABCD')>0 ]
            # [plot.set_xlabel('') for plot in pip_by_pip_sim_plot[1][-1]]
            pip_by_pip_sim_plot[0].set_size_inches(20,5*pip_by_pip_sim_plot[1].shape[0])
            pip_by_pip_sim_plot[0].set_layout_engine('tight')
            pip_by_pip_sim_plot[0].show()
            sim_figdir = ephys_figdir.with_stem(f'{ephys_figdir.stem}_by_pip_sim_plots')
            if not sim_figdir.is_dir():
                sim_figdir.mkdir()
            pip_by_pip_sim_plot[0].savefig( sim_figdir/ f'by_pip_similarity_{sessname}{"_permute" if permute else ""}_reordered.pdf')

            # regression model of similarity
            event_responses = preds_pips4sim
            # construct features
            features2use = ['idx','position','ptype_i','group']
            # features2use = [features2use[idx] for idx in [2,0,1]]
            event_features = [{f:[pip_desc[e][f]] * event_responses[e].shape[0] for f in features2use}
                              for e in event_responses]
            f_df = pd.DataFrame.from_records(event_features)
            f_arr = np.vstack([np.hstack(f_df[f].to_list()) for f in features2use])
            f_arr_df = pd.DataFrame(f_arr.T,columns=features2use)
            # construct predictors
            # event_predictors = [e.mean(axis=-1) for e in event_responses.values()]
            event_predictors = [e[:,:,-1] for e in event_responses.values()]
            regr = run_regression(np.vstack(event_predictors).mean(axis=1).reshape(-1,1),
                                  f_arr.T,split_kwargs={'test_size':0.2},)
            x_df = pd.DataFrame(np.vstack(event_predictors))

            # model,result = run_glm(x_df, f_arr_df)
            # print(result.summary())
            # r_coef_plot = plot_2d_array_with_subplots(regr.coef_)
            # r_coef_plot[0].show()

            glm_by_units = [run_glm(x_df[col], f_arr_df) for col in x_df.columns]
            for unit_glm in glm_by_units:
                print(unit_glm[1].summary())

            pvals_by_units = pd.concat([unit_glm[1].pvalues for unit_glm in glm_by_units],axis=1).T
            betas_by_units = pd.concat([unit_glm[1].params for unit_glm in glm_by_units],axis=1).T
            glm_pval_plot = plt.subplots(figsize=(12,6))
            [glm_pval_plot[1].scatter(pvals_by_units.index, pvals_by_units[col],c=f'C{ci}',label=col,s=15)
             for ci,col in enumerate(pvals_by_units.columns) if col != 'const']

            sig_thresh = 0.05
            glm_pval_plot[1].axhline(sig_thresh,ls='--',c='k')
            glm_pval_plot[1].set_xlabel('unit')
            glm_pval_plot[1].set_yscale('log')
            glm_pval_plot[1].set_ylabel('pval')
            glm_pval_plot[1].legend(loc='lower right',ncols=len(pvals_by_units.columns))
            glm_pval_plot[0].show()
            glm_pval_plot[0].savefig(ephys_figdir/ f'glm_pvals_{sessname}.pdf')

            sig_regr_by_type = {regr:(pvals_by_units.query(f'{regr} < {sig_thresh}').index)
                                for regr in pvals_by_units}

            sig_units= np.unique(np.hstack([sig_regr_by_type[regr].to_list()
                                            for regr in features2use]))
            # sig_units= np.unique(np.hstack([sig_regr_by_type[regr].to_list() for regr in features2use]))
            similarity_sig_units_only = cosine_similarity([pred[:,sig_units,t4sim].mean(axis=0)
                                                           for pred in preds_pips4sim.values()])
            # v_zeroed_normed = TwoSlopeNorm(vmin=similarity_sig_units_only.min(), vmax=similarity_sig_units_only.max(), vcenter=0)
            # plot similarity for sig units
            for grouping in [['group','name'],['position'],['idx'],['ptype_i']]:
                similarity_sig_units_plot = plot_sim_by_grouping(similarity_sig_units_only,grouping,pip_desc,'bwr',
                                                                 # im_kwargs=dict(norm=v_zeroed_normed)
                                                                 )
                grouping_name = '_'.join(grouping)
                similarity_sig_units_plot[1].set_title(f'{sessname} {grouping_name}')
                similarity_sig_units_plot[0].set_size_inches(15, 13)
                similarity_sig_units_plot[0].show()
                similarity_sig_units_plot[0].savefig(ephys_figdir/f'pip_similarity_sig_units_only_{sessname}_{grouping_name}.pdf')

            # regress out regressor
            regr2remove = 'position'
            # glm_by_units_regr = [run_glm(x_df[col], f_arr_df[regr2remove]) for col in x_df.columns]
            ntrials_by_event = np.cumsum([e.shape[0] for e in event_responses.values()])
            # regr_contrib_by_units = [glm[0].endog*glm[1].params[regr2remove] for glm in glm_by_units]
            residuals_by_units_by_event = [np.split(glm[1].fittedvalues-glm[0].endog*glm[1].params[regr2remove],
                                                    ntrials_by_event)[:-1]
                                           # for glm in glm_by_units_regr]
                                           for glm in glm_by_units]
            residuals_by_event = [np.array(
                [residuals_by_units_by_event[j][i] for j in range(len(residuals_by_units_by_event))]).T
                                  for i,ii in enumerate(preds_pips4sim)]

            event_responses_regressed = [resp-res for resp,res in zip(event_predictors,residuals_by_event)]
            similarity_regressed = cosine_similarity([resp.mean(axis=0)
                                                      for resp in event_responses_regressed])
            # v_zeroed_normed = TwoSlopeNorm(vmin=similarity_regressed.min(), vmax=similarity_regressed.max(), vcenter=0)
            v_zeroed_normed = TwoSlopeNorm(vmin=-0.5, vmax=1, vcenter=0)
            # plot regressed similarity
            for grouping in [['group','name'],['position'],['idx'],['ptype_i']]:
                grouping_name = '_'.join(grouping)
                similarity_regressed_plot = plot_sim_by_grouping(similarity_regressed, grouping,pip_desc, cmap='bwr',
                                                                 im_kwargs=dict(norm=v_zeroed_normed))
                similarity_regressed_plot[0].suptitle(f'{sessname} {grouping}')
                similarity_regressed_plot[0].set_size_inches(15, 13)
                similarity_regressed_plot[0].show()

                similarity_regressed_plot[0].savefig(ephys_figdir/f'pip_similarity_regressed_{sessname}_{grouping_name}.pdf')
            # plot by pip for ptype
            pip_by_pip_sim_plot = plt.subplots(1,ncols=len('ABCD'),squeeze=False)
            for permute, plots in zip([False],pip_by_pip_sim_plot[1]):
                plot_sim_by_pip(preds_pips4sim, similarity_regressed, pip_by_pip_sim_plot[0], pip_by_pip_sim_plot[1][0],
                                pip_desc, im_kwargs=dict(norm=v_zeroed_normed))

            pip_by_pip_sim_plot[0].set_size_inches(20,5*pip_by_pip_sim_plot[1].shape[0])
            pip_by_pip_sim_plot[0].show()
            pip_by_pip_sim_plot[0].savefig(ephys_figdir/f'pip_similarity_by_pip_regressed_{sessname}.pdf')

            # synth based on copies
            # plot psth by group
            events_responses_as_df = gen_response_df(preds_pips4sim, pip_desc, sessions[sessname].spike_obj.units)
            events_responses_as_df.columns = x_ser
            groupings = ['idx','position','ptype_i','group']
            # groupings = ['ptype_i']
            # psth_by_grouping_plot = plt.subplots(ncols=len(groupings))
            for grouping in groupings:
                if isinstance(grouping,str):
                    grouping = [grouping]
                grouped_responses = events_responses_as_df.groupby(level=grouping)
                grouped_psths_plot = plt.subplots(1,len(grouped_responses))
                group_psth_ts_plot = plt.subplots(ncols=4,sharey=True)
                resp_range = [grouped_responses.min().values.min(),grouped_responses.max().values.max()]
                # pip_i = 2
                for gi,(grp, ax) in enumerate(zip(grouped_responses,grouped_psths_plot[1])):
                    group_psth_mat = grp[1].groupby(level='units').mean()  # xs(pip_i,level='position').
                    # if 'ptype_i' in grouping:
                    #     group_psth_mat = grp[1].xs(pip_i,level='position').groupby(level='units').mean()
                    plot_psth(group_psth_mat,grp[0],window,plot=(grouped_psths_plot[0],ax),
                              vmin=resp_range[0],vmax=resp_range[1])
                    [group_psth_ts_plot[1][pip_i].plot(x_ser,pip_group[1].mean(axis=0),
                                                       label=f'pip {pip_group[0]}: {grp[0]}')
                     for pip_i, pip_group in enumerate(grp[1].groupby(level='position'))]
                    # group_sem = sem(group_psth_mat)
                    # group_psth_ts_plot[1].fill_between(x_ser,group_psth_mat.mean(axis=0)-group_sem,
                    #                                    group_psth_mat.mean(axis=0)+group_sem,
                    #                                    alpha=0.1)

                grouped_psths_plot[0].suptitle(f'{sessname} {grouping}')
                grouped_psths_plot[0].set_size_inches(5*len(grouped_psths_plot[1]), 5)
                grouped_psths_plot[0].set_layout_engine('tight')
                # grouped_psths_plot[0].show()
                grouped_psths_plot[0].savefig(ephys_figdir/f'grouped_psths_{sessname}_{grouping}.pdf')

                # group_psth_ts_plot[1].set_title(f'{sessname} {grouping}')
                # group_psth_ts_plot[1].set_xlabel('time')
                group_psth_ts_plot[1][0].set_ylabel('Hz')
                group_psth_ts_plot[1][-1].legend()
                group_psth_ts_plot[0].set_size_inches(6*len('ABCD'), 5)
                group_psth_ts_plot[0].set_layout_engine('tight')
                group_psth_ts_plot[0].show()
                group_psth_ts_plot[0].savefig(ephys_figdir/f'group_psth_ts_{sessname}_{grouping}.pdf')


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
            prediction_ts_plot[0].savefig(ephys_figdir/f'decoding_acc_ts_no_base_{sessname}.pdf')

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

        sessions[sessname].pickle_obj(pkl_dir)
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

                    cross_sess_preds_tsplot[0].savefig(ephys_figdir/f'cross_sess_preds_{sess_decoder2use}_on_{sessname}_for_{pip}.pdf')




