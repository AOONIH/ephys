import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from ephys_analysis_funcs import *
import platform
import argparse
import yaml
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.signal import savgol_filter, correlate2d
import ruptures as rpt
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('dir2use')
    parser.add_argument('--sorter_dirname',default='kilosort2_5',required=False)
    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])

    gen_metadata(ceph_dir/posix_from_win(r'X:\Dammy\ephys\session_topology.csv'),ceph_dir,ceph_dir/'Dammy'/'harpbins')
    session_topology = pd.read_csv(ceph_dir/posix_from_win(r'X:\Dammy\ephys\session_topology.csv'))
    # win_rec_dir = r'X:\Dammy\ephys\DO79_2024-01-17_15-20-19_001\Record Node 101\experiment1\recording1'
    if args.dir2use.isnumeric():
        win_rec_dir = session_topology.iloc[int(args.dir2use)]['ephys_dir']
    else:
        win_rec_dir = args.dir2use
    print(f'analysing {win_rec_dir}')

    recording_dir = ceph_dir/posix_from_win(win_rec_dir)
    sorter_dirname = args.sorter_dirname
    sorting_dirname = f'sorting{"_concat" if "concat" in sorter_dirname else ""}'
    spike_dir = get_spikedir(recording_dir,sorter_dirname,sorting_dir_name=sorting_dirname)
    spike_cluster_path = r'spike_clusters.npy'
    spike_times_path = r'spike_times.npy'

    ephys_figdir = ceph_dir/'Dammy'/'figures'/'test_concat_sorting'
    if not ephys_figdir.is_dir():
        ephys_figdir.mkdir()

    with open(recording_dir / 'metadata.json', 'r') as jsonfile:
        recording_meta = json.load(jsonfile)
    start_time = recording_meta['trigger_time']
    sessname = session_topology[session_topology['ephys_dir'] == win_rec_dir]['sound_bin_stem'].values[0]
    sessname = sessname.replace('_SoundData','')
    sessions = {sessname: Session(sessname,ceph_dir )}
    psth_window = [-2, 3]

    home_dir = config[f'home_dir_{sys_os}']
    sess_td_path = session_topology[session_topology['ephys_dir'] == win_rec_dir]['trialdata_path'].values[0]

    if 'passive' not in sess_td_path:
        sessions[sessname].load_trial_data(f'{sess_td_path}.csv',home_dir,
                                           rf'H:\data\Dammy\{sessname.split("_")[0]}\TrialData')
        normal = sessions[sessname].td_df[sessions[sessname].td_df['Tone_Position'] == 0]['PatternID'].iloc[0]
        normal = [int(pip) for pip in normal.split(';')]
        if -1 in sessions[sessname].td_df['Pattern_Type'].unique():
            new_normal = sessions[sessname].td_df[sessions[sessname].td_df['Pattern_Type'] == -1]['PatternID'].iloc[0]
            new_normal = [int(pip) for pip in new_normal.split(';')]
        else:
            new_normal = None

    else:
        normal = [int(pip) for pip in sess_td_path.split('_')[-1].split(';')]
        new_normal = None
    if not sessions[sessname].sound_event_dict:
        sessions[sessname].init_spike_obj(spike_times_path, spike_cluster_path, start_time, parent_dir=spike_dir)
        _parts = sessname.split('_')
        '_'.join([_parts[0],'SoundData',_parts[1]])
        labels = ['A', 'B', 'C', 'D', 'X', 'base','newA']

        sessions[sessname].init_sound_event_dict('_'.join([_parts[0],'SoundData',_parts[1]]),labels,
                                                 harpbin_dir=ceph_dir/'Dammy'/'harpbins',normal=normal,
                                                 new_normal=new_normal)
        sessions[sessname].get_sound_psth(psth_window=psth_window,zscore_flag=True,baseline_dur=0,redo_psth=True,
                                          use_iti_zscore=True)
        # sessions[sessname].reorder_psth(labels[0],['B', 'C', 'D'])
        sessions[sessname].get_sound_psth(psth_window=psth_window,use_iti_zscore=True, redo_psth_plot=True,)

        [sessions[sessname].sound_event_dict['A'].psth_plot[1].axvline(t, c='white', ls='--') for t in
         np.arange(0, 1, 0.25) if 'passive' not in sess_td_path]

        psth_ts_plot = plt.subplots()
        psth_ts_plot[1].plot(sessions[sessname].sound_event_dict['A'].psth[1].columns.to_series().dt.total_seconds(),
                             sessions[sessname].sound_event_dict['A'].psth[1].mean(axis=0),
                             c='k',lw=3)
        # psth_ts_plot[1].set_ylim(-0.17,0.04)
        psth_ts_plot[1].set(frame_on=False)
        psth_ts_plot[1].set_xticklabels([])
        psth_ts_plot[1].set_yticklabels(psth_ts_plot[1].get_yticklabels())
        psth_ts_plot[0].set_size_inches(6.4,1)
        psth_ts_plot[1].axvline(0,c='k',ls='--')
        psth_ts_plot[0].savefig(ephys_figdir/f'A_psth_ts_{sessname}.svg')

        plot_2d_array_with_subplots(sessions[sessname].sound_event_dict['D'].psth[1].loc[sessions[sessname].sound_event_dict['A'].psth[2]])
        sessions[sessname].save_psth(figdir=ephys_figdir)

        # plot all on 1
        # X and A
        if 'X' in list(sessions[sessname].sound_event_dict.keys()):
            psth_XA_plot = plt.subplots(ncols=2,figsize=(4.5,3.5),sharey='all')
            for ei, e in enumerate(['X','A']):
                psth_mat = get_predictor_from_psth(sessions[sessname], e, psth_window,[-0.5,1],mean=np.mean,mean_axis=0)
                plot_psth(psth_mat,f'Time from {e} onset', [-0.5,1],vmin=-1,vmax=4.5,plot_cbar=(True if ei==1 else False),
                          plot=(psth_XA_plot[0],psth_XA_plot[1][ei]))
                if e =='A':
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

        if 'passive' in sess_td_path:
            psth_ABCD_plot = plt.subplots(ncols=4,figsize=(9,3.5),sharey='all')
            for ei, e in enumerate('ABCD'):
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

    for pip in 'ABCD':
        _predictor_list = [get_predictor_from_psth(sessions[sessname], key, psth_window, window,mean=np.mean)
                           for key in [pip, 'base']]
        subset_base_preds = _predictor_list[-1][np.random.choice(_predictor_list[-1].shape[0],
                                                                 min(max([e.shape[0] for e in _predictor_list[:-1]]),
                                                                     _predictor_list[-1].shape[0]),
                                                                 replace=False)]
        # _predictor_list[-1] = subset_base_preds
        _feature_list = [np.full(mat.shape[0], i) for i, mat in enumerate(_predictor_list)]

        sessions[sessname].init_decoder(f'{pip}_to_base', np.vstack(_predictor_list), np.hstack(_feature_list))
        sessions[sessname].run_decoder(f'{pip}_to_base', ['data','shuffle'], dec_kwargs={'cv_folds':0})
        # sessions[sessname].decoders[f'{pip}_to_base'].accuracy_plot[0].set_constrained_layout('constrained')
        # sessions[sessname].decoders[f'{pip}_to_base'].accuracy_plot[0].savefig(ceph_dir/'Dammy'/'figures'/'ephys'/f'{pip}_to_base_accr.svg',)

    tone2base_all_plot = plt.subplots()
    for pi, pip in enumerate('ABCD'):
        metric2plot = sessions[sessname].decoders[f'{pip}_to_base'].fold_accuracy
        plot_decoder_accuracy(metric2plot, pip, fig=tone2base_all_plot[0], ax=tone2base_all_plot[1],
                              start_loc=pi, n_features=2)
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

    tone2base_all_plot[1].set_xticks(np.arange(len('ABCD')+1))
    tone2base_all_plot[1].set_xticklabels(list('ABCD')+['A_shuffle'])
    tone2base_all_plot[1].get_legend().remove()
    tone2base_all_plot[0].show()
    tone2base_all_plot[0].set_constrained_layout('constrained')
    tone2base_all_plot[0].savefig(ephys_figdir/f'ABCD_to_base_perf_{sessname}.svg')

    _predictor_list = [get_predictor_from_psth(sessions[sessname], key, psth_window, window, mean=np.mean)
                       for key in ['A','B','C','D','base']]
    _feature_list = [np.full(mat.shape[0], i) for i, mat in enumerate(_predictor_list)]

    dec_kwargs = {'cv_folds':0}
    all_dec_name = 'all_vs_all_no_base'
    all_dec_lbls = ['A', 'B', 'C', 'D', 'base']
    sessions[sessname].init_decoder(all_dec_name, np.vstack(_predictor_list), np.hstack(_feature_list))
    sessions[sessname].run_decoder(all_dec_name, ['data', 'shuffle'], dec_kwargs=dec_kwargs)

    sessions[sessname].decoders[all_dec_name].plot_confusion_matrix(all_dec_lbls,include_values=False,cmap='PiYG',)
    sessions[sessname].decoders[all_dec_name].cm_plot[1].tick_params(axis='both', which='major', labelsize=14)
    sessions[sessname].decoders[all_dec_name].cm_plot[1].tick_params(axis='y', which='major', labelrotation=90)
    # sessions[sessname].decoders[all_dec_name].cm_plot[1].set_xticklabels(['A', 'B', 'C', 'D'],fontsize=14)
    # sessions[sessname].decoders[all_dec_name].cm_plot[1].set_yticklabels(['A', 'B', 'C', 'D'],fontsize=14)
    # sessions[sessname].decoders[all_dec_name].cm_plot[1].set_yticklabels(['A', 'B', 'C', 'D'],fontsize=14)
    sessions[sessname].decoders[all_dec_name].cm_plot[1].set_xlabel('Predicted label',fontsize=14)
    sessions[sessname].decoders[all_dec_name].cm_plot[1].set_ylabel('True label',fontsize=14)
    sessions[sessname].decoders[all_dec_name].cm_plot[0].set_size_inches(3.3,2.33)
    # sessions[sessname].decoders[all_dec_name].cm_plot[0].set_constrained_layout('tight')
    sessions[sessname].decoders[all_dec_name].cm_plot[0].savefig(ephys_figdir/
                                                                 f'ABCD_cm_no_base_{sessname}.svg')

    sessions[sessname].decoders[all_dec_name].plot_decoder_accuracy( all_dec_lbls,)
    sessions[sessname].decoders[all_dec_name].accuracy_plot[0].savefig(ephys_figdir/
                                                                 f'all_vs_all_accuracy_no_base_{sessname}.svg')

    pearson_corr_all = np.zeros((len(_predictor_list),len(_predictor_list)))
    for i,ii in enumerate(_predictor_list):
        midpoint = int(ii.shape[0]/2)
        for j, jj in enumerate(_predictor_list):
            pearson_corr_all[i,j] = pearsonr(np.array_split(ii,2)[0].mean(axis=0),
                                             np.array_split(jj,2)[1].mean(axis=0))[0]

    pearson_plot = plot_2d_array_with_subplots(pearson_corr_all,cbar_height=20)
    pearson_plot[1].invert_yaxis()
    pearson_plot[1].set_xticklabels(['','A','B','C','D','base'])
    # pearson_plot[1].set_xticklabels(['','A','B','C','D',])
    pearson_plot[1].set_xlabel('second half')
    pearson_plot[1].set_yticklabels(['','A','B','C','D','base'])
    # pearson_plot[1].set_yticklabels(['','A','B','C','D'])
    pearson_plot[1].set_ylabel('first half')
    pearson_plot[2].ax.set_ylabel("Pearson's correlation",rotation=270,labelpad=12)
    pearson_plot[0].show()
    pearson_plot[0].savefig(ephys_figdir/f'pearson_no_base_corr_matrix_{sessname}.svg',)

    # run all decoder with base for tseries
    all_dec_lbls = ['A', 'B', 'C', 'D', 'base']
    all_dec_name = 'all_vs_all'

    sessions[sessname].init_decoder(all_dec_name, np.vstack(_predictor_list), np.hstack(_feature_list))
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

    home_dir = config[f'home_dir_{sys_os}']
    sess_td_path = session_topology[session_topology['ephys_dir'] == win_rec_dir]['trialdata_path'].values[0]
    if 'passive' not in sess_td_path:
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
            # dec_events = ['A','B','C','D','A_shuffle','A_halves']
            dec_events = ['A','A_shuffle']
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

                dec_kwargs = {'cv_folds': 10}
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
            recent_idx_bool = np.isin(sessions[sessname].sound_event_dict['A'].trial_nums - 1, recent_pattern_trials)
            distant_idx_bool = np.isin(sessions[sessname].sound_event_dict['A'].trial_nums - 1, distant_pattern_trials)
            A_psth = get_predictor_from_psth(sessions[sessname], 'A', psth_window, [-0, .25], mean=None)
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
            # pearson_plot[1].set_xticklabels(['','A','B','C','D',])
            # pearson_plot[1].set_yticklabels(['','A','B','C','D'])
            pearson_plot[2].ax.set_ylabel("Pearson's correlation", rotation=270, labelpad=12)
            pearson_plot[1].tick_params(axis='both', which='major', labelsize=18)
            pearson_plot[2].ax.tick_params(axis='y', which='major', labelsize=14)
            pearson_plot[0].set_size_inches(3.5,3)
            pearson_plot[0].show()
            pearson_plot[0].savefig(ephys_figdir / f'pearson_tofirst_A_{sessname}.svg', )

    if 'passive' not in sess_td_path:
        if 4 in sessions[sessname].td_df['Stage'].values:
            new_window = [-1, 2]
            pip_predictor = get_predictor_from_psth(sessions[sessname], 'A', psth_window, new_window, mean=None)

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
                idx_bool = np.isin(sessions[sessname].sound_event_dict['A'].trial_nums - 1,
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

    # decoding accuracy over time
    psth_window = psth_window
    window = [-1,2]
    predictions_ts_array = sessions[sessname].map_preds2sound_ts('A',['all_vs_all'],
                                                                 psth_window,window,)
    predictions_ts_array = np.squeeze(predictions_ts_array)
    x_ser = np.linspace(window[0],window[1],predictions_ts_array.shape[2])
    # x_ser = np.linspace(0,1,predictions_ts_array.shape[2])

    prediction_ts_plot = plt.subplots(sharex='all',sharey='all')
    cmap = plt.get_cmap('PuBuGn')
    # colors = list(cmap(np.linspace(0, 1, len(all_dec_lbls))+0.25))
    colors =['k','slategray','midnightblue','darkslategray']
    for li, lbl in enumerate('ABCD'):
        lbl_pred = predictions_ts_array==li
        prediction_ts_plot[1].plot(x_ser,savgol_filter(lbl_pred.mean(axis=0).mean(axis=0),51,2),c=colors[li], label=lbl)


        # prediction_ts_plot[1].plot(x_ser,lbl_pred.mean(axis=0).mean(axis=0), label=lbl)
        # prediction_ts_plot[1][li].axis('off')
        # for dec_i, (dec_preds, dec_name) in enumerate(zip(predictions_ts_array,list('ABCD'))):
    [prediction_ts_plot[1].axvspan(t,t+0.15,fc='k',alpha=0.1) for t in np.arange(0,1,.25)]
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
    # for li,lbl in enumerate('ABCD'):
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



    sessions[sessname].pickle_obj(ceph_dir/'Dammy'/'ephys_pkls')
