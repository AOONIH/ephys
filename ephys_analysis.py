import numpy as np

from ephys_analysis_funcs import *
import platform
import argparse
import yaml
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.signal import savgol_filter
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('dir2use')
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
    spike_dir = get_spikedir(recording_dir)
    spike_cluster_path = r'spike_clusters.npy'
    spike_times_path = r'spike_times.npy'

    # spike_dir = r'X:\Dammy\ephys\DO79_2024-01-17_15-20-19_001\sorting\kilosort2_5\sorter_output'
    # spike_dir = r'X:\Dammy\ephys\DO79_2024-01-25_16-07-42_001\sorting\kilosort2_5\sorter_output'
    # recording_dir = Path(r'X:\Dammy\ephys\DO79_2024-01-17_15-20-19_001\Record Node 101\experiment1\recording1')

    with open(recording_dir / 'metadata.json', 'r') as jsonfile:
        recording_meta = json.load(jsonfile)
    start_time = recording_meta['trigger_time']
    sessname = session_topology[session_topology['ephys_dir'] == win_rec_dir]['sound_bin_stem'].values[0]
    sessname = sessname.replace('_SoundData','')
    sessions = {sessname: Session(sessname,ceph_dir )}
    psth_window = [-2, 3]

    if not sessions[sessname].sound_event_dict:
        sessions[sessname].init_spike_obj(spike_times_path, spike_cluster_path, start_time, parent_dir=spike_dir)
        _parts = sessname.split('_')
        '_'.join([_parts[0],'SoundData',_parts[1]])
        labels = ['A', 'B', 'C', 'D', 'X', 'base']
        sessions[sessname].init_sound_event_dict('_'.join([_parts[0],'SoundData',_parts[1]]),labels,
                                                 harpbin_dir=ceph_dir/'Dammy'/'harpbins')
        sessions[sessname].get_sound_psth(psth_window=psth_window,zscore_flag=True,baseline_dur=0,redo_psth=True)
        sessions[sessname].save_psth(figdir=ceph_dir/'Dammy'/'figures'/'ephys')

    window = [0, 0.25]
    if 'X' in list(sessions[sessname].sound_event_dict.keys()):
        _predictor_list = [get_predictor_from_psth(sessions[sessname], key, psth_window, window)
                           for key in ['X', 'base']]
        _feature_list = [np.full(mat.shape[0], i) for i, mat in enumerate(_predictor_list)]

        sessions[sessname].init_decoder('X_to_base', np.vstack(_predictor_list), np.hstack(_feature_list))
        sessions[sessname].run_decoder('X_to_base', ['data','shuffle'], dec_kwargs={'cv_folds':0},
                                       plot_flag=True)
        sessions[sessname].decoders['X_to_base'].accuracy_plot[0].show()

    for pip in 'ABCD':
        _predictor_list = [get_predictor_from_psth(sessions[sessname], key, psth_window, window) for key in [pip, 'base']]
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
    tone2base_all_plot[0].savefig(ceph_dir/'Dammy'/'figures'/'ephys'/f'ABCD_to_base_perf_{sessname}.svg')

    _predictor_list = [get_predictor_from_psth(sessions[sessname], key, psth_window, window) for key in ['A','B','C','D','base']]
    _feature_list = [np.full(mat.shape[0], i) for i, mat in enumerate(_predictor_list)]

    dec_kwargs = {'cv_folds':0}
    all_dec_name = 'all_vs_all_no_base'
    all_dec_lbls = ['A', 'B', 'C', 'D', 'base'][:-1]
    sessions[sessname].init_decoder(all_dec_name, np.vstack(_predictor_list[:-1]), np.hstack(_feature_list[:-1]))
    sessions[sessname].run_decoder(all_dec_name, ['data', 'shuffle'], dec_kwargs=dec_kwargs)

    sessions[sessname].decoders[all_dec_name].plot_confusion_matrix(all_dec_lbls,include_values=False)
    # sessions[sessname].decoders['all_vs_all'].cm_plot[0].set_constrained_layout('constrained')
    sessions[sessname].decoders[all_dec_name].cm_plot[0].savefig(ceph_dir/'Dammy'/'figures'/'ephys'/
                                                                 f'ABCD_cm_no_base_{sessname}.svg')

    sessions[sessname].decoders[all_dec_name].plot_decoder_accuracy( all_dec_lbls,)
    sessions[sessname].decoders[all_dec_name].accuracy_plot[0].savefig(ceph_dir/'Dammy'/'figures'/'ephys'/
                                                                 f'all_vs_all_accuracy_no_base_{sessname}.svg')

    pearson_corr_all = np.zeros((len(_predictor_list),len(_predictor_list)))
    for i,ii in enumerate(_predictor_list):
        midpoint = int(ii.shape[0]/2)
        for j, jj in enumerate(_predictor_list):
            pearson_corr_all[i,j] = pearsonr(np.array_split(ii,2)[0].mean(axis=0),
                                             np.array_split(jj,2)[1].mean(axis=0))[0]

    pearson_plot = plot_2d_array_with_subplots(pearson_corr_all,cbar_height=20)
    pearson_plot[1].invert_yaxis()
    # pearson_plot[1].set_xticklabels(['','A','B','C','D','base'])
    pearson_plot[1].set_xticklabels(['','A','B','C','D',])
    pearson_plot[1].set_xlabel('second half')
    # pearson_plot[1].set_yticklabels(['','A','B','C','D','base'])
    pearson_plot[1].set_yticklabels(['','A','B','C','D'])
    pearson_plot[1].set_ylabel('first half')
    pearson_plot[2].ax.set_ylabel("Pearson's correlation",rotation=270,labelpad=12)
    pearson_plot[0].show()
    pearson_plot[0].savefig(ceph_dir/'Dammy'/'figures'/'ephys'/f'pearson_no_base_corr_matrix_{sessname}.svg',)

    for n in range(100):

        _feature_list = [np.full(mat.shape[0], i) for i, mat in enumerate(_predictor_list)]

    home_dir = config[f'home_dir_{sys_os}']
    sess_td_path = session_topology[session_topology['ephys_dir'] == win_rec_dir]['trialdata_path'].values[0]
    if 'passive' not in sess_td_path:
        sessions[sessname].load_trial_data(f'{sess_td_path}.csv',home_dir,r'H:\data\Dammy\DO79\TrialData')
        idx_bool = sessions[sessname].td_df['n_since_last'] <= 3
        recent_pattern_trials = sessions[sessname].td_df[idx_bool].index.to_numpy()
        distant_pattern_trials = sessions[sessname].td_df[~idx_bool].index.to_numpy()

        rec_dist_decoder_plot = plt.subplots()
        dec_events = ['A','B','C','D','A_shuffle','A_badpred']
        for pi,pip in enumerate(dec_events):
            pip_id = pip.split('_')[0]

            pip_predictor = get_predictor_from_psth(sessions[sessname], pip_id, psth_window, [0, 0.25])
            recent_idx_bool = np.isin(sessions[sessname].sound_event_dict[pip_id].trial_nums-1,recent_pattern_trials)
            if 'badpred' in pip:
                rec_dist_predictors = pip_predictor[recent_idx_bool], pip_predictor[recent_idx_bool]
            else:
                rec_dist_predictors = pip_predictor[recent_idx_bool], pip_predictor[~recent_idx_bool]
            rec_dist_features = [np.full(e.shape[0], ei) for ei, e in enumerate(rec_dist_predictors)]

            dec_kwargs = {'cv_folds': 0}
            plt_kwargs = {}
            if 'shuffle' in pip:
                # print('shuffle')
                dec_kwargs['shuffle'] = True
                plt_kwargs['c'] = 'k'

            sessions[sessname].init_decoder(f'rec_dist_{pip}', np.vstack(rec_dist_predictors),
                                            np.hstack(rec_dist_features))
            sessions[sessname].run_decoder(f'rec_dist_{pip}', ['data', 'shuffle'], dec_kwargs=dec_kwargs)
            plot_decoder_accuracy(sessions[sessname].decoders[f'rec_dist_{pip}'].fold_accuracy, [pip],
                                  ax=rec_dist_decoder_plot[1], start_loc=pi,plt_kwargs=plt_kwargs)
        rec_dist_decoder_plot[1].set_xticks(np.arange(len(dec_events)))
        rec_dist_decoder_plot[1].set_xticklabels(dec_events)
        rec_dist_decoder_plot[1].get_legend().remove()
        rec_dist_decoder_plot[0].set_constrained_layout('constrained')
        rec_dist_decoder_plot[0].savefig(ceph_dir/'Dammy'/'figures'/'ephys'/f'recent_vs_distant_{sessname}.svg')

        # run all decoder with base for tseries
        all_dec_lbls = ['A', 'B', 'C', 'D', 'base']
        all_dec_name = 'all_vs_all'

        sessions[sessname].init_decoder(all_dec_name, np.vstack(_predictor_list), np.hstack(_feature_list))
        sessions[sessname].run_decoder(all_dec_name, ['data', 'shuffle'], dec_kwargs=dec_kwargs)

        sessions[sessname].decoders[all_dec_name].plot_confusion_matrix(all_dec_lbls, include_values=False)
        # sessions[sessname].decoders['all_vs_all'].cm_plot[0].set_constrained_layout('constrained')
        sessions[sessname].decoders[all_dec_name].cm_plot[0].savefig(ceph_dir / 'Dammy' / 'figures' / 'ephys' /
                                                                     f'ABCD_base_cm_{sessname}.svg')

        sessions[sessname].decoders[all_dec_name].plot_decoder_accuracy(all_dec_lbls, )
        # sessions[sessname].decoders[all_dec_name].accuracy_plot[0].savefig(ceph_dir / 'Dammy' / 'figures' / 'ephys' /
        #                                                                    f'all_vs_all_accuracy{sessname}.svg')


    # decoding accuracy over time
    psth_window = psth_window
    window = [-1,2]
    predictions_ts_array = sessions[sessname].map_preds2sound_ts('A',['all_vs_all'],
                                                                 psth_window,window,)
    predictions_ts_array = np.squeeze(predictions_ts_array)
    x_ser = np.linspace(window[0],window[1],predictions_ts_array.shape[2])
    # x_ser = np.linspace(0,1,predictions_ts_array.shape[2])
    prediction_ts_plot = plt.subplots(sharex='all',sharey='all')
    for li, lbl in enumerate('ABCD'):
        lbl_pred = predictions_ts_array==li
        prediction_ts_plot[1].plot(x_ser,savgol_filter(lbl_pred.mean(axis=0).mean(axis=0),51,2),c=f'C{li}', label=lbl)
        # prediction_ts_plot[1].plot(x_ser,lbl_pred.mean(axis=0).mean(axis=0), label=lbl)
        # prediction_ts_plot[1][li].axis('off')
        # for dec_i, (dec_preds, dec_name) in enumerate(zip(predictions_ts_array,list('ABCD'))):
    [prediction_ts_plot[1].axvspan(t,t+0.15,fc='k',alpha=0.1) for t in np.arange(0,1,.25)]
    prediction_ts_plot[1].set_ylabel('prediction rate')
    prediction_ts_plot[1].set_ylim(0,.8)
    prediction_ts_plot[1].set_xlabel('Time from A onset (s)')
    prediction_ts_plot[1].set_ylabel('prediction rate')
    prediction_ts_plot[1].legend(loc=1)
    prediction_ts_plot[0].set_constrained_layout('constrained')
    prediction_ts_plot[0].show()
    prediction_ts_plot[0].savefig(ceph_dir/'Dammy'/'figures'/'ephys'/f'decoding_acc_ts_no_base_{sessname}.svg')



    sessions[sessname].pickle_obj(ceph_dir/'Dammy'/'ephys_pkls')



