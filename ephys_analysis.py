import numpy as np

from ephys_analysis_funcs import *
import platform
import argparse
import yaml
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])

    gen_metadata(ceph_dir/posix_from_win(r'X:\Dammy\ephys\session_topology.csv'),ceph_dir,ceph_dir/'Dammy'/'harpbins')
    win_rec_dir = r'X:\Dammy\ephys\DO79_2024-01-17_15-20-19_001\Record Node 101\experiment1\recording1'
    recording_dir = ceph_dir/posix_from_win(win_rec_dir)
    spike_dir = get_spikedir(recording_dir)
    spike_cluster_path = r'spike_clusters.npy'
    spike_times_path = r'spike_times.npy'

    # spike_dir = r'X:\Dammy\ephys\DO79_2024-01-17_15-20-19_001\sorting\kilosort2_5\sorter_output'
    # spike_dir = r'X:\Dammy\ephys\DO79_2024-01-25_16-07-42_001\sorting\kilosort2_5\sorter_output'
    # recording_dir = Path(r'X:\Dammy\ephys\DO79_2024-01-17_15-20-19_001\Record Node 101\experiment1\recording1')

    session_topology = pd.read_csv(ceph_dir/posix_from_win(r'X:\Dammy\ephys\session_topology.csv'))
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
        sessions[sessname].pickle_obj(ceph_dir/'Dammy'/'ephys_pkls')

    window = [0, 0.25]
    _predictor_list = [get_predictor_from_psth(sessions[sessname], key, psth_window, window) for key in ['X', 'base']]
    _feature_list = [np.full(mat.shape[0], i) for i, mat in enumerate(_predictor_list)]

    sessions[sessname].init_decoder('X_to_base', np.vstack(_predictor_list), np.hstack(_feature_list))
    sessions[sessname].run_decoder('X_to_base', ['data','shuffle'], n_features=2,cv_folds=10,plot_flag=True)
    sessions[sessname].decoders['X_to_base'].accuracy_plot[0].show()

    for pip in 'ABCD':
        _predictor_list = [get_predictor_from_psth(sessions[sessname], key, psth_window, window) for key in [pip, 'base']]
        _feature_list = [np.full(mat.shape[0], i) for i, mat in enumerate(_predictor_list)]

        sessions[sessname].init_decoder(f'{pip}_to_base', np.vstack(_predictor_list), np.hstack(_feature_list))
        sessions[sessname].run_decoder(f'{pip}_to_base', ['data','shuffle'], n_features=2,cv_folds=10)
        # sessions[sessname].decoders[f'{pip}_to_base'].accuracy_plot[0].set_constrained_layout('constrained')
        # sessions[sessname].decoders[f'{pip}_to_base'].accuracy_plot[0].savefig(ceph_dir/'Dammy'/'figures'/'ephys'/f'{pip}_to_base_accr.svg',)

    tone2base_all_plot = plt.subplots()
    for pi, pip in enumerate('ABCD'):
        metric2plot = sessions[sessname].decoders[f'{pip}_to_base'].fold_accuracy
        plot_decoder_accuracy(metric2plot, pip, fig=tone2base_all_plot[0], ax=tone2base_all_plot[1],
                              start_loc=pi, n_features=2)
    tone2base_all_plot[1].set_xticks(np.arange(len('ABCD')))
    tone2base_all_plot[1].set_xticklabels(list('ABCD'))
    tone2base_all_plot[1].get_legend().remove()
    tone2base_all_plot[0].show()

    _predictor_list = [get_predictor_from_psth(sessions[sessname], key, psth_window, window) for key in ['A','B','C','D','base']]
    _feature_list = [np.full(mat.shape[0], i) for i, mat in enumerate(_predictor_list)]

    pearson_corr_all = np.zeros((len(_predictor_list),len(_predictor_list)))
    for i,ii in enumerate(_predictor_list):
        midpoint = int(ii.shape[0]/2)
        for j, jj in enumerate(_predictor_list):
            pearson_corr_all[i,j] = pearsonr(np.array_split(ii,2)[0].mean(axis=0),
                                             np.array_split(jj,2)[1].mean(axis=0))[0]

    pearson_plot = plot_2d_array_with_subplots(pearson_corr_all,cbar_height=20)
    pearson_plot[1].invert_yaxis()
    pearson_plot[1].set_xticklabels(['','A','B','C','D','base'])
    pearson_plot[1].set_xlabel('second half')
    pearson_plot[1].set_yticklabels(['','A','B','C','D','base'])
    pearson_plot[1].set_ylabel('first half')
    pearson_plot[2].ax.set_ylabel("Pearson's correlation",rotation=270,labelpad=12)
    pearson_plot[0].show()
    pearson_plot[0].savefig(ceph_dir/'Dammy'/'figures'/'ephys'/f'pearson_corr_matrix_240125a.svg',)

    for n in range(100):

        _feature_list = [np.full(mat.shape[0], i) for i, mat in enumerate(_predictor_list)]

    home_dir = config[f'home_dir_{sys_os}']
    sess_td_path = session_topology[session_topology['ephys_dir'] == win_rec_dir]['trialdata_path'].values[0]
    if 'passive' not in sess_td_path:
        sessions[sessname].load_trial_data(f'{sess_td_path}.csv',home_dir,r'H:\data\Dammy\DO79\TrialData')



    # td_df = pd.read_csv(r'/nfs/home/live/aonih/data/Dammy/DO79/TrialData/DO79_TrialData_240125a.csv')
    # local_rate = 1-td_df['PatternPresentation_Rate'].rolling(10).mean()
    # pattern_w_local_rate = sessions[sessname].td_df['Tone_Position'] == 0


    #
    # #
    # # sess_spike_obj = SessionSpikes(spike_times_path, spike_cluster_path, start_time, parent_dir=spike_dir)
    # #
    # # sound_writes = load_sound_bin(
    # #     session_topology[session_topology['ephys_dir'] == str(recording_dir)]['sound_bin_stem'].values[0])
    # # # assign sounds to trials
    # # sound_writes['Trial_Number'] = np.full_like(sound_writes.index, -1)
    # # sound_writes_diff = sound_writes['Time_diff'] = sound_writes['Timestamp'].diff()
    # # long_dt = np.squeeze(np.argwhere(sound_writes_diff > 1))
    # # for n, idx in enumerate(long_dt):
    # #     sound_writes.loc[idx:, 'Trial_Number'] = n
    # # sound_writes['Trial_Number'] = sound_writes['Trial_Number'] + 1
    # # base_pip_idx = sound_writes['Payload'].mode()[0]
    # # a_idx = base_pip_idx + 2
    # # tone_space = 5
    # # pattern_idxs = [a_idx+i*tone_space for i in range(4)]
    # # sound_events = pattern_idxs + [3,base_pip_idx]
    # # sound_event_labels = ['A', 'B', 'C', 'D', 'X', 'base']
    # #
    # # # get baseline pip
    # # sound_writes['idx_diff'] = sound_writes['Payload'].diff()
    # # writes_to_use = np.all([sound_writes['Payload']==base_pip_idx,sound_writes['idx_diff']==0,
    # #                         sound_writes['Trial_Number']>10, sound_writes['Time_diff']<0.3],axis=0)
    # # base_pips_times = base_pip_times_same_prev = sound_writes[writes_to_use]['Timestamp']
    # #
    # # # event sound times
    # # if sound_writes[sound_writes['Payload'] == 3].empty:
    # #     sound_events.remove(3), sound_event_labels.remove('X')
    # # all_sound_times = [sound_writes[sound_writes['Payload'] == i]['Timestamp'] for i in sound_events
    # #                    if i != base_pip_idx]
    # # if not base_pips_times.empty:
    # #     all_sound_times.append(np.random.choice(base_pips_times,np.max([len(e) for e in all_sound_times]),replace=False))
    # #
    # # # psth analysis:
    # # sound_event_dict = {}
    # # assert len(sound_events) == len(all_sound_times), Warning('sound event and labels must be equal length')
    # # psth_window = [-2,3]
    # # for event, event_times, event_lbl in zip(sound_events,all_sound_times,sound_event_labels):
    # #     sound_event_dict[event_lbl] = SoundEvent(event, event_times, event_lbl)
    # #     sound_event_dict[event_lbl].get_psth(sess_spike_obj, psth_window)
    # # for e in sound_event_dict:
    # #     sound_event_dict[e].psth_plot[0].show()
    # #     sound_event_dict[e].save_plot_as_svg()
    # # # decoder
    # # x_tseries = list(sess_spike_obj.event_spike_matrices.values())[0].columns
    # # sound_tseries = pd.Series(np.zeros_like(x_tseries), index=x_tseries)
    # # pulse_dur = 0.15
    # # pulse_times = np.arange(0, 1, 0.25)
    # # for pulse_t, pulse_id in zip(pulse_times, sound_events):
    # #     sound_tseries.loc[pulse_t:pulse_t + pulse_dur] = pulse_id
    #
    # `# dataset
    # sess_spike_obj.event_spike_matrices = {}  # clear event matrix dict
    # datasets = [get_event_rate_mat(et,e,sess_spike_obj,[0,0.25]) for e,et in tqdm(zip(sound_events,all_sound_times),
    #                                                                               total=len(sound_events),
    #                                                                               desc='generating dataset')]
    # decode_ABCD_none = False
    # predictors_list = [data_matrix.values.reshape([s_times.shape[0],-1,data_matrix.shape[1]]).mean(axis=2)
    #                    for data_matrix,s_times in zip(datasets,all_sound_times)]
    # features_list = [np.full(predictors.shape[0],i) for i, predictors in enumerate(predictors_list)]
    #
    # manual_split = False
    # if decode_ABCD_none:
    #     accuracy = []
    #     n_runs = 500
    #     combos = [[0,4],[0,1],[0,2],[0,3]]
    #     # combos = [[0,4,1],[0,4,2],[0,4,3]]
    #     combo_labels = ['A vs None', 'A vs B', 'A vs C', 'A vs D']
    #     # combo_labels = ['A vs B', 'A vs C', 'A vs D']
    #     for combo in combos:
    #         combo_accuracy = []
    #         for to_shuffle in [False, True]:
    #             predictors = np.vstack([predictors_list[i] for i in combo])
    #             features = np.hstack([features_list[i] for i in combo])
    #             if len(features_list) > 2:
    #                 n = predictors_list[combo[0]].shape[0] + predictors_list[combo[1]].shape[0]
    #                 features[n:] = combo[0]
    #
    #             with multiprocessing.Pool() as pool:
    #                 results = list(tqdm(pool.imap(partial(run_decoder, features=features, shuffle=to_shuffle,
    #                                                       model='svc'),
    #                                               [predictors] * n_runs), total=n_runs))
    #             combo_accuracy.append(list(res[0] for res in results))
    #         accuracy.append(combo_accuracy)
    #
    #     decoder_plot = plt.subplots()
    #     # decoder_labels = ['data', 'shuffle']
    #     decoder_labels = ['data', 'alternate']
    #     x_pad = 1
    #     for ci, (combo, combo_results) in enumerate(zip(combos,accuracy)):
    #         for ri,res in enumerate(combo_results):
    #             if ri == 1:
    #                 col = 'k'
    #                 lbl = None
    #             else:
    #                 col=f'C{ci}'
    #                 lbl = combo_labels[ci]
    #             decoder_plot[1].scatter(simple_beeswarm2(res, width=0.1) + len(decoder_labels)*ci + ri, np.array(res),
    #                                     label=lbl,alpha=0.1,c=col)
    #         ttest = ttest_ind(combo_results[0], combo_results[1], equal_var=False)
    #         # decoder_plot[1].text(0.5+ci*2, np.max(combo_results)+0.08, f'pval = {ttest[1]:.2e}', ha='center')
    #         decoder_plot[1].plot([ci*2, ci*2+1], [np.max(combo_results)+0.04, np.max(combo_results)+0.04], ls='-', c='darkgrey')
    #         if ttest[1] < 0.01:
    #             decoder_plot[1].scatter(0.5+ci*2, np.max(combo_results)+0.08, marker='x', s=30, c='k')
    #
    #     decoder_plot[1].set_ylim(0, 1.19)
    #     # decoder_plot[1].set_xlim(0-0.5,len(accuracy)*len(decoder_labels)+1.5)
    #     # decoder_plot[1].set_xlim(-.25, 1.25)
    #     decoder_plot[1].set_ylabel('decoder accuracy')
    #     decoder_plot[1].axhline(0.5, c='k', ls='--')
    #
    #     decoder_plot[1].set_xticks(np.arange(0, len(accuracy)*len(decoder_labels), 1))
    #     decoder_plot[1].set_xticklabels(np.ravel([decoder_labels]*len(accuracy)))
    #     decoder_plot[1].legend(ncols=len(combo_labels),loc=1)
    #     decoder_plot[1].set_title(f'decoding from population means firing rate: {n_runs} runs')
    #     decoder_plot[0].show()
    #
    # decode_all = False
    # if decode_all:
    #     # decode all
    #     all_decoder_accuracy = []
    #     all_decoder_models = []
    #     all_decoder_predictions = []
    #     n_runs = 100
    #     n_tests = 10
    #     for to_shuffle in [False, True]:
    #         # predictors = np.vstack(predictors_list)
    #         predictors = np.vstack([e[:-n_tests] for e in _predictor_list[:-1]])
    #         features = np.hstack([e[:-n_tests] for e in _feature_list[:-1]])
    #
    #         with multiprocessing.Pool() as pool:
    #             results = list(tqdm(pool.imap(partial(run_decoder, features=features, shuffle=to_shuffle, model='svc',cv_folds=10),
    #                                           [predictors] * n_runs), total=n_runs))
    #         all_decoder_accuracy.append(list(res[2] for res in results))
    #         all_decoder_models.append(list(res[1] for res in results))
    #
    #     decode_all_perf_plot = plot_decoder_accuracy(all_decoder_accuracy,['data', 'shuffle'],n_features=4)
    #     decode_all_perf_plot[0].show()
    #
    #     n_features = len(sound_events)
    #
    #     # model_prediction_array = np.zeros((n_runs,n_features,n_features))  # col is label, row is predicted
    #     model_prediction_array = []
    #     all_labels = ['A', 'B', 'C', 'D',]
    #     for run_i, model_run in enumerate(np.array(all_decoder_models[0]).flatten()):
    #         test_predictors = np.vstack([e[-n_tests:] for e in _predictor_list[:-1]])
    #         test_features = np.hstack([e[-n_tests:] for e in _feature_list[:-1]])
    #         prediction = model_run.predict(test_predictors)
    #         model_prediction_array.append(confusion_matrix(test_features,prediction))
    #         # for feature_i,_ in enumerate(sound_events):
    #         #     model_prediction_array[run_i,:,feature_i] = [(prediction[test_features==feature_i]==predicted_i).sum()
    #         #                                                  for predicted_i,_ in enumerate(sound_events)]
    #     model_prediction_array = np.array(model_prediction_array)
    #     cm_ = ConfusionMatrixDisplay(model_prediction_array.mean(axis=0)/n_tests,display_labels=all_labels)
    #     cm__ = cm_.plot(cmap='bwr',include_values=False)
    #     prediction_comp_plot = cm__.figure_,cm__.ax_
    #     # prediction_comp_plot = plot_2d_array_with_subplots(model_prediction_array.mean(axis=0)/n_tests,extent=[0,4,4,0],
    #     #                                                    aspect=0.01, cbar_height=20,cmap='bwr',
    #     #                                                    vcenter=1/n_features,vmin=0)
    #     prediction_comp_plot[1].invert_yaxis()
    #     prediction_comp_plot[1].set_title('model predictions from population', fontsize=14)
    #     prediction_comp_plot[0].show()
    #
    #     all_decoder_accuracy = []
    #     predictors = np.vstack(predictors_list)
    #     features = np.hstack(features_list)
    #     with multiprocessing.Pool() as pool:
    #         results = list(tqdm(pool.imap(partial(run_decoder, features=features, shuffle=to_shuffle, model='svc',cv_folds=None),
    #                                       [predictors] * n_runs), total=n_runs))
    #     all_decoder_accuracy.append(list(res[2] for res in results))
    #
    #     all_decoder_accuracy_plot = plot_decoder_accuracy([model_prediction_array[:,i,i]/n_tests
    #                                                        for i in range(model_prediction_array.shape[2])],all_labels,
    #                                                       n_features=4)
    #     all_decoder_accuracy_plot[0].show()
    # #
    # # decode_white_noise = False & sound_writes[sound_writes['Payload'] == 3].empty
    # # if decode_white_noise:
    # #     window = [0,0.25]
    # #     _predictor_list = [get_predictor_from_psth(sessions[sessname],key,psth_window,window) for key in ['X','base']]
    # #     _feature_list = [np.full(mat.shape[0],i) for i, mat in enumerate(_predictor_list)]
    # #
    # #     sessions[sessname].init_decoder('X_to_base',np.vstack(_predictor_list),np.hstack(_feature_list))
    # #     sessions[sessname].run_decoder('X_to_base',['data'],n_features=2)
    # #
    # #
    # #     x_decode_accuracy = []
    # #     x_decode_model = []
    # #     x_decode_fold_perf = []
    # #     n_runs = 500
    # #     # combos = [[0,4,1],[0,4,2],[0,4,3]]
    # #     x_sound_times = sound_writes[sound_writes['Payload'] == 3]['Timestamp']
    # #     x_none_sounds = [x_sound_times.values,np.random.choice(base_pips_times, len(x_sound_times), replace=False)]
    # #
    # #     sess_spike_obj.event_spike_matrices = {}
    # #     x_dataset = [get_event_rate_mat(et, e, sess_spike_obj, [0, 0.25]) for et,e in zip(x_none_sounds,[3,-2])]
    # #     x_predictors_list = [data_matrix.values.reshape([s_times.shape[0], -1, data_matrix.shape[1]]).mean(axis=2)
    # #                          for data_matrix, s_times in zip(x_dataset, x_none_sounds)]
    # #     x_features_list = [np.full(predictors.shape[0], i) for i, predictors in enumerate(x_predictors_list)]
    # #     predictors = np.vstack(x_predictors_list)
    # #     features = np.hstack(x_features_list)
    # #     for to_shuffle in [False, True]:
    # #         with multiprocessing.Pool() as pool:
    # #             results = list(tqdm(pool.imap(partial(run_decoder, features=features, shuffle=to_shuffle, model='svc',
    # #                                                   cv_folds=10),
    # #                                           [predictors] * n_runs), total=n_runs))
    # #         x_decode_accuracy.append(list(res[0] for res in results))
    # #         x_decode_model.append(list(res[1] for res in results))
    # #         x_decode_fold_perf.append(list(res[2] for res in results))
    # #
    # #     x_decoder_plot = plt.subplots()
    # #     x_events = ['X']
    # #     x_decoder_labels = ['data','shuffle']
    # #     for ri,_ in enumerate(x_decode_accuracy):
    # #         res = np.array(x_decode_fold_perf[ri]).flatten()
    # #         if ri == 1:
    # #             col = 'k'
    # #             lbl = None
    # #         else:
    # #             col = f'C{ri}'
    # #             lbl = x_events[ri]
    # #         x_decoder_plot[1].scatter(simple_beeswarm2(res, width=0.1) + ri, np.array(res),
    # #                                   label=lbl, alpha=0.1, c=col)
    # #     x_decoder_plot[1].set_ylim(0, 1.19)
    # #     x_decoder_plot[1].set_ylabel('decoder accuracy')
    # #     x_decoder_plot[1].axhline(0.5, c='k', ls='--')
    # #
    # #     x_decoder_plot[1].set_xticks(np.arange(0, len(x_decode_accuracy), 1))
    # #     x_decoder_plot[1].set_xticklabels(x_decoder_labels)
    # #     x_decoder_plot[1].legend(loc=1)
    # #     x_decoder_plot[1].set_title(f'decoding from population means firing rate: {n_runs} runs')
    # #     x_decoder_plot[0].show()
