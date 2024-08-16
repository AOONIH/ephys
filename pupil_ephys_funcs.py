from behviour_analysis_funcs import sync_beh2sound, get_all_cond_filts, add_datetimecol, get_drug_dates, \
    get_n_since_last, get_earlyX_trials
from ephys_analysis_funcs import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import platform
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm
from scipy.stats import sem


def group_pupil_across_sessions(sessions_objs: dict,sessnames:list,event:str, cond_name:str,cond_filters:dict,
                                sess_list=None):
    all_cond_pupil = []
    cond_filter = cond_filters[cond_name]
    if cond_name == 'none':
        pass
    for sessname in sessnames:
        assert isinstance(sessions_objs[sessname],Session)
        if sessions_objs[sessname].pupil_obj is None:
            continue
        if sessions_objs[sessname].pupil_obj.aligned_pupil is None or \
                len(sessions_objs[sessname].pupil_obj.aligned_pupil) == 0:
            continue
        trial_nums = sessions_objs[sessname].td_df.query(cond_filter).index.get_level_values('trial_num').values
        cond_pupil = sessions[sessname].pupil_obj.aligned_pupil[event].loc[:, trial_nums, :]
        all_cond_pupil.append(cond_pupil)

    return pd.concat(all_cond_pupil,axis=0)


def get_pupil_diff_by_session(pupil_data1:pd.DataFrame, pupil_data2:pd.DataFrame,) -> pd.DataFrame:
    unique_sess = np.unique(pupil_data1.index.get_level_values('sess'))
    pupil_diff_by_sess = []
    used_sess = []
    for sess in unique_sess:
        if not all(sess in df.index.get_level_values('sess') for df in [pupil_data1,pupil_data2]):
            print(sess)
            continue
        if any([pupil_data1.xs(sess,level='sess').shape[0]<7, pupil_data2.xs(sess,level='sess').shape[0]<7]):
            print(f'{sess} not enough data for pupil diff')
            continue
        mean_diff = (pupil_data1.xs(sess,level='sess').mean(axis=0)-
                     pupil_data2.xs(sess,level='sess').mean(axis=0))
        pupil_diff_by_sess.append(mean_diff)
        used_sess.append(sess)
    return pd.DataFrame(pupil_diff_by_sess,index=pd.Index(used_sess,name='sess'))


def plot_pupil_diff_across_sessions(cond_list, event_responses, sess_drug_types, drug_sess_dict, dates=None, plot=None,
                                    plt_kwargs=None):
    if plot is None:
        plot = plt.subplots()
    if plt_kwargs is None:
        plt_kwargs = {}
    response_diff_df = get_pupil_diff_by_session(event_responses[cond_list[0]], event_responses[cond_list[1]])
    sess_idx_bool = np.any([response_diff_df.index.isin(drug_sess_dict[drug]) for drug in sess_drug_types], axis=0)
    response_diff_df_subset = response_diff_df[sess_idx_bool]
    if dates is not None:
        response_diff_df_subset = response_diff_df_subset.loc[dates]
    plot[1].plot(response_diff_df_subset.mean(axis=0), **plt_kwargs)

    # plot_ts_var(response_diff_df_subset.columns, response_diff_df_subset.values, plt_kwargs['c'], plot[1],
    #             ci_kwargs={'var_func': sem, 'confidence': 0.99})
    plot[1].fill_between(response_diff_df_subset.columns,response_diff_df_subset.mean(axis=0)-response_diff_df_subset.sem(axis=0),
                                        response_diff_df_subset.mean(axis=0)+response_diff_df_subset.sem(axis=0),
                                        color=plt_kwargs.get('c','k'),alpha=0.1)
    return response_diff_df_subset


def decode_responses(predictors, features, model_name='logistic', n_runs=100, dec_kwargs=None):
    decoder = {dec_lbl: Decoder(np.vstack(predictors), np.hstack(features), model_name=model_name)
               for dec_lbl in ['data', 'shuffled']}
    dec_kwargs = dec_kwargs or {}
    [decoder[dec_lbl].decode(dec_kwargs=dec_kwargs | {'shuffle': shuffle_flag}, parallel_flag=False,
                             n_runs=n_runs)
     for dec_lbl, shuffle_flag in zip(decoder, [False, True])]

    return decoder


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])

    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['axes.spines.top'] = False
    matplotlib.rcParams['legend.fontsize'] = 13
    matplotlib.rcParams['axes.titlesize'] = 16
    matplotlib.rcParams['axes.labelsize'] = 14
    matplotlib.rcParams['xtick.labelsize'] = 13
    matplotlib.rcParams['ytick.labelsize'] = 13
    matplotlib.rcParams['figure.labelsize'] = 18


    # sess_topology_path = ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology.csv')
    sess_topology_path = ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_musc_2406.csv')
    # sess_topology_path = ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_musc_may23.csv')
    # sess_topology_path = ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_musc_sept23.csv')
    # sess_topology_path = ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_fam_jan23.csv')

    # try:
    #     gen_metadata(sess_topology_path,ceph_dir,col_name='beh_bin',harp_bin_dir='')
    # except OSError:
    #     pass
    session_topology = pd.read_csv(sess_topology_path)
    sessions = {}
    all_sess_info = session_topology.query('sess_order=="main"')

    cohort_tag = "_".join(sess_topology_path.stem.split(('_'))[-2:])

    ephys_figdir = ceph_dir/'Dammy'/'figures'/'fch2_3_pupil_fam_no_drug'
    if not ephys_figdir.is_dir():
        ephys_figdir.mkdir()

    pkls = [ceph_dir / posix_from_win(p) for p in [
        r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_ephys_2024_allsess_fam_2d_90Hz_hpass01_lpass4hanning015_TOM.pkl',
        # r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_musc_2406_allsess_fam_2d_90Hz_hpass01_lpass4hanning015_TOM.pkl',
        r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_musc_2406_allsess_v2_fam_2d_90Hz_hpass01_lpass4_TOM.pkl',
        r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_sept23_w_canny_fam_2d_90Hz_hpass01_lpass4_TOM.pkl',
        r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_musc_may23_fam_2d_90Hz_hpass01_lpass4_TOM.pkl',
        r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_fam_jan23_fam_2d_90Hz_hpass01_lpass4_TOM.pkl',]]

    cohort_pkl_dict = {k:v for k,v in zip(['ephys_2401','musc_2406','musc_sept23','musc_may23','fam_jan23'],pkls)}

    # all_pupil_data = load_pupil_data(Path(ceph_dir/posix_from_win(
    #     r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_ephys_2024_allsess_fam_2d_90Hz_hpass01_lpass4hanning015_TOM.pkl',
    #     # r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_musc_2406_allsess_fam_2d_90Hz_hpass01_lpass4hanning015_TOM.pkl'
    #     # r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_musc_2406_allsess_v2_fam_2d_90Hz_hpass01_lpass4_TOM.pkl'
    #     # r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_sept23_w_canny_fam_2d_90Hz_hpass01_lpass4_TOM.pkl'
    #     # r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_musc_may23_fam_2d_90Hz_hpass01_lpass4_TOM.pkl'
    # )))
    all_pupil_data = load_pupil_data(cohort_pkl_dict[cohort_tag])
    print(all_pupil_data.keys())
    home_dir = Path(config[f'home_dir_{sys_os}'])
    td_path_pattern = 'data/Dammy/<name>/TrialData'

    cohort_config_path = home_dir/'gd_analysis'/'config'/f'{cohort_tag}.yaml'
    # cohort_config_path = home_dir/'gd_analysis'/'config'/'musc_sept23.yaml'
    # cohort_config_path = home_dir/'gd_analysis'/'config'/'musc_may23.yaml'
    with open(cohort_config_path, 'r') as file:
        cohort_config = yaml.safe_load(file)
    drug_sess_dict = {}
    get_drug_dates(cohort_config,session_topology,drug_sess_dict,date_end=240717)

    cond_filters = get_all_cond_filts()
    # common_filters = 'Tone_Position==0 & Stage==3 & N_TonesPlayed==4 & Trial_Outcome==1'
    # common_filters = 'Trial_Outcome==1 & Session_Block==0'
    good_trial_filt = 'Stage>=3 & n_since_last_Trial_Outcome <=5 & Session_Block==0 & Trial_Outcome==1'

    [cond_filters.update({k:' & '.join([v,good_trial_filt])}) for k,v in cond_filters.items()]

    for sessname in tqdm(list(all_pupil_data.keys()), desc='processing sessions'):
        print(sessname)
        sessions[sessname] = Session(sessname, ceph_dir)
        name, date = sessname.split('_')[:2]
        date = int(date)
        sess_info = all_sess_info.query('name==@name & date==@date').reset_index().query('sess_order=="main"').iloc[0]

        # get main sess pattern
        main_sess_td_name = Path(sess_info['sound_bin'].replace('_SoundData', '_TrialData')).with_suffix(
            '.csv').name

        td_path = td_path_pattern.replace('<name>', main_sess_td_name.split('_')[0])
        abs_td_path_dir = home_dir / td_path
        abs_td_path = next(abs_td_path_dir.glob(f'{name}_TrialData_{date}*.csv'))
        sessions[sessname].load_trial_data(abs_td_path)
        get_n_since_last(sessions[sessname].td_df, 'Trial_Outcome', 0)

        if not all([3 in sessions[sessname].td_df['Stage'].values, len(sessions[sessname].td_df) > 100]):
            if sessname in drug_sess_dict['muscimol']:
                drug_sess_dict['muscimol'].remove(sessname)
            if sessname in drug_sess_dict['saline']:
                drug_sess_dict['saline'].remove(sessname)
            if sessname in drug_sess_dict['none']:
                drug_sess_dict['none'].remove(sessname)
            sessions.pop(sessname)

    for sessname in tqdm(list(sessions.keys()), desc='processing sessions'):
        print(sessname)
        name, date = sessname.split('_')[:2]
        date = int(date)
        sess_info = all_sess_info.query('name==@name & date==@date').reset_index().query('sess_order=="main"').iloc[0]
        sound_bin_path = Path(sess_info['sound_bin'])
        beh_bin_path = Path(sess_info['beh_bin'])

        # get main sess pattern
        normal = get_main_sess_patterns(td_df=sessions[sessname].td_df)
        [add_datetimecol(sessions[sessname].td_df, col) for col in ['Trial_Start', 'ToneTime', 'Trial_End', 'Gap_Time',
                                                                    'Bonsai_Time']]
        get_earlyX_trials(sessions[sessname].td_df)

        sound_write_path = sound_bin_path.with_stem(f'{sound_bin_path.stem}_write_indices').with_suffix('.csv')
        sound_write_path = ceph_dir/posix_from_win(str(sound_write_path))
        sound_events_path = sound_bin_path.with_stem(f'{sound_bin_path.stem}_event_data_81').with_suffix('.csv')
        sound_events_path = ceph_dir/posix_from_win(str(sound_events_path))
        beh_events_path = beh_bin_path.with_stem(f'{beh_bin_path.stem}_event_data_32').with_suffix('.csv')
        beh_events_path = ceph_dir/posix_from_win(str(beh_events_path))
        beh_events_path_44 = beh_bin_path.with_stem(f'{beh_bin_path.stem}_event_data_44').with_suffix('.csv')
        beh_events_path_44 = ceph_dir/posix_from_win(str(beh_events_path_44))
        beh_writes_path = beh_bin_path.with_stem(f'{beh_bin_path.stem}_write_data').with_suffix('.csv')
        beh_writes_path = ceph_dir/posix_from_win(str(beh_writes_path))

        if not beh_events_path.is_file():
            print(f'ERROR {beh_events_path} not found')
            continue
        sessions[sessname].init_pupil_obj(all_pupil_data[sessname].pupildf,sound_write_path,
                                          beh_events_path,normal)
        if date < 240101:
            sound_events_df = pd.read_csv(sound_events_path,nrows=1)
            beh_events_44_df = pd.read_csv(beh_events_path_44,nrows=1)
            sync_beh2sound(sessions[sessname].pupil_obj, beh_events_44_df, sound_events_df)
        #
        # [sessions[sessname].get_pupil_to_event(e_idx,e_name,[-1,3], alignmethod='w_td_df',
        #                                        align_kwargs=dict(baseline_dur=1,))  # size_col='canny_raddi_a_zscored'
        #  for e_idx,e_name in tqdm(zip([3,normal[0][0]],['X','A']),total=2,desc='pupil data processing')]

        main_patterns = get_main_sess_patterns(td_df=sessions[sessname].td_df)
        normal_patterns = [pattern for pattern in main_patterns if np.all(np.diff(pattern) > 0)]

        normal = normal_patterns[0]
        # none = idx 8 pip_diff = 0 then only times > 2 secs from t start
        base_idx = normal[0] - 2
        events = {e: {'idx': idx, 'filt': filt} for e, idx, filt in zip(['X', 'A', 'Start', 'none'][:2],
                                                                        [3, normal[0], base_idx, base_idx],
                                                                        ['', '', 'Time_diff>1',
                                                                         'Payload_diff==0&rand_n==1&d_X_times>3&d_X_times<6'])}
        [sessions[sessname].get_pupil_to_event(e_dict['idx'], e,[-1,3],
                                               align_kwargs=dict(sound_df_query=e_dict['filt'],baseline_dur=1),
                                               alignmethod='w_soundcard')
         for e, e_dict in events.items()]


        # X_raster_plot = plot_2d_array_with_subplots(sessions[sessname].pupil_obj.aligned_pupil['X'],plot_cbar=False,
        #                                             cmap='Reds')
        # X_raster_plot[0].show()


        # trial_nums_by_cond = [sessions[sessname].td_df.query(cond_filter).index.values for cond_filter in cond_filters]

        # for event in ['X','A']:

        #
        # A_raster_plot = plot_2d_array_with_subplots(sessions[sessname].pupil_obj.aligned_pupil['A'],plot_cbar=False)
        # A_raster_plot[0].show()
        # # histofram of pupil diff
        # diff_hist_plot = plt.subplots()
        # diff_hist_plot[1].hist(sessions[sessname].pupil_obj.aligned_pupil['X'].diff(axis=1).values.flatten(),bins=100,
        #                        density=True)
        # diff_hist_plot[0].show()
        # e_diff = sessions[sessname].pupil_obj.aligned_pupil['X'].diff(axis=1)
        # A_pdr = e_diff > np.nanmean(e_diff)+np.nanstd(e_diff)*1.96
        # A_raster_pdr_plot = plt.subplots(2)
        # plot_2d_array_with_subplots(A_pdr.values,plot_cbar=False,cmap='Reds',
        #                             plot=(A_raster_pdr_plot[0],A_raster_pdr_plot[1][0]))
        # A_raster_pdr_plot[1][1].plot(A_pdr.mean(axis=0).loc[-0.75:],color='k')
        # # A_raster_pdr_plot[1][1].set_ylim(np.quantile(A_pdr.mean(axis=0),[0.01,1])*[0.95,1.1])
        # A_raster_pdr_plot[0].show()
    # [get_n_since_last(sess.td_df,'Trial_Outcome',1) for sess in sessions.values()]


    A_by_cond = {cond: group_pupil_across_sessions(sessions,list(sessions.keys()),'A',cond,cond_filters,)
                 for cond in ['rare','frequent',]}
    # A_by_cond['none'] = group_pupil_across_sessions(sessions,list(sessions.keys()),'none','none',cond_filters)
    X_by_cond = {cond: group_pupil_across_sessions(sessions,list(sessions.keys()),'X',cond,cond_filters)
                 for cond in ['rare','frequent']}

    assert pd.concat([A_by_cond['rare'],A_by_cond['frequent']]).drop_duplicates().shape[0] == pd.concat([A_by_cond['rare'],A_by_cond['frequent']]).shape[0]
    # pickle session dict
    pkl_dir = ceph_dir/'Dammy'/'pupil_data'
    if not pkl_dir.is_dir():
        pkl_dir.mkdir(parents=True)
    with open(pkl_dir/f'{cohort_tag}_sessions.pkl', 'wb') as f:
        pickle.dump(sessions, f)
    # pick A and X by cond
    pickle.dump(A_by_cond, open(pkl_dir/f'{cohort_tag}_A_by_cond.pkl', 'wb'))
    pickle.dump(X_by_cond, open(pkl_dir/f'{cohort_tag}_X_by_cond.pkl', 'wb'))

    rare_vs_frequent_drug_plots = plt.subplots(2,len(drug_sess_dict),sharey='row',figsize=(6*len(drug_sess_dict),5*2))
    for ei,(event,event_response) in enumerate(zip(['A','X'],[A_by_cond,X_by_cond])):
        for drug_i,drug in enumerate(['none','muscimol','saline']):
            for cond_i, cond_name in enumerate(event_response):
                plot = rare_vs_frequent_drug_plots[1][ei,drug_i]
                if cond_name in ['mid1','mid2']:
                    continue
                cond_pupil: pd.DataFrame = event_response[cond_name]
                sess_idx_bool = cond_pupil.index.get_level_values('sess').isin(drug_sess_dict[drug])
                cond_pupil_control_sess = cond_pupil[sess_idx_bool]
                mean_means_pupil = cond_pupil_control_sess.groupby(level='sess').mean().mean(axis=0)
                plot.plot(mean_means_pupil, label=cond_name)
                cis = sem(cond_pupil_control_sess.groupby(level='sess').mean())
                plot.fill_between(cond_pupil_control_sess.columns, mean_means_pupil-cis, mean_means_pupil+cis, alpha=0.1)
                # plot_ts_var(cond_pupil_control_sess.columns,cond_pupil_control_sess.values,f'C{cond_i}',plot)
                plot.set_xlabel(f'time from {event.replace("A","pattern")} onset(s)')
                plot.set_ylabel('pupil size')
                plot.legend()
                plot.axvline(0, color='k',ls='--')
                if event == 'A':
                    [plot.axvspan(t,t+0.15,fc='grey',alpha=0.1) for t in np.arange(0,1,0.25)]

    [plot.set_title(drug) for drug,plot in zip(['none','muscimol','saline'],rare_vs_frequent_drug_plots[1][0])]
    rare_vs_frequent_drug_plots[0].suptitle('Pupil response to rare vs. frequent')
    rare_vs_frequent_drug_plots[0].set_layout_engine('tight')
    # rare_vs_frequent_drug_plots[0].set_size_inches(8*2,3*len(drug_sess_dict))
    rare_vs_frequent_drug_plots[0].show()
    rare_vs_frequent_drug_plots[0].savefig(ephys_figdir/f'rare_vs_frequent_pupil_response_all_sessions_{cohort_tag}.svg')

    # plot rare vs freq none drug separate figure
    line_colours = ['darkblue','darkgreen','darkred','darkorange','darkcyan']
    rare_freq_plots = {}
    for ei,(event,event_response) in enumerate(zip(['A','X'],[A_by_cond,X_by_cond])):
        pupil_to_event_plot = plt.subplots()
        for cond_i, cond_name in enumerate(event_response):
            if cond_name in ['mid1','mid2']:
                continue
            cond_pupil: pd.DataFrame = event_response[cond_name]
            sess_idx_bool = cond_pupil.index.get_level_values('sess').isin(drug_sess_dict['none'])
            cond_pupil_control_sess = cond_pupil[sess_idx_bool]
            mean_means_pupil = cond_pupil_control_sess.groupby(level='sess').mean().mean(axis=0)
            plot = pupil_to_event_plot[1]
            plot.plot(mean_means_pupil, label=cond_name, color=line_colours[cond_i] if cond_name not in ['none','base'] else 'darkgrey')
            cis = sem(cond_pupil_control_sess.groupby(level='sess').mean())
            plot.fill_between(cond_pupil_control_sess.columns, mean_means_pupil-cis, mean_means_pupil+cis, alpha=0.1,
                              fc=line_colours[cond_i] if cond_name not in ['none','base'] else 'darkgrey', )
            plot.set_xlabel(f'time from {event.replace("A","pattern")} onset(s)')
            plot.set_ylabel('pupil size')
            plot.legend()
            plot.axvline(0, color='k',ls='--')
            if event == 'A':
                [plot.axvspan(t,t+0.15,fc='grey',alpha=0.1) for t in np.arange(0,1,0.25)]

        pupil_to_event_plot[1].set_title('Pupil response to rare vs. frequent')
        pupil_to_event_plot[0].set_layout_engine('tight')
        # pupil_to_event_plot[0].set_size_inches(8,3)
        pupil_to_event_plot[0].show()
        pupil_to_event_plot[0].savefig(ephys_figdir/f'rare_vs_frequent_pupil_response_none_drug_{event}_{cohort_tag}.svg')
        rare_freq_plots[event] = copy(pupil_to_event_plot)

    plot_indv_sessions = False
    # plot indv sessions for none drug
    if plot_indv_sessions:
        for drug in drug_sess_dict:
            if not drug_sess_dict[drug]:
                continue
            rare_vs_freq_by_sess_plot = plt.subplots(len(drug_sess_dict[drug]),figsize=(8, 6*len(drug_sess_dict[drug])))
            for sess,ax in tqdm(zip(drug_sess_dict[drug],rare_vs_freq_by_sess_plot[1].flatten()),
                                 total=len(drug_sess_dict[drug]),desc='plotting sessions'):
                for cond_i, cond_name in enumerate(['rare','frequent']):
                    sess_idx_bool = A_by_cond[cond_name].index.get_level_values('sess')==sess
                    cond_pupil = A_by_cond[cond_name][sess_idx_bool]
                    ax.plot(cond_pupil.mean(axis=0), label=cond_name)
                    plot_ts_var(cond_pupil.columns,cond_pupil.values,f'C{cond_i}',plt_ax=ax)
                    ax.set_title(f'{drug} drug session {sess}')
                    ax.set_xlabel(f'time from pattern onset (s)')
                    ax.set_ylabel('pupil size')
                    ax.axhline(0,color='k',ls='--')
            rare_vs_freq_by_sess_plot[0].set_layout_engine('tight')
            rare_vs_freq_by_sess_plot[0].show()

    rare_freq_diff_plot = plt.subplots(ncols=len(drug_sess_dict),sharey='row', figsize=(12, 5))
    for i,sess_type in enumerate(['none','saline','muscimol']):
        for event,event_response,col in zip(['Pattern','X'],[A_by_cond,X_by_cond],['g','b']):
            plot_pupil_diff_across_sessions(['rare','frequent'],event_response,[sess_type],drug_sess_dict,
                                            plot=[rare_freq_diff_plot[0],rare_freq_diff_plot[1][i]],
                                            plt_kwargs=dict(c=col,label=event))

            # rare_freq_diff_plot[1][i].set_xlabel(f'time from pattern onset (s))
            [rare_freq_diff_plot[1][i].axvspan(t, t+0.15,fc='grey',alpha=0.1) for t in np.arange(0,1,0.25)]
            rare_freq_diff_plot[1][i].set_title(f'{sess_type} sessions')
            rare_freq_diff_plot[1][i].axhline(0,color='k',ls='--')
            rare_freq_diff_plot[1][i].axvline(0,color='k',ls='--')



    # rare_freq_diff_plot[0].suptitle(f'Pupil difference between rare and frequent',fontsize=18)
    # [ax.tick_params(axis='both', labelsize=13) for ax in rare_freq_diff_plot[1]]
    [ax.locator_params(axis='both',nbins=4) for ax in rare_freq_diff_plot[1]]
    # rare_freq_diff_plot[1][0].set_ylabel('pupil size (rare - frequent)',fontsize=14)
    rare_freq_diff_plot[1][-1].legend()
    rare_freq_diff_plot[0].set_layout_engine('tight')
    rare_freq_diff_plot[0].show()

    rare_freq_diff_plot[0].savefig(ephys_figdir/f'rare_vs_frequent_pupil_diff_across_sessions_{cohort_tag}.svg')

    rare_freq_diff_all_drugs = plt.subplots()
    for i,(sess_type,col) in enumerate(zip(['none','saline','muscimol'],['darkgreen','darkblue','darkred'])):
        pupil_diff_data = plot_pupil_diff_across_sessions(['rare','frequent'], A_by_cond, [sess_type], drug_sess_dict,
                                                         plot=rare_freq_diff_all_drugs,
                                                         plt_kwargs=dict(c=col,label=sess_type))

    rare_freq_diff_all_drugs[1].set_title(f'Pupil difference between rare and frequent')
    rare_freq_diff_all_drugs[1].set_xlabel(f'Time from pattern onset (s)')
    rare_freq_diff_all_drugs[1].set_ylabel('pupil size (rare - frequent)')
    # rare_freq_diff_all_drugs[0].set_layout_engine('tight')
    box = rare_freq_diff_all_drugs[1].get_position()
    rare_freq_diff_all_drugs[1].set_position([box.x0, box.y0, box.width * 0.9, box.height])
    rare_freq_diff_all_drugs[0].legend(bbox_to_anchor=(.85, .88))
    rare_freq_diff_all_drugs[1].axhline(0, color='k', ls='--')
    rare_freq_diff_all_drugs[1].locator_params(axis='both', nbins=4)
    rare_freq_diff_all_drugs[1].tick_params(axis='both')
    [rare_freq_diff_all_drugs[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
    rare_freq_diff_all_drugs[0].set_size_inches(8, 6)
    rare_freq_diff_all_drugs[0].show()
    rare_freq_diff_all_drugs[0].savefig(ephys_figdir / f'rare_vs_frequent_pupil_diff_across_sessions_all_drugs_{cohort_tag}.svg')

    # # decode cond from pupil response to A use session means
    # A_predictors = []
    # A_labels = []
    # t_start = -0.75
    # x_ser = A_by_cond['rare'].loc[:, t_start:].columns
    # for cond_i,cond in enumerate(A_by_cond):
    #     cond_pupil = A_by_cond[cond]
    #     sess_idx_bool = cond_pupil.index.get_level_values('sess').isin(drug_sess_dict['none'])
    #     A_predictors.append(cond_pupil.diff(axis=1).loc[sess_idx_bool, t_start:].values)
    #     A_labels.append([cond_i]*cond_pupil[sess_idx_bool].shape[0])
    # A_vs_X_predictors = []
    # A_vs_X_labels = []
    # for cond_i, cond_pupil in enumerate([A_by_cond['frequent'], X_by_cond['frequent']]):
    #     sess_idx_bool = cond_pupil.index.get_level_values('sess').isin(drug_sess_dict['none'])
    #     A_vs_X_predictors.append(cond_pupil.diff(axis=1).loc[sess_idx_bool, t_start:].values)
    #     A_vs_X_labels.append([cond_i] * cond_pupil[sess_idx_bool].shape[0])
    # decoders_dict = {}
    # for xys,lbl in zip([[A_predictors,A_labels],[A_vs_X_predictors,A_vs_X_labels]],['rare vs frequent','Pattern vs X']):
    #     decoders_dict[lbl] = decode_pupil(xys[0], xys[1],dec_kwargs=dict(cv_folds=0,penalty='l1',solver='saga',
    #                                                                      n_jobs=os.cpu_count()))
    #
    # for lbl,decoder in decoders_dict.items():
    #     # plot accuracy
    #
    #     decoder_acc_plot = plt.subplots()
    #     decoder_acc_plot[1].boxplot([np.array(dec.accuracy) for dec in decoder.values()],
    #                                 labels=decoder.keys())
    #     decoder_acc_plot[1].set_title(f'Accuracy of {lbl}')
    #     decoder_acc_plot[0].set_layout_engine('tight')
    #     decoder_acc_plot[0].show()
    #
    #     # ttest on fold accuracy
    #     ttest = ttest_ind(*[np.array(dec.fold_accuracy).flatten() for dec in decoder.values()],
    #                       alternative='greater')
    #     dec_coefs_plot = plt.subplots()
    #     [dec_coefs_plot[1].plot(x_ser,
    #                             np.mean([e.coef_[0] for e in decoder['data'].models[run]], axis=0),
    #                             c='gray',alpha=0.25)
    #      for run in range(len(decoder['data'].models))]
    #     # dec_coefs_plot[1].plot(x_ser,
    #     #                         np.mean([[e.coef_[0] for e in rare_vs_freq_decoder['data'].models[run]]], axis=0),
    #     #                        c='r')
    #     dec_coefs_plot[1].set_title(f'Coefficients time series of {lbl}')
    #     dec_coefs_plot[1].axvline(0, color='k', ls='--')
    #     dec_coefs_plot[0].set_layout_engine('tight')
    #
    #     dec_coefs_plot[0].show()


