from ephys_analysis_funcs import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import platform
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.stats.mstats import zscore
from scipy.stats import bootstrap, sem, t


def group_pupil_across_sessions(sessions_objs: dict,sessnames:list,event:str, cond_name:str,cond_filters:dict):
    all_cond_pupil = []
    cond_filter = cond_filters[cond_name]
    for sessname in sessnames:
        assert isinstance(sessions_objs[sessname],Session)
        if sessions_objs[sessname].pupil_obj is None:
            continue
        if sessions_objs[sessname].pupil_obj.aligned_pupil is None:
            continue
        trial_nums = sessions_objs[sessname].td_df.query(cond_filter).index.values
        cond_pupil = sessions[sessname].pupil_obj.aligned_pupil[event].loc[:, trial_nums+1, :]
        all_cond_pupil.append(cond_pupil)

    return pd.concat(all_cond_pupil,axis=0)


def get_pupil_diff_by_session(pupil_data1:pd.DataFrame, pupil_data2:pd.DataFrame) -> pd.DataFrame:
    unique_sess = np.unique(pupil_data1.index.get_level_values('sess'))
    pupil_diff_by_sess = []
    for sess in unique_sess:
        if not all(sess in df.index.get_level_values('sess') for df in [pupil_data1,pupil_data2]):
            print(sess)
            continue
        mean_diff = (pupil_data1.xs(sess,level='sess').mean(axis=0) - pupil_data2.xs(sess,level='sess').mean(axis=0))
        pupil_diff_by_sess.append(mean_diff)
    return pd.DataFrame(pupil_diff_by_sess)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])

    sess_topology_path = ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology.csv')
    try:
        gen_metadata(sess_topology_path,ceph_dir,col_name='beh_bin',harp_bin_dir='')
    except OSError:
        pass
    session_topology = pd.read_csv(sess_topology_path)
    sessions = {}
    all_sess_info = session_topology.query('sess_order=="main"')


    ephys_figdir = ceph_dir/'Dammy'/'figures'/'new_animals'
    if not ephys_figdir.is_dir():
        ephys_figdir.mkdir()

    all_pupil_data = load_pupil_data(Path(ceph_dir/posix_from_win(
        r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_ephys_2024_allsess_fam_2d_90Hz_hpass01_lpass4hanning015_TOM.pkl'
    )))
    home_dir = Path(config[f'home_dir_{sys_os}'])
    td_path_pattern = 'data/Dammy/<name>/TrialData'

    cond_filters = dict(rare='Tone_Position==0 & Stage==3 & local_rate >= 0.6',
                        frequent='Tone_Position==0 & Stage==3 & local_rate <= 0.1',
                        mid1='Tone_Position==0 & Stage==3 & local_rate > 0.3 & local_rate < 0.5',
                        mid2='Tone_Position==0 & Stage==3 & local_rate < 0.7 & local_rate >= 0.5')

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

        # get main sess pattern
        normal = get_main_sess_patterns(name, date, main_sess_td_name, home_dir)
        sessions[sessname].load_trial_data(abs_td_path)
        if not normal or 3 not in sessions[sessname].td_df['Stage'].values:
            continue

        sound_bin_path = Path(sess_info['sound_bin'])
        beh_bin_path = Path(sess_info['beh_bin'])

        sound_write_path = sound_bin_path.with_stem(f'{sound_bin_path.stem}_write_indices').with_suffix('.csv')
        beh_events_path = beh_bin_path.with_stem(f'{beh_bin_path.stem}_event_data_32').with_suffix('.csv')
        if not beh_events_path.is_file():
            print(f'ERROR {beh_events_path} not found')
            continue
        sessions[sessname].init_pupil_obj(all_pupil_data[sessname].pupildf,sound_write_path,
                                          beh_events_path,normal)
        [sessions[sessname].get_pupil_to_event(e_idx,e_name,[-1,3],align_kwargs=dict(baseline_dur=1))
         for e_idx,e_name in tqdm(zip([3,normal[0][0]],['X','A']),total=2,desc='pupil data processing')]

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

    A_by_cond = {cond: group_pupil_across_sessions(sessions,list(sessions.keys()),'A',cond,cond_filters)
                 for cond in ['rare','frequent',]}
    X_by_cond = {cond: group_pupil_across_sessions(sessions,list(sessions.keys()),'X',cond,cond_filters)
                 for cond in ['rare','frequent']}
    for event,event_response in zip(['A','X'],[A_by_cond,X_by_cond]):
        plot = plt.subplots()
        for cond_i, cond_name in enumerate(event_response):
            if cond_name in ['mid1','mid2']:
                continue
            cond_pupil: pd.DataFrame = event_response[cond_name]
            plot[1].plot(cond_pupil.mean(axis=0), label=cond_name)
            plot_ts_var(cond_pupil.columns,cond_pupil.values,f'C{cond_i}',plot[1])
            plot[1].set_title(f'pupil aligned to {event.replace("A","pattern")} onset')
            plot[1].set_xlabel(f'time from {event.replace("A","pattern")} onset(s)')
            plot[1].set_ylabel('pupil size')
            plot[1].legend()
        plot[0].show()

    rare_freq_diff_plot = plt.subplots()
    for event,event_response,col in zip(['A','X'],[A_by_cond,X_by_cond],['g','b']):
        rare_freq_diff = get_pupil_diff_by_session(event_response['rare'],event_response['frequent'])
        rare_freq_diff_plot[1].plot(rare_freq_diff.mean(axis=0),color=col,label=event.replace("A","pattern"))

        # plot_ts_var(rare_freq_diff.columns,rare_freq_diff.values,col,rare_freq_diff_plot[1],
        #             ci_kwargs={'var_func':sem,'confidence':0.99})
        rare_freq_diff_plot[1].fill_between(rare_freq_diff.columns,rare_freq_diff.mean(axis=0)-rare_freq_diff.sem(axis=0),
                                            rare_freq_diff.mean(axis=0)+rare_freq_diff.sem(axis=0),
                                            color=col,alpha=0.1)
    rare_freq_diff_plot[1].set_title(f'pupil difference between rare and frequent')
    rare_freq_diff_plot[1].set_xlabel(f'time from pattern onset(s)')
    rare_freq_diff_plot[1].set_ylabel('pupil size')
    rare_freq_diff_plot[1].legend()
    rare_freq_diff_plot[1].axhline(0,color='k',ls='--')
    rare_freq_diff_plot[1].axvline(0,color='k',ls='--')
    rare_freq_diff_plot[0].show()
