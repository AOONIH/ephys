import argparse
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

from aggregate_ephys_funcs import load_aggregate_td_df
from behviour_analysis_funcs import get_sess_name_date_idx, get_main_sess_patterns, get_main_sess_td_df
from sess_dataclasses import Session
from io_utils import posix_from_win

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])
    home_dir = Path(config[f'home_dir_{sys_os}'])
    td_path_pattern = 'data/Dammy/<name>/TrialData'

    plt.style.use('figure_stylesheet.mplstyle')


    session_topology_paths  = [
        ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_ephys_2401.csv'),
        ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_musc_2406.csv'),
    ]
    session_topology_dfs = [pd.read_csv(sess_topology_path) for sess_topology_path in session_topology_paths]
    all_sess_info = pd.concat(session_topology_dfs, axis=0).query('sess_order=="main"')
    all_sess_td_df = load_aggregate_td_df(all_sess_info,home_dir,'Stage==2')
    sessions2use = all_sess_td_df.index.get_level_values('sess').unique()
    sessions2use = [sorted([sess for sess in sessions2use if 'DO82' in sess])[i] for i in [0,1,3]]
    sessions2use = ['DO82_240404a'] + sessions2use

    sessions = {}
    figdir = ceph_dir/'Dammy'/'figures'/'lick_refinement_fig'
    if not figdir.is_dir():
        figdir.mkdir()

    for sessname in tqdm(sessions2use, total=len(sessions2use), desc='Getting sessions lick objs'):

        name, date, sess_idx, sess_info = get_sess_name_date_idx(sessname, all_sess_info)
        main_sess_td_name = Path(sess_info['sound_bin'].replace('_SoundData', '_TrialData')).with_suffix(
            '.csv').name

        sessions[sessname] = Session(sessname, ceph_dir)
        sessions[sessname].load_trial_data(get_main_sess_td_df(name, date, main_sess_td_name, home_dir)[1])
        if sessions[sessname].td_df.shape[0] < 100:
            print(f'{sessname} has too few trials: {sessions[sessname].td_df.shape[0]}')
            continue
        sound_bin_path = Path(sess_info['sound_bin'])
        beh_bin_path = Path(sess_info['beh_bin'])

        sound_write_path = sound_bin_path.with_stem(f'{sound_bin_path.stem}_write_indices').with_suffix('.csv')
        sound_write_path = ceph_dir / posix_from_win(str(sound_write_path))
        beh_events_path = beh_bin_path.with_stem(f'{beh_bin_path.stem}_event_data_32').with_suffix('.csv')
        beh_events_path = ceph_dir / posix_from_win(str(beh_events_path))

        if not beh_events_path.is_file():
            print(f'ERROR {beh_events_path} not found')
            continue


        normal_patterns = None
        sessions[sessname].init_lick_obj(beh_events_path, sound_write_path,normal_patterns)

        events = {e: {'idx':idx, 'filt':filt} for e, idx, filt in zip(['X'][:2],
                                                                      [3,8],
                                                                      ['','Time_diff>2'])}
        [sessions[sessname].get_licks_to_event(e_dict['idx'], e, align_kwargs=dict(sound_df_query=e_dict['filt'],
                                                                                   kernel_width=1))
         for e, e_dict in events.items()]

    X_licks = pd.concat([sessions[sess].lick_obj.event_licks['X_licks'] for sess in sessions],axis=0)
    X_licks.columns = X_licks.columns.total_seconds()
    sess_breaks = [np.where(X_licks.index.get_level_values('sess')==sess)[0][0] for sess in sessions] + [X_licks.shape[0]]
    x_sers = X_licks.columns
    x_lick_times = np.where(X_licks>1)
    cmap = plt.get_cmap('bone')
    x_licks_plot = plt.subplots()
    colors = [cmap(i / (len(sess_breaks) - 1)) for i in range(len(sess_breaks))]

    [x_licks_plot[1].scatter(x_sers[x_lick_times[1][np.all([x_lick_times[0]>=start,x_lick_times[0]<end],axis=0)]],
                             x_lick_times[0][np.all([x_lick_times[0]>=start,x_lick_times[0]<end],axis=0)],s=1,color=c,
                             marker='o')
     for start, end, c in zip(sess_breaks[:],sess_breaks[1:], colors)]
    x_licks_plot[1].axvline(0,ls='--',color='k')
    # [x_licks_plot[1].axhspan(sess_breaks[i],sess_breaks[i+1],fc='grey',alpha=0.1)
    #  for i,_ in enumerate(sess_breaks[:-1]) if i%2==0]
    x_licks_plot[1].set_xlim(-2,2)
    x_licks_plot[1].set_ylim(0,X_licks.shape[0]+1)
    x_licks_plot[1].invert_yaxis()
    x_licks_plot[1].set_yticks(np.arange(0,X_licks.shape[0]+1,1000))
    x_licks_plot[1].set_yticklabels(np.arange(0,X_licks.shape[0]+1,1000))
    x_licks_plot[1].locator_params(axis='x',nbins=3)
    x_licks_plot[0].set_size_inches(3,7)
    x_licks_plot[0].set_layout_engine('tight')
    x_licks_plot[0].show()
    x_licks_plot[0].savefig(figdir / 'lick_refinement2X_plus_stage1.svg')