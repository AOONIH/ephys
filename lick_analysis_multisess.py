from ephys_analysis_funcs import *
import platform
import argparse
import yaml
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.signal import savgol_filter, correlate2d
import ruptures as rpt
import time
from postprocessing_utils import get_sorting_dirs
from datetime import datetime
import pandas as pd
import re


def generate_lick_data_for_variable_pip(sessions, pip):
    all_licks_to_pip = pd.concat(
        sessions[sessname].lick_obj.event_licks[f'{pip}_licks']
        for sessname in sessions.keys()
        if sessions[sessname].lick_obj and f'{pip}_licks' in sessions[sessname].lick_obj.event_licks
    )
    all_licks_to_pip.index = all_licks_to_pip.index.reorder_levels(['sess', 'trial', 'time'])

    patt_trials_to_pip = pd.concat([
        all_licks_to_pip.loc[sessname, sessions[sessname].td_df.query('Tone_Position==0 & Stage==3').index.values, :]
        for sessname in sessions
        if sessname in all_licks_to_pip.index.get_level_values('sess')
    ])
    none_trials_to_pip = pd.concat([
        all_licks_to_pip.loc[sessname, sessions[sessname].td_df.query('Tone_Position==1 & Stage==3').index.values, :]
        for sessname in sessions
        if sessname in all_licks_to_pip.index.get_level_values('sess')
    ])

    return patt_trials_to_pip, none_trials_to_pip


def plot_lick_data(all_licks_to_pip, pip,fig_kwargs=None,plt_kwargs=None) -> (plt.Figure, plt.Axes):
    fig, ax = plt.subplots(**fig_kwargs if fig_kwargs else {})
    for trial_type_licks, trial_type_name in zip(all_licks_to_pip, ['pattern', 'none']):
        ax.plot(trial_type_licks.mean(axis=0), label=trial_type_name, **plt_kwargs if plt_kwargs else {})
    ax.set_xlabel(f'time from {pip} (s)')
    ax.set_ylabel('lick rate')
    ax.legend()
    fig.set_constrained_layout(True)
    return fig, ax


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

    # get main sess pattern
    home_dir = Path(config[f'home_dir_{sys_os}'])
    td_path_pattern = 'data/Dammy/<name>/TrialData'

    ephys_figdir = ceph_dir/'Dammy'/'figures'/'new_animals'
    if not ephys_figdir.is_dir():
        ephys_figdir.mkdir()

    for _,sess_info in tqdm(all_sess_info.iterrows(),total=all_sess_info.shape[0],desc='Analyzing sessions'):
        name = sess_info['name']
        date = sess_info['date']

        sound_bin_path = Path(sess_info['sound_bin'])
        beh_bin_path = Path(sess_info['beh_bin'])

        sessname = Path(sess_info['sound_bin']).stem
        sessname = sessname.replace('_SoundData', '')
        sessions[sessname] = Session(sessname, ceph_dir)

        main_sess_td_name = Path(sess_info['sound_bin'].replace('_SoundData', '_TrialData')).with_suffix(
            '.csv').name
        td_path = td_path_pattern.replace('<name>', main_sess_td_name.split('_')[0])
        abs_td_path_dir = home_dir / td_path
        try:
            abs_td_path = next(abs_td_path_dir.glob(f'{name}_TrialData_{date}*.csv'))
        except StopIteration:
            print(f'{abs_td_path_dir} not found')
            continue
        print(f'{abs_td_path = }')
        sessions[sessname].load_trial_data(abs_td_path)

        sound_write_path = sound_bin_path.with_stem(f'{sound_bin_path.stem}_write_indices').with_suffix('.csv')
        beh_events_path = beh_bin_path.with_stem(f'{beh_bin_path.stem}_event_data_32').with_suffix('.csv')
        if any([not e.is_file() for e in [sound_write_path, beh_events_path]]):
            print(f'{sound_write_path} or {beh_events_path} not found')
            continue
        if sessions[sessname].td_df.query('Stage>=2')['PatternID'].empty:
            continue
        if sessions[sessname].td_df.query('Session_Block>=0 & Tone_Position==0')['PatternID'].empty:
            normal = sessions[sessname].td_df['PatternID'][0]
        else:
            normal = sessions[sessname].td_df.query('Session_Block>=0 & Tone_Position==0')['PatternID'].mode()[0]
        normal = [int(pip) for pip in normal.split(';')]
        sessions[sessname].init_lick_obj(beh_events_path, sound_write_path,normal)
        sessions[sessname].get_licks_to_event(3, 'X')
        [sessions[sessname].lick_obj.event_lick_plots[f'licks_to_{event}'][0].savefig(
            ephys_figdir / f'licks_to_{event}_{sessname}.svg', )
         for event in ['X']]
        sessions[sessname].lick_obj.event_lick_plots[f'licks_to_X'][1].set_title(f'{sessname}: licks to X')
        # sessions[sessname].lick_obj.event_lick_plots[f'licks_to_X'][0].show()
        if normal != [0, 0, 0, 0]:
            sessions[sessname].get_licks_to_event(normal[0], 'A')
            [sessions[sessname].lick_obj.event_lick_plots[f'licks_to_{event}'][0].savefig(
                ephys_figdir / f'licks_to_{event}_{sessname}.svg', )
                for event in ['A']]
            sessions[sessname].lick_obj.event_lick_plots[f'licks_to_A'][1].set_title(f'{sessname}: licks to A')
            # sessions[sessname].lick_obj.event_lick_plots[f'licks_to_A'][0].show()

    # get all licks to X

    for pip in ['X','A']:
        patt_trials_to_pip, none_trials_to_pip = generate_lick_data_for_variable_pip(sessions, pip)
        lick_to_pip_plot = plot_lick_data([patt_trials_to_pip, none_trials_to_pip], pip,dict(figsize=(3,2.5)))
        lick_to_pip_plot[1].set_ylim(0,0.15)
        lick_to_pip_plot[1].axvline(0,ls='--',color='k',lw=1)
        lick_to_pip_plot[1].locator_params(axis='both', nbins=3)
        lick_to_pip_plot[0].savefig(ephys_figdir / f'all_licks_to_{pip}.svg')
