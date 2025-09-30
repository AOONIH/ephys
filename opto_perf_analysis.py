import platform
from copy import copy
from itertools import combinations

import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')

from sess_dataclasses import Session

import numpy as np
import pandas as pd
from tqdm import tqdm

from behviour_analysis_funcs import group_td_df_across_sessions, get_all_cond_filts, parse_perturbation_log, get_perturbation_sessions
from io_utils import posix_from_win, load_pupil_data
from plot_funcs import format_axis
from pupil_analysis_funcs import load_pupil_sess_pkl, run_pupil_cond_analysis, PupilCondAnalysis
from scipy.stats import ttest_ind, tukey_hsd

from save_utils import save_stats_to_tex


if __name__ == '__main__':
    import argparse, yaml
    from pathlib import Path

    # Parse arguments and load config
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('--ow', default=0, type=int)
    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])
    home_dir = Path(config[f'home_dir_{sys_os}'])
    stats_dir = ceph_dir / posix_from_win(r'X:\Dammy\stats')

    fig_dir = ceph_dir / 'Dammy/figures/supp_perf_plots'
    if not fig_dir.exists():
        fig_dir.mkdir()

    plt.style.use('figure_stylesheet.mplstyle')

    # Load session pickle
    sess_pkl_path = Path(r'D:') / 'all_opto_w_pupilsense_analysis.joblib'
    sessions: dict = load_pupil_sess_pkl(sess_pkl_path)
    loaded_sess_dict_sig = copy(tuple(sorted(sessions.items())))

    window = [-1, 3]

    # Parse drug/perturbation log and get drug session dictionary
    drug_log_xlsx_path = r"X:\Dammy\Xdetection_mouse_hf_test\metadata\animal_log.xlsx"
    perturb_log = parse_perturbation_log(drug_log_xlsx_path)
    opto_sess_dict = get_perturbation_sessions(perturb_log['opto_log'], 'opto_state')

    # merge session_long_light_on with light_on
    if 'session_long_light_on' in opto_sess_dict:
        opto_sess_dict['light_on'] = list(set(opto_sess_dict['light_on'] + opto_sess_dict['session_long_light_on']))
        del opto_sess_dict['session_long_light_on']

    # opto_batch = [f'RS0{i}' for i in '123678']
    opto_batch = [f'RS0{i}' for i in '123']
    control_batch = [f'RS0{i}' for i in '459']


    # Group trial data across sessions
    all_td_df = group_td_df_across_sessions(sessions, list(sessions.keys()))
    all_td_df['PatternPresentation_Rate'] = all_td_df['PatternPresentation_Rate'].round(1)
    cond_filts = get_all_cond_filts()
    good_trial_filt = 'n_since_last_Trial_Outcome <=5 '  # Optionally add a good trial filter

    td_df_by_cond = {}
    rare_freq_condnames = ['frequent', 'rare']
    rare_freq_line_kwargs = {
        'rare': {'c': '#00cbfcff', 'ls': '-'},
        'frequent': {'c': '#1530e9ff', 'ls': '-'}
    }

    #### --- Rare vs frequent analysis --- ####
    for cond in rare_freq_condnames:
        td_df_by_cond[cond] = all_td_df.query('&'.join([cond_filts[cond], good_trial_filt]))
    all_rare_freq_sess = [sess for sess in sessions
                          if all([sess in td_df_by_cond[cond].index.get_level_values('sess')
                                  for cond in rare_freq_condnames])]
    opto_batch_dict = {'opto_batch': opto_batch}
    perf_stats = {}
    rt_stats = {}
    for opto_state in ['light_off','light_on']:
        for batch_name, batch_mice in opto_batch_dict.items():
            opto_state_sesslist = [sess for sess in sessions
                                   if sess in opto_sess_dict[opto_state] and sess.split('_')[0] in batch_mice]
            if len(opto_state_sesslist) == 0:
                print(f'{opto_state}: {batch_name} rare freq empty')
                continue

            # plot performance as barplot
            all_sess_perf_bar_plot = plt.subplots()
            patt_nonpatt_cols = ['dimgray', 'indigo']

            rare_freq_sess_df = all_td_df.loc[all_td_df.index.get_level_values('sess').isin(opto_state_sesslist)]

            perf_data = rare_freq_sess_df.query('Stage>=3 & 20<trial_num<=250').groupby(['name', 'Tone_Position'])[
                'Trial_Outcome'].mean().unstack().dropna(axis=0)[[1, 0]].values
            perf_stats[opto_state] = perf_data

            all_sess_perf_bar_plot[1].bar(['Pattern', 'Non pattern'], perf_data.mean(axis=0),
                                          ec='black', fc='w', lw=0.5, width=0.4)
            [all_sess_perf_bar_plot[1].scatter([cond] * len(cond_data), cond_data, facecolor=c, alpha=0.5, lw=0.01)
             for cond, cond_data, c in zip(['Non pattern', 'Pattern'], perf_data.T, patt_nonpatt_cols)]
            [all_sess_perf_bar_plot[1].plot(['Non pattern', 'Pattern'], a_data, lw=0.25, color='gray')
             for a_data in perf_data]
            all_sess_perf_bar_plot[1].set_ylim(.4, 1.01)
            format_axis(all_sess_perf_bar_plot[1])
            all_sess_perf_bar_plot[1].set_ylabel('Hit rate')
            all_sess_perf_bar_plot[1].set_title(f'{opto_state}: {batch_name}')
            all_sess_perf_bar_plot[0].set_size_inches(1.25, 1.5)
            all_sess_perf_bar_plot[0].show()
            all_sess_perf_bar_plot[0].savefig(fig_dir / f'performance_by_tone_position_{opto_state}_{batch_name}_barplot.pdf')

            rt_df = rare_freq_sess_df.query('Stage>=3 & Trial_Outcome==1')
            rt_df.set_index('Tone_Position', append=True,inplace=True)
            rt_df = (rt_df['Trial_End_dt'] - rt_df['Gap_Time_dt']).dt.total_seconds()
            rt_df = rt_df.groupby(['name', 'Tone_Position']).mean().unstack().dropna(axis=0)[[1, 0]]
            rt_stats[opto_state] = rt_df.values

    # plot all on 1 plot, same state closer together
    all_states_perf_plot = plt.subplots()
    data = [perf_stats[state][:,idx] for state in perf_stats for idx in range(2)]
    labels = [f'{state} \n {ttype}' for state in perf_stats for ttype in ['Non pattern', 'Pattern']]

    all_states_perf_plot[1].bar(labels, [d.mean() for d in data],
                                ec='black', fc='w', lw=0.5, width=0.4)
    [all_states_perf_plot[1].scatter([label]*len(d), d, facecolor=patt_nonpatt_cols[i%2], alpha=0.5, lw=0.01)
     for i, (label, d) in enumerate(zip(labels, data))]
    # all_states_perf_plot[1].plot()
    all_states_perf_plot[1].set_ylim(.4, 1.01)
    all_states_perf_plot[1].set_ylabel('Hit rate')
    all_states_perf_plot[0].set_size_inches(2.5, 1.5)
    all_states_perf_plot[0].show()
    all_states_perf_plot[0].set_layout_engine('tight')
    all_states_perf_plot[0].savefig(fig_dir / f'performance_by_tone_position_all_states_barplot.pdf')

    ttest_ind(perf_stats['light_off'][0], perf_stats['light_on'][0])
    ttest_ind(perf_stats['light_off'][1], perf_stats['light_on'][1])
    ttest_ind(perf_stats['light_on'][0], perf_stats['light_on'][1])

    tukey_test = tukey_hsd(*data)
    tukey_test_names = [f'{state} {idx}' for state in perf_stats for idx in range(2)]
    print(tukey_test)
    save_stats_to_tex(tukey_test, stats_dir/'opto_perf_tukey_test.tex')

    # plot rt data and tukey on rt times

    all_states_rt_plot = plt.subplots()
    data = [rt_stats[state][:,idx] for state in rt_stats for idx in range(2)]
    labels = [f'{state} \n {ttype}' for state in rt_stats for ttype in ['Non pattern', 'Pattern']] 
    ylabel = 'Reaction time (s)'
    all_states_rt_plot[1].bar(labels, [d.mean() for d in data],
                                ec='black', fc='w', lw=0.5, width=0.4)
    [all_states_rt_plot[1].scatter([label]*len(d), d, facecolor=patt_nonpatt_cols[i%2], alpha=0.5, lw=0.01)
     for i, (label, d) in enumerate(zip(labels, data))]
    all_states_rt_plot[1].set_ylabel(ylabel)
    all_states_rt_plot[1].set_ylim(0, .35)
    all_states_rt_plot[0].set_size_inches(2.5, 1.5)
    all_states_rt_plot[0].show()
    all_states_rt_plot[0].set_layout_engine('tight')
    all_states_rt_plot[0].savefig(fig_dir / f'rt_by_tone_position_all_states_barplot.pdf')
