import platform
from copy import copy
from itertools import combinations

import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl

from TimeSeries_clasification.cluster_analysis import plot_clusters

mpl.use('TkAgg')

from sess_dataclasses import Session

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from aggregate_psth_analysis import padded_rolling_mean

import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from mne.stats import permutation_cluster_test

from behviour_analysis_funcs import group_td_df_across_sessions, get_all_cond_filts, get_drug_dates, \
    parse_perturbation_log, get_perturbation_sessions, get_main_sess_patterns
from io_utils import posix_from_win, load_pupil_data
from plot_funcs import format_axis
from pupil_analysis_funcs import PupilCondAnalysis, init_pupil_td_obj, load_pupil_sess_pkl, process_pupil_td_data, \
    init_sess_pupil_obj, process_pupil_obj, add_early_late_to_by_cond, group_pupil_across_sessions, \
    add_name_to_response_dfs, run_pupil_cond_analysis
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

    td_path_pattern = 'data/Hilde/<name>/TrialData'

    plt.style.use('figure_stylesheet.mplstyle')

    # Load session topology and pupil data
    session_topology_paths = [
        # ceph_dir / posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_musc_sept23.csv'),
        ceph_dir / posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_musc_2406.csv'),
        ceph_dir / posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_ephys_2401.csv'),
    ]
    # pupil_pkl_paths = [ceph_dir / posix_from_win(p) for p in [
    #     # r'X:\Dammy\mouse_pupillometry\pickles\musc_sept23_stage3_up_90Hz_hpass00_lpass0.joblib',
    #     r'X:\Dammy\mouse_pupillometry\pickles\musc_2406_stage3_up_90Hz_hpass0_lpass0.joblib',
    #     r'X:\Dammy\mouse_pupillometry\pickles\ephys_2401_stage3_up_90Hz_hpass00_lpass0.joblib'    ]]
    pupil_pkl_paths = [
        Path(r'D:\pupil\musc_2406_stage3_up_90Hz_hpass0_lpass0.joblib',),
        Path(r'D:\pupil\ephys_2401_stage3_up_90Hz_hpass00_lpass0.joblib')
    ]


    # Concatenate session topology dataframes
    session_topology_dfs = [pd.read_csv(sess_topology_path) for sess_topology_path in session_topology_paths]
    session_topology_df = pd.concat(session_topology_dfs, names=['cohort']).reset_index()
    all_sess_info = session_topology_df

    # Load session pickle
    sess_pkl_path = Path(r'D:') / 'all_w_stage4.joblib'
    sessions: dict = load_pupil_sess_pkl(sess_pkl_path)
    loaded_sess_dict_sig = copy(tuple(sorted(sessions.items())))

    window = [-1, 3]

    # Parse drug/perturbation log and get drug session dictionary
    drug_log_xlsx_path = r"X:\Dammy\Xdetection_mouse_hf_test\metadata\animal_log.xlsx"
    perturb_log = parse_perturbation_log(drug_log_xlsx_path)
    drug_sess_dict = get_perturbation_sessions(perturb_log['infusion_log'], 'drug')

    # Process trial data for each session
    if len(sessions) == 0:
        # Merge pupil data from both cohorts
        all_pupil_data = [load_pupil_data(pkl_path) for pkl_path in pupil_pkl_paths]
        pupil_data_dict = {**all_pupil_data[0], **all_pupil_data[1]}

        for sessname in tqdm(list(pupil_data_dict.keys()), desc='processing sessions'):
            if sessname not in sessions:
                sessions[sessname] = Session(sessname, ceph_dir)
            if sessions[sessname].td_df is not None and not not args.ow:
                continue
            print(sessname)
            print(f'initializing and processing td for {sessname}')
            init_pupil_td_obj(sessions, sessname, ceph_dir, all_sess_info, td_path_pattern, home_dir)
            process_pupil_td_data(sessions, sessname,{})

        # Process pupil object for each session
        for sessname in tqdm(list(sessions.keys()), desc='processing sessions'):
            if sessions[sessname].pupil_obj is not None and not args.ow:
                continue
            print(sessname)
            init_sess_pupil_obj(sessions, sessname, ceph_dir, all_sess_info, pupil_data_dict)
            process_pupil_obj(sessions, sessname, alignmethod='w_soundcard', align_kwargs={'size_col': 'pupilsense_raddi_a_zscored'})

        # save
        joblib.dump(sessions, sess_pkl_path)

    # Get condition filters
    cond_filters = get_all_cond_filts()

    # Group trial data across sessions
    all_td_df = group_td_df_across_sessions(sessions, list(sessions.keys()))
    all_td_df['PatternPresentation_Rate'] = all_td_df['PatternPresentation_Rate'].round(1)
    good_trial_filt = 'n_since_last_Trial_Outcome <=5 '  # Optionally add a good trial filter

    # Define conditions to analyze
    conds2analyze = ['rare_prate', 'frequent_prate', 'rare', 'frequent']
    # Group pupil data by condition
    A_by_cond = {cond: group_pupil_across_sessions(
        sessions, list(sessions.keys()), 'A', cond, cond_filters)
        for cond in conds2analyze}
    X_by_cond = {cond: group_pupil_across_sessions(
        sessions, list(sessions.keys()), 'X', cond, cond_filters)
        for cond in ['hit_all', 'miss_all']+conds2analyze}

    # Add early/middle/late filters to frequent_prate
    rare_freq_early_late_dict = {'early': '<=8', 'middle': '>9 & <col> <=17', 'late': '>19 & frequent_block_num==1'}
    add_early_late_to_by_cond(A_by_cond, ['frequent_prate'], rare_freq_early_late_dict, sessions, 'A', cond_filters)

    # Plot X response by condition
    X_resp = PupilCondAnalysis(
        by_cond_dict=X_by_cond,
        conditions=['hit_all', 'miss_all'],
    )
    X_resp.plot_ts_by_cond()
    X_resp.plots['ts_by_cond'][0].show()
    X_resp.plots['ts_by_cond'][0].savefig('X_resp.pdf')


    # Rare vs frequent analysis
    rare_freq_figdir = ceph_dir / 'Dammy' / 'figures' / 'rare_freq_ctrl_all_pupilsense'
    if not rare_freq_figdir.exists():
        rare_freq_figdir.mkdir(exist_ok=True)

    rare_freq_line_kwargs = {
        'rare': {'c': '#00cbfcff', 'ls': '-'},
        'frequent': {'c': '#1530e9ff', 'ls': '-'}
    }

    boxplot_kwargs = dict(
        widths=0.5,
        patch_artist=False,
        showmeans=False,
        showfliers=False,
        medianprops=dict(mfc='k'),
        boxprops=dict(lw=0.5),
        whiskerprops=dict(lw=0.5),
        capprops=dict(lw=0.5),
    )
    rare_freq_condnames = ['frequent', 'rare']
    for cond in rare_freq_condnames:
        if cond not in X_by_cond:
            X_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()), 'X', cond, cond_filters)

    all_rare_freq_sess = [sess for sess in sessions
                          if all([sess in A_by_cond[cond].index.get_level_values('sess')
                                  for cond in rare_freq_condnames])]
    drug_rare_freq_sesslist = [sess for sess in all_rare_freq_sess
                               if sess not in drug_sess_dict['muscimol']
                               and sess not in drug_sess_dict['saline']]

    for stim_name, stim_by_cond in zip([ 'A', 'X'], [A_by_cond, X_by_cond]):
        # continue
        rare_freq_analysis=run_pupil_cond_analysis(
            by_cond_dict=stim_by_cond,
            sess_list=drug_rare_freq_sesslist,
            conditions=rare_freq_condnames,
            figdir=rare_freq_figdir,
            line_kwargs=rare_freq_line_kwargs,
            event_name='pattern' if stim_name == 'A' else stim_name,
            boxplot_kwargs=boxplot_kwargs,
            window_by_stim=(1.5, 2.5),
            smoothing_window=25,
            max_window_kwargs={'max_func': 'max'},
            group_name='sess',
            cluster_groupname='sess',
            p_alpha=0.05,
            n_permutations=100,
            permutation_test=False,
            stats_dir=stats_dir,
            tex_name=f'rarefreq_{stim_name}_data_vs_shuffled_no_filt.tex',
            ylabel="Δ pupil size",
            figsize=(2.2, 1.8),
            fig_savename=f'rare_freq_{stim_name}_ts',
            # ylim_ts=(-0.1, 0.5) if stim_name == 'A' else (-0.05, 0.6),
            ylim_maxdiff=(-0.05, 0.3) if stim_name == 'A' else None,
        )

    ### Normal vs deviant ###
    norm_dev_figdir = ceph_dir / 'Dammy' / 'figures' / 'norm_dev_ctrl_all_pupilsense'
    if not norm_dev_figdir.exists():
        norm_dev_figdir.mkdir()

    norm_dev_line_kwargs = {
        'normal': {'c': 'black', 'ls': '-'},
        'deviant_C': {'c': '#cd2727ff', 'ls': '-'}
    }
    norm_dev_cond_names = ['normal', 'deviant_C']

    for cond in norm_dev_cond_names:
        A_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()), 'A', cond, cond_filters)
        X_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()),'X', cond, cond_filters)

    norm_dev_sess = [sess for sess in sessions
                     if all([sess in A_by_cond[cond].index.get_level_values('sess')
                             for cond in norm_dev_cond_names])]
    norm_dev_sesslist = [sess for sess in norm_dev_sess
                         if sess not in drug_sess_dict['muscimol']
                         and sess not in drug_sess_dict['saline']]

    for stim_name, stim_by_cond in zip(['A', 'X'], [A_by_cond, X_by_cond]):
        # continue
        norm_dev_analysis = run_pupil_cond_analysis(
            by_cond_dict=stim_by_cond,
            sess_list=norm_dev_sesslist,
            conditions=norm_dev_cond_names,
            figdir=norm_dev_figdir,
            line_kwargs=norm_dev_line_kwargs,
            event_name='pattern' if stim_name == 'A' else stim_name,
            boxplot_kwargs=boxplot_kwargs,
            window_by_stim=(1.5, 2.9),
            smoothing_window=25,
            max_window_kwargs={'max_func': 'max'},
            group_name=['sess'],
            cluster_groupname='sess',
            p_alpha=0.05,
            n_permutations=100,
            permutation_test=False,
            stats_dir=stats_dir,
            fig_savename=f'normdev_{stim_name}_ts.pdf',
            tex_name=f'normdev_{stim_name}_data_vs_shuffled_no_filt.tex',
            ylabel="Δ pupil size",
            figsize=(2.2, 1.8),
            # ylim_ts=(-0.2, 0.5) if stim_name == 'A' else (-0.05, 0.6),
            ylim_maxdiff=(-0.05, 0.3)
        )

    # ttest_ind(norm_dev_analysis.by_cond['deviant_C'],norm_dev_analysis.by_cond['normal'],alternative='greater')

    # --- Add dev_ABCD1 and dev_ABBA1 analysis ---
    dev_figdir = ceph_dir / 'Dammy' / 'figures' / 'dev_ABCD_ABBA_ctrl_all_pupilsense'
    if not dev_figdir.exists():
        dev_figdir.mkdir()

    dev_cond_names = ['normal','dev_ABCD1', 'dev_ABBA1']
    dev_line_kwargs = {
        'normal': {'c': 'k', 'ls': '-'},
        'dev_ABCD1': {'c': '#e8739bff', 'ls': '-'},
        'dev_ABBA1': {'c': '#e47b15ff', 'ls': '-'}
    }

    # Add dev_ABCD1 and dev_ABBA1 to A_by_cond
    all_abstr_df = all_td_df.query(cond_filters['dev_ABCD1'])
    all_abstr_df.index.get_level_values('sess').unique()
    for sessname in sessions:
        main_patterns = get_main_sess_patterns(td_df=sessions[sessname].td_df)
        start_ids = list(set([patt[0] for patt in main_patterns]))
        if len(start_ids) <= 1:
            continue
        events = {e: {'idx': idx, 'filt': filt} for e, idx, filt in zip(['A_dev', ],
                                                                        [start_ids[1]],
                                                                        ['pip_counter==1'])}
        [sessions[sessname].get_pupil_to_event(e_dict['idx'], e, window,
                                               align_kwargs=dict(sound_df_query=e_dict['filt'], baseline_dur=1,
                                                                 size_col='pupilsense_raddi_a_zscored'),
                                               alignmethod='w_soundcard', )
         for e, e_dict in events.items()]
    a_dev_sess = [sess for sess in sessions if 'A_dev' in sessions[sess].pupil_obj.aligned_pupil]
    A_by_cond['normal'] = group_pupil_across_sessions(sessions,list(sessions.keys()),'A','normal',cond_filters)
    for cond in dev_cond_names[1:]:
        # A_by_cond[cond] = group_pupil_across_sessions(sessions, a_dev_sess, 'A_dev', cond, cond_filters)
        A_by_cond[cond] = group_pupil_across_sessions(sessions, a_dev_sess,
                                               'A_dev', cond_name=cond,
                                               cond_filters=cond_filters)

    add_name_to_response_dfs(A_by_cond)
    dev_sess = [sess for sess in sessions
                if all([sess in A_by_cond[cond].index.get_level_values('sess')
                        for cond in dev_cond_names])]
    dev_sesslist = [sess for sess in dev_sess
                    if sess not in drug_sess_dict['muscimol']
                    and sess not in drug_sess_dict['saline']]

    dev_ABBA1_ABCD_analysis = run_pupil_cond_analysis(
        by_cond_dict=A_by_cond,
        sess_list=dev_sesslist,
        conditions=dev_cond_names,
        figdir=dev_figdir,
        line_kwargs=dev_line_kwargs,
        event_name='pattern',
        boxplot_kwargs=boxplot_kwargs,
        window_by_stim=(0, 0.5),
        smoothing_window=1,
        max_window_kwargs={'max_func': 'max'},
        group_name=['sess','name'],
        cluster_groupname=['trial','sess'],
        cluster_comps=[['dev_ABCD1','dev_ABBA1']],
        p_alpha=0.05,
        n_permutations=100,
        permutation_test=False,
        stats_dir=stats_dir,
        tex_name='dev_ABCD_ABBA_data_vs_shuffled_no_filt.tex',
        ylabel="Δ pupil size",
        figsize=(2.2, 1.8),
        inset_max_diff=True,
        fig_savename=dev_figdir/'dev_ABCD_ABBA_ts_plot.pdf',
        # ylim_ts=(-0.1, 0.5),
        # ylim_maxdiff=(-0.05, 0.3)
    )

    # dev_ABBA1_ABCD_analysis.compute_clusters('dev_ABCD1','sess',p_alpha=0.05)
    # plot_clusters(A_by_cond['dev_ABCD1'].columns.values,
    #               dev_ABBA1_ABCD_analysis.cluster_analysis['dev_ABBA1_vs_dev_ABCD1'][0],
    #               dev_ABBA1_ABCD_analysis.cluster_analysis['dev_ABBA1_vs_dev_ABCD1'][1],
    #               dev_ABBA1_ABCD_analysis.plots['ts_by_cond'],p_alpha=0.05,
    #               plot_kwargs=dict(c='darkred')
    #               )
    dev_ABBA1_ABCD_analysis.plots['ts_by_cond'][0].show()
    # dev_ABBA1_ABCD_analysis.plots['ts_by_cond'][0].savefig(dev_figdir / 'dev_ABCD_ABBA_ts.pdf')

    ### Rare freq early late ###
    cond_filters['frequent_late'] = f'frequent_prate_cumsum >= {25}  & frequent_block_num == 2'
    rare_freq_early_late_conds =  [f'frequent_prate_{ti}' for ti in np.arange(0,25,5)][:3][::2]
    # add to A_by_cond
    for cond in rare_freq_early_late_conds:
        A_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()), 'A', cond, cond_filters)
        X_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()), 'X', cond, cond_filters)
    cmap = plt.get_cmap('Blues_r')  # or any other colormap
    colors = ['k'] + [cmap(i / (len(rare_freq_early_late_conds) - 1)) for i in range(len(rare_freq_early_late_conds))]
    rare_freq_early_late_line_kwargs = {
        cond: {'c': col, 'ls': '-'} for cond, col in zip(rare_freq_early_late_conds, colors)
    }

    all_rare_freq_early_late_sess = [sess for sess in sessions
                                     if all([sess in A_by_cond[cond].index.get_level_values('sess')
                                             for cond in rare_freq_early_late_conds])]
    early_late_diff_by_state = {}
    # Loop through unique opto states and per batch for rare freq early late analysis

    analysis = run_pupil_cond_analysis(
        by_cond_dict=A_by_cond,
        sess_list=all_rare_freq_early_late_sess,
        conditions=rare_freq_early_late_conds,
        figdir=rare_freq_figdir,
        line_kwargs=rare_freq_early_late_line_kwargs,
        boxplot_kwargs=boxplot_kwargs,
        window_by_stim=(1, 2),
        smoothing_window=5,
        max_window_kwargs={'max_func': 'max'},
        group_name=['sess'],
        permutation_test=False,
        n_permutations=100,
        cluster_groupname='sess',
        stats_dir=stats_dir,
        fig_savename=f'_rarefreq_early_late_ts_.pdf',
        tex_name=f'_rarefreq_early_late_data_vs_shuffled_no_filt_.tex',
        ylabel="Δ pupil size",
        figsize=(2.5, 2),
        # ylim_ts=(-0.05, 0.35),
        # ylim_maxdiff=(-0.25, 0.25),
        inset_max_diff=True
    )
    analysis.plots['ts_by_cond'][1].legend(loc='upper left')
    analysis.plots['ts_by_cond'][0].show()


