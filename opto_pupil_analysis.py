import platform
from copy import copy
from itertools import combinations

import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl

from reformat_dir_struct import extract_date

mpl.use('TkAgg')

from sess_dataclasses import Session

import numpy as np
import pandas as pd
from tqdm import tqdm

from behviour_analysis_funcs import group_td_df_across_sessions, get_all_cond_filts, parse_perturbation_log, get_perturbation_sessions
from io_utils import posix_from_win, load_pupil_data
from plot_funcs import format_axis
from pupil_analysis_funcs import PupilCondAnalysis, init_pupil_td_obj, load_pupil_sess_pkl, process_pupil_td_data, \
    init_sess_pupil_obj, process_pupil_obj, add_early_late_to_by_cond, group_pupil_across_sessions, \
    add_name_to_response_dfs, run_pupil_cond_analysis
from scipy.stats import ttest_ind

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
        ceph_dir / posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_opto_2503.csv'),
        ceph_dir / posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_ephys_2504.csv'),

    ]
    pupil_pkl_paths = [Path(p) for p in [
        r"D:\pupil\opto_2503_stage3_up_90Hz_hpass00_lpass0.joblib",
        # r'X:\Dammy\mouse_pupillometry\pickles\ephys_2504_stage3_up_90Hz_hpass00_lpass0.joblib',
        ]]


    # Concatenate session topology dataframes
    session_topology_dfs = [pd.read_csv(sess_topology_path) for sess_topology_path in session_topology_paths]
    session_topology_df = pd.concat(session_topology_dfs, names=['cohort']).reset_index()
    all_sess_info = session_topology_df

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

    # only add sessions after last loaded date
    if len(sessions) > 0:
        last_loaded_date = int(sorted([extract_date(sess) for sess in list(sessions.keys())])[-1])
    else:
        last_loaded_date = 0
    
    # if len(sessions) > -1:
    #     # Merge pupil data from both cohorts
    #     all_pupil_data = [load_pupil_data(pkl_path) for pkl_path in pupil_pkl_paths]
    #     pupil_data_dict = all_pupil_data[0]
    #     for data in all_pupil_data[1:]:
    #         pupil_data_dict.update(data)
    
    #     for sessname in tqdm(list(pupil_data_dict.keys()), desc='processing sessions'):
    #         if int(extract_date(sessname)) < last_loaded_date:
    #         # if int(extract_date(sessname)) < 250900:
    #             continue
    #         if sessname not in sessions:
    #             sessions[sessname] = Session(sessname, ceph_dir)
    #         if sessions[sessname].td_df is not None and not not args.ow:
    #             continue
    #         print(sessname)
    #         print(f'initializing and processing td for {sessname}')
    #         init_pupil_td_obj(sessions, sessname, ceph_dir, all_sess_info, td_path_pattern, home_dir)
    #         process_pupil_td_data(sessions, sessname, {})
    
    #     # Process pupil object for each session
    #     for sessname in tqdm(list(sessions.keys()), desc='processing sessions'):
    #         if sessions[sessname].pupil_obj is not None and not args.ow:
    #             continue
    #         print(sessname)
    #         try:
    #             init_sess_pupil_obj(sessions, sessname, ceph_dir, all_sess_info, pupil_data_dict)
    #         except Exception as e:
    #             print(f'error processing {sessname}: {e}')
    #             sessions.pop(sessname)
    #             continue
    
    #         process_pupil_obj(sessions, sessname, alignmethod='w_soundcard', align_kwargs={'size_col': 'pupilsense_raddi_a_zscored'})
    
    #     # save
    #     joblib.dump(sessions, sess_pkl_path)

    # exit()
    # Get condition filters
    cond_filters = get_all_cond_filts()

    # Group trial data across sessions
    all_td_df = group_td_df_across_sessions(sessions, list(sessions.keys()))
    all_td_df['PatternPresentation_Rate'] = all_td_df['PatternPresentation_Rate'].round(1)
    good_trial_filt = 'n_since_last_Trial_Outcome <=5 '  # Optionally add a good trial filter

    abcd_sess = all_td_df.query('PatternID == "10;15;20;25" or PatternID=="10;12;14;16" and Session_Block==0 or Session_Block==2').index.get_level_values('sess').unique()
    abad_sess = all_td_df.query('PatternID == "10;15;10;25" and Session_Block==0 or Session_Block==2').index.get_level_values('sess').unique()

    # Define conditions to analyze
    conds2analyze = ['rare_prate', 'frequent_prate', 'rare', 'frequent']
    # Group pupil data by condition
    A_by_cond = {cond: group_pupil_across_sessions(
        sessions, list(sessions.keys()), 'A', cond, cond_filters)
        for cond in conds2analyze}
    X_by_cond = {cond: group_pupil_across_sessions(
        sessions, list(sessions.keys()), 'X', cond, cond_filters)
        for cond in ['hit_all', 'miss_all']}

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


    # Rare vs frequent analysis for each opto state
    rare_freq_figdir = ceph_dir / 'Dammy' / 'figures' / 'rare_freq_opto_all_pupilsense'
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
    all_rare_freq_sess = [sess for sess in sessions
                          if all([sess in A_by_cond[cond].index.get_level_values('sess')
                                  for cond in rare_freq_condnames])]

    # Loop through unique opto states and run analysis

    opto_batch_dict = {'opto_batch': opto_batch, 'control_batch': control_batch}
    for opto_state in ['light_off']:
        continue
        for batch_name, batch_mice in opto_batch_dict.items():
            opto_state_sesslist = [sess for sess in all_rare_freq_sess
                                   if sess in opto_sess_dict[opto_state] and sess.split('_')[0] in batch_mice
                                   and sess in abcd_sess]
            if len(opto_state_sesslist) == 0:
                print(f'{opto_state}: {batch_name} rare freq empty')
                continue
            run_pupil_cond_analysis(
                by_cond_dict=A_by_cond,
                sess_list=opto_state_sesslist,
                conditions=rare_freq_condnames,
                figdir=rare_freq_figdir,
                line_kwargs=rare_freq_line_kwargs,
                boxplot_kwargs=boxplot_kwargs,
                window_by_stim=(1, 2.9),
                smoothing_window=25,
                max_window_kwargs={'max_func': 'max'},
                group_name='sess',
                n_permutations=500,
                permutation_test=False,
                inset_max_diff=True,
                stats_dir=stats_dir,
                cluster_groupname='sess',
                fig_savename=f'rarefreq_{opto_state}_{batch_name}_ts.pdf',
                tex_name=f'rarefreq_{opto_state}__{batch_name}_data_vs_shuffled_no_filt.tex',
                ylabel="Δ pupil size",
                figsize=(2.5, 2),
                # ylim_ts=(-0.05, 0.35),
                # ylim_maxdiff=(-0.25, 0.25)
            )

    ### rare freq on abad sess ###
    opto_state = 'light_off'
    not_opto_states = ['light_on_masking_control', 'first_10_trial_freq_block_light_on',
                                       'learning_block_light_on', 'testing_block_light_on', 'light_on']
    not_opto_sess = sum([opto_sess_dict[state] for state in not_opto_states],[])
    abad_rarefreq_sess = [sess for sess in all_rare_freq_sess
                          if sess not in not_opto_sess and
                          sess in abad_sess.tolist()
                          ]

    run_pupil_cond_analysis(
        by_cond_dict=A_by_cond,
        sess_list=abad_rarefreq_sess,
        conditions=rare_freq_condnames,
        figdir=rare_freq_figdir,
        line_kwargs=rare_freq_line_kwargs,
        boxplot_kwargs=boxplot_kwargs,
        window_by_stim=(1, 2.9),
        smoothing_window=25,
        max_window_kwargs={'max_func': 'max'},
        group_name='sess',
        n_permutations=500,
        permutation_test=False,
        inset_max_diff=True,
        stats_dir=stats_dir,
        cluster_groupname='sess',
        fig_savename=f'rarefreq_{opto_state}_abad_ts.pdf',
        tex_name=f'rarefreq_{opto_state}_abad_data_vs_shuffled_no_filt.tex',
        ylabel="Δ pupil size",
        figsize=(2.5, 2),
        # ylim_ts=(-0.05, 0.35),
        # ylim_maxdiff=(-0.25, 0.25)
    )

    ### Normal vs deviant analysis for each opto state ###
    norm_dev_figdir = ceph_dir / 'Dammy' / 'figures' / 'norm_dev_opto_all_pupilsense'
    if not norm_dev_figdir.exists():
        norm_dev_figdir.mkdir()

    norm_dev_line_kwargs = {
        'normal': {'c': 'black', 'ls': '-'},
        'deviant_C': {'c': '#cd2727ff', 'ls': '-'}
    }
    norm_dev_cond_names = ['normal', 'deviant_C']
    abcd_sess = all_td_df.query('PatternID == "10;15;20;25" and Session_Block==2').index.get_level_values('sess').unique()

    for cond in norm_dev_cond_names:
        cond_filters[cond] = cond_filters[cond]+' & Trial_Outcome==1'

    for cond in norm_dev_cond_names:
        A_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()), 'A', cond, cond_filters)
        X_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()), 'X', cond, cond_filters)
    # A_by_cond['normal'] = group_pupil_across_sessions(sessions, list(sessions.keys()), 'A', 'normal_exp_midlate', cond_filters)
    add_name_to_response_dfs(A_by_cond)

    norm_dev_sess = [sess for sess in sessions
                     if all([sess in A_by_cond[cond].index.get_level_values('sess')
                             for cond in norm_dev_cond_names])
                     ]
    # opto_batch = [f'RS0{i}' for i in '123678']
    opto_batch = [f'RS0{i}' for i in '123']
    kinda_batch = [f'RS0{i}' for i in '678']
    all_batch = [f'RS0{i}' for i in '12368']
    control_batch = [f'RS0{i}' for i in '459']
    opto_batch_dict = {'opto_batch': opto_batch, 'control_batch': control_batch, 'kinda_batch': kinda_batch,
                       'all_batch': all_batch}


    # Loop through unique opto states and per batch for norm dev analysis
    for stim_name, stim_by_cond in zip(['A','X'][:1], [ A_by_cond,X_by_cond ]):
        continue
        for opto_state in ['light_off','learning_block_light_on'][:1]:
            for batch_name, batch_mice in opto_batch_dict.items():
            # for batch_name, batch_mice in [[e,e] for e in [f'RS0{i}' for i in '123456789']]:
                opto_state_sesslist = [sess for sess in norm_dev_sess
                                       if sess in opto_sess_dict[opto_state] and sess.split('_')[0] in batch_mice
                                       # and '2509' in sess
                                       # and sess in abcd_sess
                                       ]
                if len(opto_state_sesslist) == 0:
                    print(f'{opto_state}: {batch_name} normdev empty')
                    continue
                analysis = run_pupil_cond_analysis(
                    by_cond_dict=stim_by_cond,
                    sess_list=opto_state_sesslist,
                    conditions=norm_dev_cond_names,
                    figdir=norm_dev_figdir,
                    line_kwargs=norm_dev_line_kwargs,
                    boxplot_kwargs=boxplot_kwargs,
                    window_by_stim=(1, 2) if stim_name == 'A' else (0,1),
                    smoothing_window=100,
                    max_window_kwargs={'max_func': 'max'},
                    group_name=['name'],
                    fig_savename=f'normdev_{stim_name}_{opto_state}_{batch_name}_ts.pdf',
                    n_permutations=1,
                    permutation_test=False,
                    cluster_groupname='sess',
                    inset_max_diff=True,
                    stats_dir=stats_dir,
                    tex_name=f'normdev_{stim_name}_{opto_state}_{batch_name}_data_vs_shuffled_no_filt.tex',
                    ylabel="Δ pupil size",
                    figsize=(2.5, 2),
                    event_name='pattern' if stim_name == 'A' else stim_name,
                    # ylim_ts=(-0.05, 0.35),
                    # ylim_maxdiff=(-0.25, 0.25)
                )
                analysis.plots['ts_by_cond'][1].set_title(batch_name)
                analysis.plots['ts_by_cond'][0].show()


    ### normdev ABAD sess ###
    abad_sess = all_td_df.query('PatternID == "10;15;10;25" and Session_Block==2').index.get_level_values('sess').unique()

    abad_normdev_sess = [sess for sess in norm_dev_sess
                         if sess in abad_sess
                         and sess in opto_sess_dict['light_off']
                         ]
    # analysis = run_pupil_cond_analysis(
    #     by_cond_dict=A_by_cond,
    #     sess_list=abad_normdev_sess,
    #     conditions=norm_dev_cond_names,
    #     figdir=norm_dev_figdir,
    #     line_kwargs=norm_dev_line_kwargs,
    #     boxplot_kwargs=boxplot_kwargs,
    #     window_by_stim=(1, 2),
    #     smoothing_window=100,
    #     max_window_kwargs={'max_func': 'max'},
    #     group_name=['sess'],
    #     fig_savename=f'normdev_A_light_off_abad_ts.pdf',
    #     n_permutations=100,
    #     permutation_test=False,
    #     cluster_groupname='sess',
    #     inset_max_diff=True,
    #     stats_dir=stats_dir,
    #     tex_name=f'normdev_A_light_off_abad_data_vs_shuffled_no_filt.tex',
    #     ylabel="Δ pupil size",
    #     figsize=(2.5, 2),
    #     # ylim_ts=(-0.05, 0.35),
    #     # ylim_maxdiff=(-0.25, 0.25)
    # )

    ### Rare freq early late analysis for opto ###
    rare_freq_early_late_figdir = ceph_dir / 'Dammy' / 'figures' / 'rarefreq_early_late_opto_all_pupilsense'
    if not rare_freq_early_late_figdir.exists():
        rare_freq_early_late_figdir.mkdir()
    
    early_late_opto_states = ['light_off', 'first_10_trial_freq_block_light_on']
    cond_filters['frequent_late'] = f'frequent_prate_cumsum >= {25}  & frequent_block_num == 2'
    # rare_freq_early_late_conds = [f'frequent_prate_{ti}_block_1' for ti in np.arange(0,25,5)][0:5]
    rare_freq_early_late_conds = [f'frequent_prate_{ti}' for ti in np.arange(0,25,5)][0:5]

    # add to A_by_cond
    for cond in rare_freq_early_late_conds:
        A_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()), 'A', cond, cond_filters)
        X_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()), 'X', cond, cond_filters)

    cmap = plt.get_cmap('Blues_r')  # or any other colormap
    colors = [cmap(i / (len(rare_freq_early_late_conds))) for i in range(len(rare_freq_early_late_conds)+1)]
    rare_freq_early_late_line_kwargs = {
        cond: {'c': col, 'ls': '-'} for cond, col in zip(rare_freq_early_late_conds, colors)
    }
    all_rare_freq_early_late_sess = [sess for sess in sessions
                          if all([sess in A_by_cond[cond].index.get_level_values('sess')
                                  for cond in rare_freq_early_late_conds])]

    # all none opto early late
    non_opto_early_late_sess = [sess for sess in all_rare_freq_early_late_sess if all([sess not in opto_sess_dict[state] for state in
                                                                            ['light_on','first_10_trial_freq_block_light_on']])]
    analysis = run_pupil_cond_analysis(
        by_cond_dict=A_by_cond,
        sess_list=non_opto_early_late_sess,
        conditions=rare_freq_early_late_conds,
        figdir=rare_freq_early_late_figdir,
        line_kwargs=rare_freq_early_late_line_kwargs,
        boxplot_kwargs=boxplot_kwargs,
        group_name='sess',
        permutation_test=False,
        n_permutations=1,
        cluster_groupname=['sess'],
        cluster_comps=[['frequent_prate_20','frequent_prate_0'],['frequent_prate_20','frequent_prate_10']],
        stats_dir=stats_dir,
        fig_savename=f'rarefreq_early_late_w_ephys_ts.pdf',
        tex_name=f'rarefreq_early_late_w_ephys_data_vs_shuffled_no_filt.tex',
        ylabel="Δ pupil size",
        figsize=(1.6, 2),
        # ylim_ts=(-0.05, 0.35),
        # ylim_maxdiff=(-0.25, 0.25),
        inset_max_diff=False
    )

    analysis.plots['ts_by_cond'][1].set_xlim([-0.5, 2])
    analysis.plots['ts_by_cond'][0].show()
    analysis.plots['ts_by_cond'][0].savefig(rare_freq_early_late_figdir / f'rarefreq_early_late_w_ephys_ts.pdf')

    early_late_diff_by_state = {}
    # Loop through unique opto states and per batch for rare freq early late analysis
    for opto_state in early_late_opto_states:
        # continue
        # for i, (batch_name, batch_mice) in enumerate(opto_batch_dict.items()):
        for i, batch_name in enumerate(['all_batch']):
        # for i, batch_name in enumerate([f'RS0{i}' for i in range(10)]):
            batch_mice = opto_batch_dict[batch_name]
            # batch_mice = [batch_name]

            opto_state_sesslist = [sess for sess in all_rare_freq_early_late_sess
                                   if sess in opto_sess_dict[opto_state] and sess.split('_')[0] in batch_mice]
            if len(opto_state_sesslist) == 0:
                print(f'{opto_state}: {batch_name} rare freq early late empty')
                continue
            analysis = run_pupil_cond_analysis(
                by_cond_dict=A_by_cond,
                sess_list=opto_state_sesslist,
                conditions=rare_freq_early_late_conds[2:],
                figdir=rare_freq_early_late_figdir,
                line_kwargs=rare_freq_early_late_line_kwargs,
                boxplot_kwargs=boxplot_kwargs,
                window_by_stim=(1, 2),
                smoothing_window=5,
                max_window_kwargs={'max_func': 'max'},
                group_name='sess',
                permutation_test=False,
                n_permutations=1,
                cluster_groupname=['sess'],
                # cluster_comps=[['frequent_prate_20', 'frequent_prate_0'], ['frequent_prate_20', 'frequent_prate_10']],
                stats_dir=stats_dir,
                fig_savename=f'rarefreq_early_late_{opto_state}_{batch_name}_ts.pdf',
                tex_name=f'rarefreq_early_late_{opto_state}_{batch_name}_data_vs_shuffled.tex',
                ylabel="Δ pupil size",
                figsize=(1.6, 2),
                ylim_ts=(-0.05, 0.35),
                # ylim_maxdiff=(-0.25, 0.25),
                inset_max_diff=False
            )
            # analysis.plots['ts_by_cond'][1].legend(loc='upper left')
            analysis.plots['ts_by_cond'][1].set_xlim([-0.5,2])
            analysis.plots['ts_by_cond'][1].set_title(f'{opto_state}: {batch_name}')
            analysis.plots['ts_by_cond'][0].show()
            analysis.plots['ts_by_cond'][0].savefig(rare_freq_early_late_figdir/
                                                    f'rarefreq_early_late_{opto_state}_{batch_name}_ts.pdf')

            # diff early vs late
            pupil_by_cond = [analysis.by_cond[cond].groupby(['sess','name']).mean().loc[:,0.5:1.5].mean(axis=1)
                             for cond in ['frequent_prate_20','frequent_prate_10']]
            early_late_diff = (pupil_by_cond[-1]-pupil_by_cond[0]).groupby('name').mean()
            early_late_diff_by_state[f'{opto_state}_{batch_name}'] = early_late_diff


    # plot early vs late diff across opto states
    early_late_diff_by_state['first_10_trial_freq_block_light_on_all_batch'] = early_late_diff_by_state['first_10_trial_freq_block_light_on_all_batch'].drop('RS06')

    # ttest between early and late for each opto state
    ttests_early_late = ttest_ind(early_late_diff_by_state['light_off_all_batch'],
                                  early_late_diff_by_state[f'{early_late_opto_states[1]}_all_batch'],
                                  alternative='less')

    save_stats_to_tex(ttests_early_late, stats_dir /f'rarefreq_early_vs_late.tex')
    if len(early_late_diff_by_state) > 0:
        fig, ax = plt.subplots(1,1, figsize=(1.5,2))

        ax.boxplot(
            [e for e in list(early_late_diff_by_state.values())], labels=['no inh', 'inh'],
            **boxplot_kwargs
        )
        # add stats
        y, h, col = np.percentile(np.hstack(list(early_late_diff_by_state.values())),97.5) + 0.05 + i*0.05, 0.02, 'k'
        x1, x2 = [1,2]
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.5, c=col)
        ax.text((x1+x2)*.5, y+h, f"{'*' if ttests_early_late.pvalue<0.05 else 'ns'}", ha='center', va='bottom', color=col)
        ax.set_ylabel('Max pupil diff (frequent - rare)')
        ax.set_title(f"Opto batch (n={len(early_late_diff_by_state['light_off_all_batch'])})")
        format_axis(ax)
        fig.tight_layout()
        fig.set_size_inches((1.1,2))
        fig.show()
        fig.savefig(rare_freq_early_late_figdir / 'rarefreq_early_late_diff.pdf')

        # plot rare freq musc none by animal
        rare_freq_diff_ts_plot = plt.subplots()
        rare_freq_by_drug_by_cond = {}

        opto_state_sesslist = {opto_state: [sess for sess in norm_dev_sess
                                            if sess in opto_sess_dict[opto_state] and sess.split('_')[0]
                                            in opto_batch_dict['opto_batch']]
                               for opto_state in ['light_off','learning_block_light_on']}

        resps_by_cond = {'_'.join([e, opto_state]): A_by_cond[e].T.rolling(window=25).mean().T.loc[
                            A_by_cond[e].index.get_level_values('sess').isin(o_sess), 0.5:1.5].max(axis=1).groupby(['sess','name']).mean()
                         for e in norm_dev_cond_names
                         for opto_state, o_sess in opto_state_sesslist.items()}
        resp_by_opto_by_cond_df = pd.DataFrame.from_dict(resps_by_cond)


    # plot patt response for recent sessions
    patt_resps_obj = PupilCondAnalysis(A_by_cond,['rare','frequent'],[sess for sess in opto_sess_dict['first_10_trial_freq_block_light_on']
                                                                     if int(extract_date(sess)) >=250923])
    patt_resps_obj.plot_ts_by_cond(rare_freq_line_kwargs)
    patt_resps_obj.plots['ts_by_cond'][0].show()