import platform
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from behviour_analysis_funcs import group_td_df_across_sessions, get_all_cond_filts, get_drug_dates, \
    parse_perturbation_log, get_perturbation_sessions
from io_utils import posix_from_win, load_pupil_data
from plot_funcs import format_axis
from pupil_analysis_funcs import PupilCondAnalysis, init_pupil_td_obj, load_pupil_sess_pkl, process_pupil_td_data, \
    init_sess_pupil_obj, process_pupil_obj, add_early_late_to_by_cond, group_pupil_across_sessions
from scipy.stats import ttest_ind

from save_utils import save_stats_to_tex


if __name__ == '__main__':
    import argparse, yaml
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('--ow', default=0, type=int)
    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])
    home_dir = Path(config[f'home_dir_{sys_os}'])
    stats_dir = ceph_dir/ posix_from_win(r'X:\Dammy\stats')

    td_path_pattern = 'data/Hilde/<name>/TrialData'

    plt.style.use('figure_stylesheet.mplstyle')

    session_topology_paths  = [
        ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_ephys_2504.csv'),
    ]
    pupil_pkl_paths = [ceph_dir / posix_from_win(p) for p in [
        r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_musc_2406_v2409_walt_fam_2d_90Hz_hpass0_lpass0_TOM.pkl'
    ]]

    all_pupil_data = [load_pupil_data(pkl_path) for pkl_path in pupil_pkl_paths]
    pupil_data_dict = all_pupil_data[0]
    pupil_data_dict = {**pupil_data_dict, **all_pupil_data[1]}
    session_topology_dfs = [pd.read_csv(sess_topology_path) for sess_topology_path in session_topology_paths]
    [pd.concat({cohort_i: df}, names=['cohort']) for cohort_i, df in enumerate(session_topology_dfs)]
    session_topology_df = pd.concat(session_topology_dfs, names=['cohort']).reset_index()
    all_sess_info = session_topology_df


    sess_pkl_path = Path(r'D:') / 'musc_sept23.pkl'
    sessions = load_pupil_sess_pkl(sess_pkl_path)
    loaded_sess_dict_sig = copy(tuple(sorted(sessions.items())))

    window = [-1, 3]
    drug_sess_dicts = []

    # cohort_config_paths = [r"H:\gd_analysis\config\musc_sept23.yaml",r"H:\gd_analysis\config\musc_2406.yaml"]
    # for ci,cohort_config_path in enumerate(cohort_config_paths):
    #     drug_sess_dict = {}
    #     with open(cohort_config_path, 'r') as file:
    #         cohort_config = yaml.safe_load(file)
    #     get_drug_dates(cohort_config,session_topology_dfs[ci],drug_sess_dict, date_start=None, date_end=240830)  # 240717
    #     drug_sess_dicts.append(copy(drug_sess_dict))

    # for d in drug_sess_dicts[0]:
    #     drug_sess_dicts[0][d] = drug_sess_dicts[0][d]+ drug_sess_dicts[1][d]
    # drug_sess_dicts = drug_sess_dicts[0]
    #
    # drug_dfs = [[[*sess.split('_')[::-1],drug] for sess in drug_sess_dicts[drug]] for drug in drug_sess_dicts]
    # drug_dfs = sum(drug_dfs,[])
    # mega_drug_df = pd.DataFrame.from_records(drug_dfs,columns=['date','name','drug'])
    # mega_drug_df.sort_values(by=['date'])
    # mega_drug_df.to_csv('musc_df.csv')

    drug_log_xlsx_path = r"X:\Dammy\Xdetection_mouse_hf_test\metadata\animal_log.xlsx"
    perturb_log = parse_perturbation_log(drug_log_xlsx_path)
    drug_sess_dict = get_perturbation_sessions(perturb_log['infusion_log'],'drug')


    for sessname in tqdm(list(pupil_data_dict.keys()), desc='processing sessions'):
        print(sessname)

        print(f'initializing and processing td for {sessname}')
        init_pupil_td_obj(sessions, sessname, ceph_dir, all_sess_info, td_path_pattern, home_dir)
        process_pupil_td_data(sessions,sessname, drug_sess_dict)

    for sessname in tqdm(list(sessions.keys()), desc='processing sessions'):
        if sessions[sessname].pupil_obj is not None and not args.ow:
            continue
        print(sessname)

        init_sess_pupil_obj(sessions, sessname, ceph_dir, all_sess_info, pupil_data_dict)
        # process_pupil_obj(sessions,sessname,alignmethod='w_soundcard',align_kwargs={'size_col':'pupilsense_raddi_a_zscored'})
        process_pupil_obj(sessions,sessname,alignmethod='w_soundcard',align_kwargs={'size_col':'dlc_radii_a_zscored'})


    cond_filters = get_all_cond_filts()

    all_td_df = group_td_df_across_sessions(sessions, list(sessions.keys()),)
    all_td_df['PatternPresentation_Rate'] = all_td_df['PatternPresentation_Rate'].round(1)
    # Optionally add a good trial filter here
    good_trial_filt = 'n_since_last_Trial_Outcome <=5 '

    # Group by normal and deviant
    # conds2analyze = ['normal_exp', 'deviant_C']
    # conds2analyze = ['normal_exp', 'rare_prate','frequent_prate','rare','frequent']
    conds2analyze = ['rare_prate','frequent_prate','rare','frequent']
    A_by_cond = {cond: group_pupil_across_sessions(
        sessions, list(sessions.keys()), 'A', cond, cond_filters)
        for cond in conds2analyze}

    X_by_cond = {cond: group_pupil_across_sessions(
        sessions, list(sessions.keys()), 'X', cond, cond_filters)
        for cond in ['hit_all','miss_all']}

    # Use helper to get early/middle/late filters
    # rare_freq_early_late_dict = {'early': '<=10', 'middle': '>15 & <col> <=25', 'late': '> <col>.max()-10'}
    # rare_freq_early_late_dict = {'early': '<=7', 'middle': '>9 & <col> <=16', 'late': '> <col>.max()-15 & frequent_block_num==1'}
    rare_freq_early_late_dict = {'early': '<=8', 'middle': '>9 & <col> <=17', 'late': '>19 & frequent_block_num==1'}
    add_early_late_to_by_cond(A_by_cond,['frequent_prate'],rare_freq_early_late_dict,sessions,'A',cond_filters,)

    # [A_by_cond.update(
    #     {cond: group_pupil_across_sessions(sessions, list(sessions.keys()), 'A', cond, td_df_query=query)})
    #     for cond, query in filters_early_late.items()]

    # Plot X resp
    X_resp = PupilCondAnalysis(
        by_cond_dict=X_by_cond,
        conditions=['hit_all', 'miss_all'],
    )
    X_resp.plot_ts_by_cond()
    X_resp.plots['ts_by_cond'][0].show()

    # rare freq analysis
    # plot rare vs frequent using PupilCondAnalysis
    rare_freq_figdir = ceph_dir / 'Dammy' / 'figures' / 'rare_freq_ctrl_musc_sep23_no_filt'
    if not rare_freq_figdir.is_dir():
        rare_freq_figdir.mkdir()

    none_drug_sess = [sess for sess in sessions
                      if not any([d in sess for e in ['muscimol','saline'] for d in drug_sess_dicts[e] ])]
    rare_freq_sesslist = [sess for sess in none_drug_sess
                          if all([sess in A_by_cond[cond].index.get_level_values('sess')
                                  for cond in ['frequent_prate', 'rare_prate']])
                          and 'DO76' not in sess]

    musc_drug_sess = [sess for sess in sessions
                      if any([d in sess for e in ['muscimol',] for d in drug_sess_dicts[e] ])
                      and 'DO76' not in sess]

    musc_rare_freq_sesslist = rare_freq_sesslist = [sess for sess in musc_drug_sess
                          if all([sess in A_by_cond[cond].index.get_level_values('sess')
                                  for cond in ['frequent_prate', 'rare_prate']])]

    rarefreq_analysis = PupilCondAnalysis(
        by_cond_dict=A_by_cond,sess_list=musc_rare_freq_sesslist,
        conditions=['frequent_prate', 'rare_prate'],
    )
    rarefreq_analysis.plot_ts_by_cond(group_name=['name'])
    rarefreq_analysis.plots['ts_by_cond'][0].show()

    # rarefreq_analysis.plot_diff_ts_by_cond()
    # rarefreq_analysis.plots['ts_by_cond'][0].show()

    all_boxplot = plt.subplots()

    rarefreq_analysis.plot_max_diff(mean=np.min, window_by_stim=(1,2.9 ), permutation_test=True,
                                    group_name=['name'],
                                    n_permutations=200,
                                    plot_kwargs={'showfliers': False, 'widths': 0.3},)
    rarefreq_analysis.plots['max_diff'][0].show()

    print(rarefreq_analysis.ttest_max_diff(['rare_prate-frequent_prate', 'data'], ['rare_prate-frequent_prate', 'shuffled']))

    # Use PupilCondAnalysis for early/late plotting
    freq_early_late_figdir = ceph_dir / 'Dammy' / 'figures' / 'freq_early_late'
    if not freq_early_late_figdir.is_dir():
        freq_early_late_figdir.mkdir()

    freq_early_late_analysis = PupilCondAnalysis(
        by_cond_dict=A_by_cond,
        conditions=['rare', 'frequent_prate_early', 'frequent_prate_middle', 'frequent_prate_late'],
    )
    freq_early_late_analysis.plot_ts_by_cond()
    freq_early_late_analysis.plot_max_diff(
        diff_conditions=['rare', 'frequent_prate_early', 'frequent_prate_middle', 'frequent_prate_late' ],
        window_by_stim=(1.5, 2.5), mean=np.max, plot_kwargs={'showfliers': False, 'widths': 0.3},
        group_name='name', permutation_test=False, scatter=True,
    )
    # format and show plot
    # ts plot
    format_axis(freq_early_late_analysis.plots['ts_by_cond'][1])
    freq_early_late_analysis.plots['ts_by_cond'][0].show()
    # max diff
    format_axis(freq_early_late_analysis.plots['max_diff'][1])
    freq_early_late_analysis.plots['max_diff'][0].show()
    # max diff scatter

    # ttest diff between early and late
    print(freq_early_late_analysis.ttest_max_diff(
        ['frequent_prate_early-rare', 'data'],
        ['frequent_prate_late-rare', 'data'],
        alternative='two-sided'
    ))


    # Find sessions with both normal and deviant_C data using PupilCondAnalysis
    norm_dev_figdir = ceph_dir / 'Dammy' / 'figures' / 'norm_dev_opto_no_filt'
    if not norm_dev_figdir.is_dir():
        norm_dev_figdir.mkdir(parents=True)

    normdev_analysis = PupilCondAnalysis(
        by_cond_dict=A_by_cond,
        conditions=conds2analyze,
    )

    normdev_analysis.plot_ts_by_cond()
    normdev_analysis.plot_diff_ts_by_cond()
    normdev_analysis.plot_max_diff(window_by_stim=(1, 2), mean=np.max, permutation_test=True)
    normdev_analysis.ttest_max_diff()
    normdev_analysis.save_ttest_to_tex()


    # [A_by_cond.update({cond: group_pupil_across_sessions(sessions, list(sessions.keys()), 'A-1', cond_name=cond,
    #                                                      cond_filters=cond_filters)})
    #  for cond in ['dev_ABCD1', 'dev_ABBA1']]