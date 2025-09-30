import argparse
from copy import copy
from pathlib import Path
import platform
import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_ind
from tqdm import tqdm
import yaml

from save_utils import save_stats_to_tex
from behviour_analysis_funcs import get_all_cond_filts, group_td_df_across_sessions
from pupil_analysis_funcs import run_pupil_cond_analysis, PupilCondAnalysis, load_pupil_sess_pkl, init_pupil_td_obj, process_pupil_td_data, init_sess_pupil_obj, process_pupil_obj, group_pupil_across_sessions, plot_pupil_ts_by_cond
from io_utils import posix_from_win, load_pupil_data
from plot_funcs import format_axis

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('--ow',default=0)
    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])
    home_dir = Path(config[f'home_dir_{sys_os}'])
    stats_dir = ceph_dir/ posix_from_win(r'X:\Dammy\stats')
    td_path_pattern = 'data/Hilde/<name>/TrialData'

    plt.style.use('figure_stylesheet.mplstyle')


    pupil_pkl_paths = [ceph_dir / posix_from_win(p) for p in [
        r'X:\Dammy\mouse_pupillometry\pickles\aligned_fam_human_fam_w_pupilsense_fam_2d_90Hz_hpass00_lpass0_TOM.pkl',
        r'X:\Dammy\mouse_pupillometry\pickles\aligned_normdev_human_normdev_w_pupilsense_normdev_2d_90Hz_hpass00_lpass0_TOM.pkl',

    ]]    

    all_pupil_data = [load_pupil_data(pkl_path) for pkl_path in pupil_pkl_paths]
    pupil_data_dict = {}
    for data in all_pupil_data:
        pupil_data_dict.update(data)
    
    all_human_td_files = list(Path(r'H:\data\Hilde\Human\TrialData').iterdir())
    all_sess_info_dict = {}
    for sess in pupil_data_dict:
        name, date = sess.split('_')
        date = int(date)
        sess_tdfile = [td for td in all_human_td_files if
                       name in td.stem and str(date) in td.stem]
        if len(sess_tdfile) == 0:
            print(f'No td file found for {sess}')
            continue
        elif len(sess_tdfile) > 1:
            print(f'Multiple td files found for {sess}, using the first one: {sess_tdfile[0]}')
        else:
            sess_tdfile = sess_tdfile[0]
        all_sess_info_dict[sess] = {
            'name': name,
            'date': date,
            'tdata_file':sess_tdfile,
            'sess_order': 'main',
            }
        
    all_sess_info = pd.DataFrame(all_sess_info_dict).T.reset_index(drop=True)

    sess_pkl_path = Path(r'D:') / 'human_fam_sess_dicts_no_filt_new_scripts.joblib'
    if not args.ow:
        sessions = load_pupil_sess_pkl(sess_pkl_path)
    else:
        sessions = {}
            
    loaded_sess_dict_sig = copy(tuple(sorted(sessions.items())))

    existing_sessions = list(sessions.keys())

    window = [-1, 3]
    drug_sess_dict = {}
    for sessname in tqdm(list(pupil_data_dict.keys()), desc='processing sessions'):
        print(sessname)

        print(f'initializing and processing td for {sessname}')
        init_pupil_td_obj(sessions, sessname, ceph_dir, all_sess_info, td_path_pattern, home_dir)
        process_pupil_td_data(sessions,sessname,drug_sess_dict)

    for sessname in tqdm(list(sessions.keys()), desc='processing sessions'):
        if sessions[sessname].pupil_obj is not None and not args.ow:
            continue
        print(sessname)

        init_sess_pupil_obj(sessions, sessname, ceph_dir, all_sess_info, pupil_data_dict,force_sync=False)
        process_pupil_obj(sessions,sessname,alignmethod='w_td_df',align_kwargs={'size_col':'diameter_2d_zscored'})

    # save sessions
    if tuple(sorted(sessions.items())) != loaded_sess_dict_sig or args.ow:
        print('saving sessions')
        joblib.dump(sessions, sess_pkl_path.with_suffix('.joblib'))

    cond_filters = get_all_cond_filts()
    # good_trial_filt = 'n_since_last_Trial_Outcome <=5 & lick_in_patt==0'
    good_trial_filt = 'n_since_last_Trial_Outcome <=5 '

    all_td_df = group_td_df_across_sessions(sessions,list(sessions.keys()),)

    # [cond_filters.update({k: ' & '.join([v, good_trial_filt])}) for k, v in cond_filters.items()]

    A_by_cond = {cond: group_pupil_across_sessions(sessions,list(sessions.keys()),'A',cond,cond_filters,)
                 for cond in ['rare_human','frequent_human','normal','normal_exp_midlate','deviant_C_human']}

    A_by_cond['rare'] = A_by_cond['rare_human']
    A_by_cond['frequent'] = A_by_cond['frequent_human']
    # A_by_cond['normal'] = A_by_cond['normal_exp_midlate']
    A_by_cond['deviant_C'] = A_by_cond['deviant_C_human']
    rare_freq_condnames = ['frequent', 'rare']

    X_by_cond = {cond: group_pupil_across_sessions(sessions,list(sessions.keys()),'X',cond,cond_filters)
                 for cond in ['hit_all']}


    # --- Rare vs Frequent Analysis ---
    rare_freq_figdir = ceph_dir/posix_from_win(r'X:\Dammy\figures')/f'rare_freq_human_no_filt'

    # Set up line colors and styles for rare/frequent conditions
    rare_freq_line_kwargs = {
        'rare': {'c': '#00cbfcff', 'ls': '-'},
        'frequent': {'c': '#1530e9ff', 'ls': '-'}
    }
    # Find sessions with both recent and distant data
    rare_freq_sessions = np.intersect1d(
        *[A_by_cond[cond].index.get_level_values('sess').values for cond in ['rare', 'frequent']])
    rare_freq_sessions = [sess for sess in sessions
                          if all([sess in A_by_cond[cond].index.get_level_values('sess')
                                  for cond in rare_freq_condnames])]
    if not rare_freq_figdir.is_dir():
        rare_freq_figdir.mkdir()

    # plot rare vs frequent using PupilCondAnalysis
    rare_freq_figdir = ceph_dir / 'Dammy' / 'figures' / 'rare_freq_ctrl_opto_no_filt'
    if not rare_freq_figdir.is_dir():
        rare_freq_figdir.mkdir()

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

    run_pupil_cond_analysis(
        by_cond_dict=A_by_cond,
        sess_list=rare_freq_sessions,
        conditions=['frequent','rare'],
        figdir=rare_freq_figdir,
        line_kwargs=rare_freq_line_kwargs,
        boxplot_kwargs=boxplot_kwargs,
        window_by_stim=(1, 2.9),
        smoothing_window=25,
        max_window_kwargs={'max_func': 'max'},
        group_name='name',
        n_permutations=500,
        permutation_test=False,
        inset_max_diff=True,
        stats_dir=stats_dir,
        cluster_groupname='sess',
        fig_savename=f'rarefreq_human_ts.pdf',
        tex_name=f'rarefreq_human_data_vs_shuffled_no_filt.tex',
        ylabel="Δ pupil size",
        figsize=(2.5, 2),
        # ylim_ts=(-0.05, 0.35),
        # ylim_maxdiff=(-0.25, 0.25)
        event_name='pattern',)
    

    # --- Normal vs deviant analysis
    dev_figdir = ceph_dir/posix_from_win(r'X:\Dammy\figures')/f'normal_dev_human_no_filt'
    if not dev_figdir.is_dir():
        dev_figdir.mkdir()
    dev_cond_names = ['normal', 'deviant_C']

    norm_dev_line_kwargs = {
            'normal': {'c': 'black', 'ls': '-'},
            'deviant_C': {'c': '#cd2727ff', 'ls': '-'}
        }
    normdev_batch = ['Human28','Human29','Human30','Human31', 'Human32']
    norm_dev_sess = [sess for sess in sessions
                     if all([sess in A_by_cond[cond].index.get_level_values('sess')
                             for cond in dev_cond_names])
                             and sess.split('_')[0] in normdev_batch
                     ]
    norm_dev_sess = [sess for sess in norm_dev_sess
                         if all(A_by_cond[cond].xs(sess, level='sess').shape[0] > 1 for cond in ['normal', 'deviant_C'])]
    
    run_pupil_cond_analysis(
        by_cond_dict=A_by_cond,
        sess_list=norm_dev_sess,
        conditions=dev_cond_names,
        figdir=dev_figdir,
        line_kwargs=norm_dev_line_kwargs,
        boxplot_kwargs=boxplot_kwargs,
        window_by_stim=(1, 2.9),
        smoothing_window=25,
        max_window_kwargs={'max_func': 'max'},
        group_name='name',
        n_permutations=500,
        permutation_test=False,
        inset_max_diff=True,
        stats_dir=stats_dir,
        cluster_groupname='sess',
        fig_savename=f'normdev_human_ts.pdf',
        tex_name=f'normdev_human_data_vs_shuffled_no_filt.tex',
        ylabel="Δ pupil size",
        figsize=(2.5, 2),
        # cluster_comps=[[1,0]],
        # ylim_ts=(-0.05, 0.35),
        # ylim_maxdiff=(-0.25, 0.25)
        event_name='pattern',)

    # plot X response on correct trials

    fig1_dir = ceph_dir / 'Dammy' / 'figures' / 'supp_human_no_filt'
    if not fig1_dir.is_dir():
        fig1_dir.mkdir()

    if 'hit_all' not in X_by_cond:
        X_by_cond['hit_all'] = group_pupil_across_sessions(
            sessions, list(sessions.keys()), 'X', 'hit_all', cond_filters
        )
    x_hit_plot = plot_pupil_ts_by_cond(X_by_cond, ['hit_all'],)
    x_hit_plot[1].set_title('X response (correct trials)')
    x_hit_plot[1].axvline(0, c='k', ls='--')
    x_hit_plot[1].locator_params(axis='both', nbins=4)
    x_hit_plot[0].set_layout_engine('tight')
    x_hit_plot[0].show()
    x_hit_plot[0].savefig(fig1_dir / 'x_hit_plot.pdf')

    # beh
 
    perf_patt_non_patt = all_td_df.groupby(['name', 'Tone_Position'])['Trial_Outcome'].mean().unstack().dropna(axis=0)[[1, 0]].values

    all_sess_perf_bar_plot = plt.subplots()
    patt_nonpatt_cols = ['dimgray','indigo']

    all_sess_perf_bar_plot[1].bar(['pattern', 'non-pattern'], perf_patt_non_patt.mean(axis=0),
                                  ec='black',fc='w',lw=0.5,width=0.4)
    [all_sess_perf_bar_plot[1].scatter([cond] * len(cond_data), cond_data, 
                                       facecolor=c, alpha=0.5, lw=0.01)
     for cond, cond_data, c in zip(['pattern', 'non-pattern'], perf_patt_non_patt.T, patt_nonpatt_cols)]
    [all_sess_perf_bar_plot[1].plot(['pattern', 'non-pattern'], a_data, lw=0.25, color='gray')
     for a_data in perf_patt_non_patt]
    all_sess_perf_bar_plot[1].set_ylim(.25, 0.9)
    format_axis(all_sess_perf_bar_plot[1])
    all_sess_perf_bar_plot[1].set_ylabel('Hit rate')
    all_sess_perf_bar_plot[0].set_size_inches(1.25, 1.5)
    all_sess_perf_bar_plot[0].show()
    all_sess_perf_bar_plot[0].savefig(fig1_dir / 'hum_perf_by_tone_position_barplot.pdf')

    # ttest
    hum_beh_perf_ttest = ttest_ind(perf_patt_non_patt[0], perf_patt_non_patt[1], alternative='greater')
    print(hum_beh_perf_ttest)
    save_stats_to_tex(hum_beh_perf_ttest,stats_dir/'human_beh_perf_ttest.tex')