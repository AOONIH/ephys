import platform
from copy import copy
from itertools import combinations

import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from aggregate_psth_analysis import padded_rolling_mean

import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from mne.stats import permutation_cluster_test

from behviour_analysis_funcs import group_td_df_across_sessions, get_all_cond_filts, get_drug_dates, \
    parse_perturbation_log, get_perturbation_sessions
from io_utils import posix_from_win, load_pupil_data
from plot_funcs import format_axis
from pupil_analysis_funcs import PupilCondAnalysis, init_pupil_td_obj, load_pupil_sess_pkl, process_pupil_td_data, \
    init_sess_pupil_obj, process_pupil_obj, add_early_late_to_by_cond, group_pupil_across_sessions, \
    run_pupil_cond_analysis
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
        ceph_dir / posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_musc_sept23.csv'),
        ceph_dir / posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_musc_2406.csv'),
    ]
    pupil_pkl_paths = [ceph_dir / posix_from_win(p) for p in [
        r'X:\Dammy\mouse_pupillometry\pickles\musc_sept23_stage3_up_90Hz_hpass00_lpass0.joblib',
        r'X:\Dammy\mouse_pupillometry\pickles\musc_2406_stage3_up_90Hz_hpass0_lpass0.joblib',
    ]]

    # Load session pickle
    sess_pkl_path = Path(r'D:') / 'musc_analysis.joblib'
    sessions: dict = load_pupil_sess_pkl(sess_pkl_path)
    loaded_sess_dict_sig = copy(tuple(sorted(sessions.items())))


    # Concatenate session topology dataframes
    session_topology_dfs = [pd.read_csv(sess_topology_path) for sess_topology_path in session_topology_paths]
    session_topology_df = pd.concat(session_topology_dfs, names=['cohort']).reset_index()
    all_sess_info = session_topology_df


    window = [-1, 3]

    # Parse drug/perturbation log and get drug session dictionary
    drug_log_xlsx_path = r"X:\Dammy\Xdetection_mouse_hf_test\metadata\animal_log.xlsx"
    perturb_log = parse_perturbation_log(drug_log_xlsx_path)
    drug_sess_dict = get_perturbation_sessions(perturb_log['infusion_log'], 'drug')

    # old drug_sess_dict
    with open(r'H:\gd_analysis\config\musc_sept23.yaml', 'r') as file:
        musc_sept23_config = yaml.safe_load(file)


    exclude_dates = ['240710',
                     '240712',
                     '240716',
                     '240717',
                     '240726']
    none_sess = drug_sess_dict['none']
    musc_sess = drug_sess_dict['muscimol']
    saline_sess = drug_sess_dict['saline']
    none_dates = list(set([sess.split('_')[1] for sess in none_sess]))
    musc_dates = list(set([sess.split('_')[1] for sess in musc_sess]))
    saline_dates = list(set([sess.split('_')[1] for sess in saline_sess]))
    none_dates_dt, musc_dates_dt, saline_dates_dt = [[datetime.strptime(d,'%y%m%d') for d in dates]
                                                     for dates in [none_dates, musc_dates,saline_dates]]
    none_dates,saline_dates = [[d for d, d_dt in zip(date_set, dt_set)
                                if all(d_dt - np.array(musc_dates_dt) != np.timedelta64(1, 'D'))]
                               for date_set,dt_set in zip([none_dates, saline_dates], [none_dates_dt, saline_dates_dt])]
    # drug_sess_dict['none'] = [sess for sess in drug_sess_dict['none'] if sess.split('_')[1] in none_dates
    #                           and sess.split('_')[1] not in exclude_dates]
    drug_sess_dict['muscimol'] = [sess for sess in drug_sess_dict['muscimol']
                                  if sess.split('_')[1] not in exclude_dates]
    drug_sess_dict['saline'] = [sess for sess in drug_sess_dict['saline'] if sess.split('_')[1] in saline_dates
                              and sess.split('_')[1] not in exclude_dates]

    # make sure no overlapping sessions
    np.intersect1d(drug_sess_dict['none'],drug_sess_dict['muscimol'])

    # Process trial data for each session
    if len(sessions) == 0:
        # Merge pupil data from both cohorts
        all_pupil_data = [load_pupil_data(pkl_path) for pkl_path in pupil_pkl_paths]
        pupil_data_dict = {**all_pupil_data[0], **all_pupil_data[1]}

        for sessname in tqdm(list(pupil_data_dict.keys()), desc='processing sessions'):
            if sessions[sessname].td_df is not None and not not args.ow:
                continue
            print(sessname)
            print(f'initializing and processing td for {sessname}')
            init_pupil_td_obj(sessions, sessname, ceph_dir, all_sess_info, td_path_pattern, home_dir)
            process_pupil_td_data(sessions, sessname, drug_sess_dict)

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

    # Rare vs frequent analysis
    rare_freq_figdir = ceph_dir / 'Dammy' / 'figures' / 'rare_freq_ctrl_all_musc_no_filt'
    if not rare_freq_figdir.exists():
        rare_freq_figdir.mkdir(exist_ok=True)

    # Find sessions with no drug
    none_drug_sess = [sess for sess in sessions
                      if not any([d in sess for e in ['muscimol', 'saline'] for d in drug_sess_dict.get(e, [])])]
    rare_freq_sesslist = [sess for sess in none_drug_sess
                          if all([sess in A_by_cond[cond].index.get_level_values('sess')
                                  for cond in ['frequent_prate', 'rare_prate']])
                          and 'DO76' not in sess]

    # Find sessions with muscimol drug
    musc_drug_sess = [sess for sess in sessions
                      if any([d in sess for e in ['muscimol'] for d in drug_sess_dict.get(e, [])])
                      and 'DO76' not in sess]
    musc_rare_freq_sesslist = [sess for sess in musc_drug_sess
                               if all([sess in A_by_cond[cond].index.get_level_values('sess')
                                       for cond in ['frequent_prate', 'rare_prate']])]
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

    # Analyze rare vs frequent for each drug sessions
    rare_freq_analysis_by_drug = {}
    rare_freq_condnames = ['frequent', 'rare']
    all_rare_freq_sess = [sess for sess in sessions
                          if all([sess in A_by_cond[cond].index.get_level_values('sess')
                                  for cond in rare_freq_condnames])]
    rare_freq_line_kwargs = {
        'rare': {'c': '#00cbfcff', 'ls': '-'},
        'frequent': {'c': '#1530e9ff', 'ls': '-'}
    }
    skip_old =True
    if not skip_old:
        for drug in ['muscimol','none','saline']:
            drug_rare_freq_sesslist = [sess for sess in all_rare_freq_sess
                                       if sess in drug_sess_dict[drug]]
            rare_freq_analysis_by_drug[drug] = PupilCondAnalysis(
                by_cond_dict=A_by_cond, sess_list=drug_rare_freq_sesslist,
                conditions=['frequent', 'rare'],
            )
            rare_freq_analysis_by_drug[drug].filter_sess_list()
            rare_freq_analysis_by_drug[drug].plot_ts_by_cond(cond_line_kwargs=rare_freq_line_kwargs)
            # create inset plot for max diff
            # axinset = inset_axes(rare_freq_analysis_by_drug[drug].plots['ts_by_cond'][1],
            #                      width="20%", height="40%", loc="upper right", borderpad=2)
            axinset = rare_freq_analysis_by_drug[drug].plots['ts_by_cond'][0].add_axes([0.8, 0.55, 0.15, 0.35])
            rare_freq_analysis_by_drug[drug].plot_max_diff(window_by_stim=(0.25, 2), permutation_test=True,
                                                           smoothing_window=50,max_window_kwargs={'max_func':'max'},
                                                           group_name='name',n_permutations=500,
                                                           plot_kwargs=boxplot_kwargs,
                                                           plot=(rare_freq_analysis_by_drug[drug].plots['ts_by_cond'][0],
                                                                 axinset))
            ttest_drug = rare_freq_analysis_by_drug[drug].ttest_max_diff(['rare-frequent', 'data'],
                                                                         ['rare-frequent', 'shuffled'],
                                                                         alternative='greater',)
            print(f'{drug} ttest: {ttest_drug}')
            # save_stats_to_tex(ttest_drug, stats_dir / f'rarefreq_{drug}_data_vs_shuffled_no_filt.tex')
        # Format and show plots
        for drug in ['muscimol', 'none','saline']:
            format_axis(rare_freq_analysis_by_drug[drug].plots['ts_by_cond'][1])
            rare_freq_analysis_by_drug[drug].plots['ts_by_cond'][1].set_ylim(-0.05,0.35)
            rare_freq_analysis_by_drug[drug].plots['max_diff'][1].set_ylim(-0.25,0.25)
            rare_freq_analysis_by_drug[drug].plots['max_diff'][1].set_title('')
            rare_freq_analysis_by_drug[drug].plots['max_diff'][1].spines["right"].set_visible(True)  # make sure it's visible
            rare_freq_analysis_by_drug[drug].plots['max_diff'][1].spines["top"].set_visible(True)
            rare_freq_analysis_by_drug[drug].plots['max_diff'][1].yaxis.set_label_position("right")
            rare_freq_analysis_by_drug[drug].plots['max_diff'][1].yaxis.tick_right()
            rare_freq_analysis_by_drug[drug].plots['max_diff'][1].set_ylabel("Δ pupil size")
            # remove legend
            if rare_freq_analysis_by_drug[drug].plots['ts_by_cond'][1].get_legend():
                rare_freq_analysis_by_drug[drug].plots['ts_by_cond'][1].get_legend().remove()
            rare_freq_analysis_by_drug[drug].plots['ts_by_cond'][0].set_size_inches(2.5,2)
            # rare_freq_analysis_by_drug[drug].plots['ts_by_cond'][0].set_layout_engine('tight')
            rare_freq_analysis_by_drug[drug].plots['ts_by_cond'][0].show()
            format_axis(rare_freq_analysis_by_drug[drug].plots['max_diff'][1])
            rare_freq_analysis_by_drug[drug].plots['max_diff'][0].show()
            rare_freq_analysis_by_drug[drug].plots['ts_by_cond'][0].savefig(rare_freq_figdir / f'rarefreq_{drug}_ts.pdf')
            # rare_freq_analysis_by_drug[drug].plots['max_diff'][0].savefig(rare_freq_figdir / f'rarefreq_{drug}_maxdiff.pdf')

        # ttest
        # plot all drugs together
        drugs2plot = ['none','muscimol']
        max_diff_by_drug = {drug: rare_freq_analysis_by_drug[drug].max_diff_data['rare-frequent']
                           for drug in drugs2plot}
        max_diff_df_data = pd.concat([drug_max_diff['data'] for drug_max_diff in max_diff_by_drug.values()],axis=1)
        ttest_ind(max_diff_df_data['none'],max_diff_df_data['muscimol'],alternative='greater')
        max_diff_df_shuf = pd.concat([pd.Series(drug_max_diff['shuffled'])
                                      for drug_max_diff in max_diff_by_drug.values()],axis=1)
        for df in [max_diff_df_data, max_diff_df_shuf]:
            df.columns = drugs2plot

        fig, ax = plt.subplots()
        # bar plot data only
        ax.boxplot(max_diff_df_data,positions=np.arange(max_diff_df_data.shape[1]),
                   **boxplot_kwargs)
        cols = []
        for di, drug in enumerate(max_diff_df_data):
            # ax.bar(di,max_diff_df_data[drug].mean()+np.min(max_diff_df_data),bottom=np.min(max_diff_df_data))  # bottom=np.min(max_diff_df_data)
            ax.scatter([di]*len(max_diff_df_data),max_diff_df_data[drug].values,c='w',marker='o',s=5,
                       facecolor="None",edgecolor='k',lw=0.1)
        for _,animal_data in max_diff_df_data.iterrows():
            ax.plot(np.arange(len(animal_data)),animal_data.values,lw=0.1,c='grey')
        format_axis(ax)
        ax.set_title('delta pupil response by drug')
        fig.set_size_inches(1,1.8)
        fig.set_layout_engine('constrained')
        fig.show()
        fig.savefig(rare_freq_figdir/'rare_freq_diff_musc_none_boxplot.pdf')

        # plot rare freq musc none by animal
        rare_freq_diff_ts_plot = plt.subplots()
        rare_freq_by_drug_by_cond = {}
        for di, drug in enumerate(drugs2plot):
            drug_sess_list = rare_freq_analysis_by_drug[drug].sess_list
            drug_sess_list = [s for s in drug_sess_list if 'DO71' not in s]
            resps_by_cond = {e: A_by_cond[e].T.rolling(window=25).mean().T.loc[A_by_cond[e].index.get_level_values('sess').isin(drug_sess_list),0.5:2.5].max(axis=1).groupby(['name']).mean()
                             for e in rare_freq_analysis_by_drug['muscimol'].by_cond}
            rare_freq_diff_ts_plot[1].boxplot(pd.DataFrame.from_dict(resps_by_cond).dropna(),
                                              positions=np.linspace(0,2,len(resps_by_cond)) + (0.3 if di else -0.3),
                                              **boxplot_kwargs)
            rare_freq_by_drug_by_cond[drug] = pd.DataFrame.from_dict(resps_by_cond)
            # for name in max_diff_df_data.index.values:
            #     rare_freq_diff_ts_plot[1][di].plot(resps_by_cond['rare'].loc[name]-resps_by_cond['frequent'].loc[name], label=name)
            # rare_freq_diff_ts_plot[1][di].legend(loc='best')
        format_axis(rare_freq_diff_ts_plot[1])
        rare_freq_diff_ts_plot[1].set_xticks(np.linspace(0,2,len(resps_by_cond)))
        rare_freq_diff_ts_plot[1].set_xticklabels(rare_freq_condnames)
        rare_freq_diff_ts_plot[0].set_size_inches(1.75,2)
        rare_freq_diff_ts_plot[0].set_layout_engine('tight')
        rare_freq_diff_ts_plot[0].show()
        rare_freq_diff_ts_plot[0].savefig(rare_freq_figdir/'rarefreq_resp_means_by_drug.pdf')


        # ttest
        res =ttest_ind(rare_freq_by_drug_by_cond['muscimol'],rare_freq_by_drug_by_cond['none'],alternative='greater')
        # save_stats_to_tex(res,stats_dir/'rare_freq_mean_resps_across_drugs.tex')
        all_combs = pd.concat([rare_freq_by_drug_by_cond['none'],rare_freq_by_drug_by_cond['muscimol']],axis=1)
        all_combs= all_combs.iloc[:,[0,2,1,3]]
        all_comb_ttest = [ttest_ind(all_combs.iloc[:,i],all_combs.iloc[:,ii],alternative='less')[1]
                          for i,ii in list(combinations(range(4),2))]
        print(all_comb_ttest)

        # mne cluster test within drug
        cluster_test_by_drug = {}
        for drug in drugs2plot:
            resps_by_cond = {
                e: A_by_cond[e].loc[A_by_cond[e].index.get_level_values('sess').isin(drug_sess_dict[drug])].groupby(['name']).mean().values
                for e in rare_freq_analysis_by_drug['muscimol'].by_cond}

            results = T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
                list(resps_by_cond.values()), n_permutations=1000, tail=1, n_jobs=30,threshold=0.025
            )
            cluster_test_by_drug[drug] = results

        freq_early_late_figdir = ceph_dir / 'Dammy' / 'figures' / 'freq_early_late'
        freq_early_late_figdir.mkdir(exist_ok=True)

        freq_early_late_analysis = PupilCondAnalysis(
            by_cond_dict=A_by_cond,
            conditions=['rare', 'frequent_prate_early', 'frequent_prate_middle', 'frequent_prate_late'],
        )
        freq_early_late_analysis.plot_ts_by_cond()
        freq_early_late_analysis.plot_max_diff(
            diff_conditions=['rare', 'frequent_prate_early', 'frequent_prate_middle', 'frequent_prate_late'],
            window_by_stim=(1.5, 2.5), mean=np.max, plot_kwargs={'showfliers': False, 'widths': 0.3},
            group_name='name', permutation_test=False, scatter=True,
        )
        # Format and show plots
        format_axis(freq_early_late_analysis.plots['ts_by_cond'][1])
        freq_early_late_analysis.plots['ts_by_cond'][0].show()
        format_axis(freq_early_late_analysis.plots['max_diff'][1])
        freq_early_late_analysis.plots['max_diff'][0].show()

        # t-test between early and late
        print(freq_early_late_analysis.ttest_max_diff(
            ['frequent_prate_early-rare', 'data'],
            ['frequent_prate_late-rare', 'data'],
            alternative='two-sided'
        ))

        # Normal vs deviant analysis
        norm_dev_figdir = ceph_dir / 'Dammy' / 'figures' / 'norm_dev_opto_no_filt'
        norm_dev_figdir.mkdir(parents=True, exist_ok=True)

        normdev_analysis = PupilCondAnalysis(
            by_cond_dict=A_by_cond,
            conditions=conds2analyze,
        )
        normdev_analysis.plot_ts_by_cond()
        normdev_analysis.plot_diff_ts_by_cond()
        normdev_analysis.plot_max_diff(window_by_stim=(1, 2), mean=np.max, permutation_test=True)
        normdev_analysis.ttest_max_diff()
        normdev_analysis.save_ttest_to_tex()


    # --- Repeat analysis for rare_freq and normdev using X_by_cond ---
    for cond in rare_freq_condnames:
        X_by_cond[cond] = group_pupil_across_sessions(sessions,list(sessions.keys()),'X',cond,cond_filters)

    # Rare vs frequent analysis for X_by_cond
    rare_freq_analysis_X_by_drug = {}
    for stim_name, stim_by_cond in zip(['A','X'],[A_by_cond, X_by_cond]):
        for drug in ['muscimol', 'none']:
            drug_rare_freq_sesslist = [sess for sess in all_rare_freq_sess if sess in drug_sess_dict[drug]]
            analysis=run_pupil_cond_analysis(
                by_cond_dict=stim_by_cond,
                sess_list=drug_rare_freq_sesslist,
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
                inset_max_diff=False,
                stats_dir=stats_dir,
                cluster_groupname='sess',
                fig_savename=f'rarefreq_{drug}_{stim_name}_ts_4_cosyne.pdf',
                tex_name=f'rarefreq_{drug}_{stim_name}_data_vs_shuffled_no_filt_4_cosyne.tex',
                ylabel="Δ pupil size",
                figsize=(1.5, 1.5),
                ylim_ts=(-0.05, 0.4),
                xlim_ts=(-.25, 2),
                # ylim_maxdiff=(-0.25, 0.25)
            )
            print(analysis.by_cond['rare'].index.get_level_values('name').unique())

            analysis.plots['ts_by_cond'][0].set_layout_engine('constrained')
            analysis.plots['ts_by_cond'][0].savefig(f'rarefreq_{drug}_{stim_name}_ts_4_cosyne.pdf')

    ### Rare freq early late analysis for drug ###
    rare_freq_early_late_figdir = ceph_dir / 'Dammy' / 'figures' / 'rarefreq_early_late_musc_all_pupilsense'
    if not rare_freq_early_late_figdir.exists():
        rare_freq_early_late_figdir.mkdir()

    cond_filters['frequent_late'] = f'frequent_prate_cumsum >= {25}  & frequent_block_num == 2'
    rare_freq_early_late_conds = [f'frequent_prate_{ti}_block_1' for ti in np.arange(0, 25, 5)][0:5]

    # add to A_by_cond
    for cond in rare_freq_early_late_conds:
        A_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()), 'A', cond, cond_filters)
        X_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()), 'X', cond, cond_filters)
    cmap = plt.get_cmap('Blues_r')  # or any other colormap
    colors = [cmap(i / (len(rare_freq_early_late_conds))) for i in range(len(rare_freq_early_late_conds) + 1)]
    rare_freq_early_late_line_kwargs = {
        cond: {'c': col, 'ls': '-'} for cond, col in zip(rare_freq_early_late_conds, colors)
    }
    rare_freq_early_late_conds = [f'frequent_prate_{ti}_block_1' for ti in np.arange(0, 25, 5)][:3]
    all_rare_freq_early_late_sess = [sess for sess in sessions
                                     if all([sess in A_by_cond[cond].index.get_level_values('sess')
                                             for cond in rare_freq_early_late_conds])]

    # all none opto early late
    for drug in ['none', 'muscimol']:
        drug_sess_list = [sess for sess in all_rare_freq_early_late_sess if sess in drug_sess_dict[drug]]
        analysis = run_pupil_cond_analysis(
            by_cond_dict=A_by_cond,
            sess_list=drug_sess_list,
            conditions=rare_freq_early_late_conds[:3],
            figdir=rare_freq_early_late_figdir,
            line_kwargs=rare_freq_early_late_line_kwargs,
            boxplot_kwargs=boxplot_kwargs,
            group_name='sess',
            permutation_test=False,
            n_permutations=100,
            cluster_groupname=['sess'],
            cluster_comps=[['frequent_prate_10_block_1', 'frequent_prate_0_block_1'], ['frequent_prate_10', 'frequent_prate_10']][:1],
            stats_dir=stats_dir,
            fig_savename=f'rarefreq_early_late_{drug}_ts.pdf',
            tex_name=f'rarefreq_early_late_{drug}_data_vs_shuffled_no_filt.tex',
            ylabel="Δ pupil size",
            figsize=(1.6, 2),
            # ylim_ts=(-0.05, 0.35),
            # ylim_maxdiff=(-0.25, 0.25),
            inset_max_diff=False
        )

        # analysis.plots['ts_by_cond'][1].set_xlim([-0.5, 2])
        analysis.plots['ts_by_cond'][0].show()
        analysis.plots['ts_by_cond'][0].savefig(rare_freq_early_late_figdir / f'rarefreq_early_late_{drug}_ts.pdf')


    # plot performance
    perf_stats = {}
    rt_stats = {}
    for drug in ['none', 'muscimol']:
        drug_sess_list = [sess for sess in sessions
                               if sess in drug_sess_dict[drug]]

        # plot performance as barplot
        all_sess_perf_bar_plot = plt.subplots()
        patt_nonpatt_cols = ['dimgray', 'indigo']

        rare_freq_sess_df = all_td_df.loc[all_td_df.index.get_level_values('sess').isin(drug_sess_list)]

        perf_data = rare_freq_sess_df.query('Stage>=3 & 20<trial_num<=250').groupby(['name', 'Tone_Position'])[
            'Trial_Outcome'].mean().unstack().dropna(axis=0)[[1, 0]].values
        perf_stats[drug] = perf_data

        all_sess_perf_bar_plot[1].bar(['Pattern', 'Non pattern'], perf_data.mean(axis=0),
                                      ec='black', fc='w', lw=0.5, width=0.4)
        [all_sess_perf_bar_plot[1].scatter([cond] * len(cond_data), cond_data, facecolor=c, alpha=0.5, lw=0.01)
         for cond, cond_data, c in zip(['Non pattern', 'Pattern'], perf_data.T, patt_nonpatt_cols)]
        [all_sess_perf_bar_plot[1].plot(['Non pattern', 'Pattern'], a_data, lw=0.25, color='gray')
         for a_data in perf_data]
        all_sess_perf_bar_plot[1].set_ylim(.4, 1.01)
        format_axis(all_sess_perf_bar_plot[1])
        all_sess_perf_bar_plot[1].set_ylabel('Hit rate')
        all_sess_perf_bar_plot[1].set_title(f'{drug}')
        all_sess_perf_bar_plot[0].set_size_inches(1.25, 1.5)
        all_sess_perf_bar_plot[0].show()
        all_sess_perf_bar_plot[0].savefig(
            rare_freq_figdir / f'performance_by_tone_position_{drug}_barplot.pdf')

        rt_df = rare_freq_sess_df.query('Stage>=3 & Trial_Outcome==1')
        rt_df.set_index('Tone_Position', append=True, inplace=True)
        rt_df = (rt_df['Trial_End_dt'] - rt_df['Gap_Time_dt']).dt.total_seconds()
        rt_df = rt_df.groupby(['name', 'Tone_Position']).mean().unstack().dropna(axis=0)[[1, 0]]
        rt_stats[drug] = rt_df.values

    # plot all on 1 plot, same state closer together
    all_states_perf_plot = plt.subplots()
    data = [perf_stats[state][:, idx] for state in perf_stats for idx in range(2)]
    labels = [f'{state} \n {ttype}' for state in perf_stats for ttype in ['Non pattern', 'Pattern']]

    all_states_perf_plot[1].bar(labels, [d.mean() for d in data],
                                ec='black', fc='w', lw=0.5, width=0.4)
    [all_states_perf_plot[1].scatter([label] * len(d), d, facecolor=patt_nonpatt_cols[i % 2], alpha=0.5, lw=0.01)
     for i, (label, d) in enumerate(zip(labels, data))]
    # all_states_perf_plot[1].plot()
    all_states_perf_plot[1].set_ylim(.4, 1.01)
    all_states_perf_plot[1].set_ylabel('Hit rate')
    all_states_perf_plot[0].set_size_inches(2.5, 1.5)
    all_states_perf_plot[0].show()
    all_states_perf_plot[0].set_layout_engine('tight')
    all_states_perf_plot[0].savefig(rare_freq_figdir / f'performance_by_tone_position_all_drugs_barplot.pdf')

    ttest_ind(perf_stats['light_off'][0], perf_stats['light_on'][0])
    ttest_ind(perf_stats['light_off'][1], perf_stats['light_on'][1])
    ttest_ind(perf_stats['light_on'][0], perf_stats['light_on'][1])

    tukey_test = tukey_hsd(*data)
    tukey_test_names = [f'{state} {idx}' for state in perf_stats for idx in range(2)]