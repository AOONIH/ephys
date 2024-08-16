import argparse
import pickle
import platform
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import ttest_ind, sem, permutation_test
from tqdm import tqdm

from behviour_analysis_funcs import get_all_cond_filts, get_drug_dates
from ephys_analysis_funcs import posix_from_win, Session, get_main_sess_patterns

import yaml

from pupil_ephys_funcs import plot_pupil_diff_across_sessions

if '__main__' == __name__:

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    home_dir = Path(config[f'home_dir_{sys_os}'])
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])

    session_topology_paths  = [
        # ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology.csv'),
        # ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_musc_may23.csv'),
        ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_musc_sept23.csv'),
        ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_musc_2406.csv'),
        # ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_fam_jan23.csv'),
    ]

    session_topology_dfs = [pd.read_csv(sess_topology_path) for sess_topology_path in session_topology_paths]
    # [df.set_index([cohort_i]*len(df),) for cohort_i, df in enumerate(session_topology_dfs)]
    [pd.concat({cohort_i: df}, names=['cohort']) for cohort_i, df in enumerate(session_topology_dfs)]
    all_sess_info = pd.concat(session_topology_dfs, axis=0).query('sess_order=="main"')

    # load config and get all drug dates
    cohort_tags = ["_".join(sess_topology_path.stem.split(('_'))[-2:]) for sess_topology_path in session_topology_paths]
    config_paths = [home_dir/'gd_analysis'/'config'/f'{cohort_tag}.yaml' for cohort_tag in cohort_tags]
    cohort_configs = []
    for config_path in config_paths:
        with open(config_path, 'r') as file:
            cohort_configs.append(yaml.safe_load(file))
    drug_sess_dict = {}
    [get_drug_dates(cohort_config, session_topology,drug_sess_dict,date_end=None)
     for cohort_config,session_topology in zip(cohort_configs,session_topology_dfs)]

    # load pupildata pickles
    pkl_dir = ceph_dir/'Dammy'/'pupil_data'
    event_tags = ['A_by_cond', 'X_by_cond']
    pkls2load = sorted(pkl_dir.glob('*by_cond.pkl'))
    event_dfs_dict_by_cohort = {tag: [pd.read_pickle(df_path) for df_path in pkls2load if tag in df_path.name]
                                for tag in event_tags}
    conds = ['rare','frequent']
    event_dfs_dict = {tag: {cond: pd.concat([cohort[cond] for cohort in event_dict],axis=0) for cond in conds}
                      for tag,event_dict in event_dfs_dict_by_cohort.items()}
    cond_filters = get_all_cond_filts()
    # common_filters = 'Trial_Outcome==1'
    good_trial_filt = 'Stage>=3 & n_since_last_Trial_Outcome <=5 & Session_Block==0 & Trial_Outcome==1'
    # update filters
    [cond_filters.update({k: ' & '.join([v, good_trial_filt])}) for k, v in cond_filters.items()]

    # update styles
    plt.style.use('figure_stylesheet.mplstyle')
    line_colours = ['darkblue','darkgreen','darkred','darkorange','darkcyan']
    boxplot_colours = ['#335C67','#9e2a2b','#e09f3e','#540b0e','#fff3b0','#dbd3c9']
    patt_nonpatt_cols = ['darkgrey','indigo']

    #  figure 1 pupil
    fig1_dir = ceph_dir / 'Dammy' / 'figures' / 'figure1_pupil'
    fig1_pkl_dir = ceph_dir / 'Dammy' / 'figures' / 'figure1_pupil' / 'pickles'
    if not fig1_dir.is_dir():
        fig1_dir.mkdir()
    if not fig1_pkl_dir.is_dir():
        fig1_pkl_dir.mkdir()

    rare_vs_frequent_drug_plots = plt.subplots(2,len(drug_sess_dict),sharey='row',figsize=(6*len(drug_sess_dict),5*2))
    for ei,(event,event_response_tag) in enumerate(zip(['A','X'],['A_by_cond','X_by_cond'])):
        event_response = event_dfs_dict[event_response_tag]
        for drug_i,drug in enumerate(['none','muscimol','saline']):
            for cond_i, cond_name in enumerate(event_response):
                plot = rare_vs_frequent_drug_plots[1][ei,drug_i]
                if cond_name in ['mid1','mid2']:
                    continue
                cond_pupil: pd.DataFrame = event_response[cond_name]
                sess_idx_bool = cond_pupil.index.get_level_values('sess').isin(drug_sess_dict[drug])
                cond_pupil_control_sess = cond_pupil[sess_idx_bool]
                mean_means_pupil = cond_pupil_control_sess.groupby(level='sess').mean().mean(axis=0)
                plot.plot(mean_means_pupil, label=cond_name, color=line_colours[cond_i])
                cis = sem(cond_pupil_control_sess.groupby(level='sess').mean())
                plot.fill_between(cond_pupil_control_sess.columns, mean_means_pupil-cis, mean_means_pupil+cis,
                                  fc=line_colours[cond_i],alpha=0.1)
                plot.set_xlabel(f'time from {event.replace("A","pattern")} onset(s)')
                plot.set_ylabel('pupil size')
                plot.legend()
                plot.axvline(0, color='k',ls='--')
                if event == 'A':
                    [plot.axvspan(t,t+0.15,fc='grey',alpha=0.1) for t in np.arange(0,1,0.25)]

    [plot.set_title(drug) for drug,plot in zip(['none','muscimol','saline'],rare_vs_frequent_drug_plots[1][0])]
    rare_vs_frequent_drug_plots[0].suptitle('Pupil response to rare vs. frequent')
    rare_vs_frequent_drug_plots[0].set_layout_engine('tight')
    rare_vs_frequent_drug_plots[0].show()
    rare_vs_frequent_drug_plots[0].savefig(fig1_dir/f'rare_vs_frequent_pupil_response_all_sessions_all_cohorts.svg')

    rare_freq_diff_plot = plt.subplots(ncols=len(drug_sess_dict), sharey='row', figsize=(12, 5))
    for i, sess_type in enumerate(['none', 'saline', 'muscimol']):
        for event, event_response_tag, col in zip(['Pattern', 'X'], ['A_by_cond', 'X_by_cond'], ['g', 'b']):
            event_response = event_dfs_dict[event_response_tag]
            response_diff = plot_pupil_diff_across_sessions(['rare', 'frequent'], event_response, [sess_type],
                                                            drug_sess_dict,
                                                            plot=[rare_freq_diff_plot[0], rare_freq_diff_plot[1][i]],
                                                            plt_kwargs=dict(c=col, label=event))

            [rare_freq_diff_plot[1][i].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
            rare_freq_diff_plot[1][i].set_title(f'{sess_type} sessions')
            rare_freq_diff_plot[1][i].axhline(0, color='k', ls='--')
            rare_freq_diff_plot[1][i].axvline(0, color='k', ls='--')

            # # by animal
            # sess_by_animal = [[sess for sess in response_diff.index.get_level_values('sess') if animal in sess]
            #                   for animal in response_diff.index.get_level_values('sess').str.split('_').str[0].unique()]
            # response_across_sess_by_animal = []
            # for animal_sess in sess_by_animal:
            #     response_by_animal_by_sess = response_diff.loc[animal_sess]
            #     rare_freq_diff_plot[1][i].plot(response_by_animal_by_sess.columns,
            #                                    response_by_animal_by_sess.mean(axis=0),c=col, alpha=0.3,lw=0.5)
            #     response_across_sess_by_animal.append(response_by_animal_by_sess.mean(axis=0))
            # rare_freq_diff_plot[1][i].plot(response_diff.columns,
            #                                np.mean(response_across_sess_by_animal,axis=0),c=col,lw=2,ls='--')

    [ax.locator_params(axis='both', nbins=4) for ax in rare_freq_diff_plot[1]]
    rare_freq_diff_plot[1][-1].legend()
    rare_freq_diff_plot[0].set_layout_engine('tight')
    rare_freq_diff_plot[0].show()

    rare_freq_diff_plot[0].savefig(fig1_dir / f'rare_vs_frequent_pupil_diff_across_sessions_all_cohorts.svg')

    rare_freq_diff_all_drugs = plt.subplots()
    max_diff_by_drug = []
    for i, (sess_type, col) in enumerate(zip(['none', 'saline', 'muscimol'], ['dimgray', 'darkblue', 'darkred'])):
        response_diff = plot_pupil_diff_across_sessions(['rare', 'frequent'], event_dfs_dict['A_by_cond'], [sess_type],
                                                        drug_sess_dict,plot=rare_freq_diff_all_drugs,
                                                        plt_kwargs=dict(c=col, label=sess_type))
        max_diff_by_drug.append(response_diff.loc[:,1.75:2.25].max(axis=1))

    rare_freq_diff_all_drugs[1].set_title(f'Pupil difference between rare and frequent')
    rare_freq_diff_all_drugs[1].set_xlabel(f'Time from pattern onset (s)')
    rare_freq_diff_all_drugs[1].set_ylabel('pupil size (rare - frequent)')
    box = rare_freq_diff_all_drugs[1].get_position()
    rare_freq_diff_all_drugs[1].set_position([box.x0, box.y0, box.width * 0.9, box.height])
    rare_freq_diff_all_drugs[0].legend(bbox_to_anchor=(.85, .88))
    rare_freq_diff_all_drugs[1].axhline(0, color='k', ls='--')
    rare_freq_diff_all_drugs[1].locator_params(axis='both', nbins=4)
    rare_freq_diff_all_drugs[1].tick_params(axis='both')
    [rare_freq_diff_all_drugs[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
    rare_freq_diff_all_drugs[0].set_size_inches(8, 6)
    rare_freq_diff_all_drugs[0].show()
    rare_freq_diff_all_drugs[0].savefig(
        fig1_dir / f'rare_vs_frequent_2A_pupil_diff_across_sessions_all_drugs_all_cohorts.svg')

    # plot max diff over window by sessions
    max_diff_plot = plt.subplots()
    bxplot = max_diff_plot[1].boxplot([drug_max_diff for drug_max_diff in max_diff_by_drug],
                                      labels=['none','saline','muscimol'],
                                      showmeans=False, showfliers=False)
    for patch, color in zip(bxplot['boxes'], ['dimgray', 'darkblue', 'darkred']):
        patch.set_facecolor(color)
    max_diff_plot[1].set_ylabel('Max pupil difference\n between rare and frequent')
    max_diff_plot[1].locator_params(axis='y', nbins=4)
    max_diff_plot[0].set_layout_engine('tight')
    max_diff_plot[0].show()
    max_diff_plot[0].savefig(fig1_dir / f'max_diff_boxplot_rare_vs_frequent_pupil_diff_across_sessions_all_drugs.svg')
    [print(len(drug_max_diff)) for drug_max_diff in max_diff_by_drug]
    [print(ttest_ind(max_diff_by_drug[i].values, max_diff_by_drug[j].values,equal_var=False,alternative='two-sided',
                     trim=0.2))
     for i, j in list(combinations(range(3), 2))]




