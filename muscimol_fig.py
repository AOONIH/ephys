import argparse
import platform
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind, tukey_hsd

from io_utils import posix_from_win
from pupil_analysis_funcs import plot_pupil_diff_across_sessions

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    home_dir = Path(config[f'home_dir_{sys_os}'])
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])
    load_diff_data = True

    plt.style.use('figure_stylesheet.mplstyle')
    musc_figdir = ceph_dir / 'Dammy' / 'figures' / 'figure_muscimol'

    max_diff_by_drug = {}
    rare_freq_diff_by_drug = {}
    if not load_diff_data:

        for i, (sess_type, col) in enumerate(zip(['none', 'saline', 'muscimol'], ['dimgray', 'darkblue', 'darkred'])):
            response_diff = plot_pupil_diff_across_sessions(['rare', 'frequent'], event_dfs_dict['A_by_cond'],
                                                            [sess_type],
                                                            drug_sess_dict,
                                                            plt_kwargs=dict(c=col, label=sess_type))
            rare_freq_diff_by_drug[sess_type] = response_diff
    else:
        diff_data_pkl_dir = ceph_dir / posix_from_win(r'X:\Dammy\figures\figure_muscimol')
        delta_pkls_by_cohort = [pd.read_pickle(pkl_file) for pkl_file in diff_data_pkl_dir.glob('*rare_freq_delta.pkl')]
        [rare_freq_diff_by_drug.update({drug: pd.concat([cohort[drug] for cohort in delta_pkls_by_cohort], axis=0)})
         for drug in ['muscimol', 'saline', 'none']]
    for drug in ['muscimol', 'saline', 'none']:
        if 'name' not in rare_freq_diff_by_drug[drug].index.names:
            rare_freq_diff_by_drug[drug]['name'] = rare_freq_diff_by_drug[drug].index.to_series().str.split('_').str[0]
            rare_freq_diff_by_drug[drug].set_index('name', append=True, inplace=True)
        rare_freq_diff_by_drug[drug] = rare_freq_diff_by_drug[drug].query('name != "DO84"')

    rare_freq_diff_all_drugs = plt.subplots()
    drug_line_cols = {'muscimol': 'darkred', 'saline': 'darkblue', 'none': 'dimgray'}
    group_by_animal = True
    for i, drug in enumerate(['none', 'saline', 'muscimol']):
        # smooothed_response = pd.DataFrame(savgol_filter(rare_freq_diff_by_drug[drug], 50,2), columns=rare_freq_diff_by_drug[drug].columns)
        # smooothed_response = rare_freq_diff_by_drug[drug].T.rolling(25).mean().T
        smooothed_response = rare_freq_diff_by_drug[drug]
        # add name to multi index
        if 'name' not in smooothed_response.index.names:
            smooothed_response['name'] = smooothed_response.index.to_series().str.split('_').str[0]
            smooothed_response.set_index('name', append=True, inplace=True)
        if group_by_animal:
            smooothed_response = smooothed_response.groupby('name').mean()
        smooothed_response = smooothed_response.query('name != "DO84"')
        rare_freq_diff_all_drugs[1].plot(rare_freq_diff_by_drug[drug].columns,
                                         smooothed_response.mean(axis=0),
                                         label=drug, color=drug_line_cols[drug])
        rare_freq_diff_all_drugs[1].fill_between(rare_freq_diff_by_drug[drug].columns.tolist(),
                                                 smooothed_response.mean(axis=0) - smooothed_response.sem(axis=0),
                                                 smooothed_response.mean(axis=0) + smooothed_response.sem(axis=0),
                                                 fc=drug_line_cols[drug], alpha=0.1)
        # max_diff_by_drug[drug] = rare_freq_diff_by_drug[drug].loc[:,2:2.5].quantile(0.999,axis=1)
        max_diff_by_drug[drug] = rare_freq_diff_by_drug[drug].loc[:, 1.75:2.25].max(axis=1)
    # rare_freq_diff_all_drugs = plt.subplots()

    # for i, (sess_type, col) in enumerate(zip(['none', 'saline', 'muscimol'], ['dimgray', 'darkblue', 'darkred'])):
    #     response_diff = plot_pupil_diff_across_sessions(['rare', 'frequent'], event_dfs_dict['A_by_cond'], [sess_type],
    #                                                     drug_sess_dict,plot=rare_freq_diff_all_drugs,
    #                                                     plt_kwargs=dict(c=col, label=sess_type))
    #     max_diff_by_drug.append(response_diff.loc[:,1.75:2.25].max(axis=1))

    # rare_freq_diff_all_drugs[1].set_title(f'Pupil difference between rare and frequent')
    rare_freq_diff_all_drugs[1].set_xlabel(f'')
    rare_freq_diff_all_drugs[1].set_ylabel('')
    box = rare_freq_diff_all_drugs[1].get_position()
    rare_freq_diff_all_drugs[1].set_position([box.x0, box.y0, box.width * 0.9, box.height])
    rare_freq_diff_all_drugs[1].legend(loc='upper left')
    rare_freq_diff_all_drugs[1].axhline(0, color='k', ls='--')
    rare_freq_diff_all_drugs[1].locator_params(axis='both', nbins=4)
    rare_freq_diff_all_drugs[1].tick_params(axis='both')
    [rare_freq_diff_all_drugs[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
    rare_freq_diff_all_drugs[0].set_size_inches(4, 3)
    rare_freq_diff_all_drugs[0].set_layout_engine('tight')
    rare_freq_diff_all_drugs[0].show()
    # assert False

    rare_freq_diff_all_drugs[0].savefig(
        musc_figdir / f'rare_vs_frequent_2A_pupil_diff_{"across_mice" if group_by_animal else "across_sessions"}_all_musc_no_84_all_cohorts.pdf')

    # max diff by animal
    max_diff_by_drug_df = pd.DataFrame(max_diff_by_drug)
    if 'name' not in max_diff_by_drug_df.index.names:
        max_diff_by_drug_df['name'] = max_diff_by_drug_df.index.to_series().str.split('_').str[0]
        max_diff_by_drug_df.set_index('name',append=True, inplace=True)
    max_diff_by_drug_df = max_diff_by_drug_df.query('name!="DO84"')

    # plot max diff over window by sessions
    max_diff_plot = plt.subplots()
    # bxplot = max_diff_plot[1].boxplot(list(max_diff_by_drug.values()),
    #                                   labels=list(max_diff_by_drug.keys()),
    #                                   showmeans=True, showfliers=False)
    max_diffs_by_name = max_diff_by_drug_df.groupby('name').mean()
    bxplot = max_diff_plot[1].boxplot([max_diffs_by_name.dropna()[drug]
                                       for drug in ['none','saline', 'muscimol']],
                                      labels=['none','saline', 'muscimol'] if 'saline' in max_diff_by_drug.keys() else ['none','muscimol'],
                                      meanprops=dict(color='k',ls='-'),meanline=False,medianprops=dict(lw=1),
                                      showmeans=False, showfliers=False,notch=False,bootstrap=10000,
                                      widths=0.4
                                      )

    max_diff_plot[0].set_layout_engine('tight')
    max_diff_plot[0].show()
    [print(len(drug_max_diff)) for drug_max_diff in max_diff_by_drug.values()]
    drugs = list(max_diff_by_drug.keys())
    [print(ttest_ind(max_diff_by_drug[i].values, max_diff_by_drug[j].values, equal_var=False, alternative='two-sided',
                     ))
     for i, j in list(combinations(drugs, 2))]
    tukey_test = tukey_hsd(*[max_diff_by_drug[drug] for drug in list(max_diff_by_drug.keys())])
    print(tukey_test)

    for drug, col in zip(drugs, ['dimgray', 'darkred', 'darkred']):
        [max_diff_plot[1].plot(max_diff_plot[1].get_xticks(),
            max_diffs_by_name.loc[name,['none','saline', 'muscimol']],
            c='grey', lw=0.5,alpha=0.25)
         for name in max_diffs_by_name.index.get_level_values('name').unique()]
         # for name in ['DO83','DO84','DO85','DO86']]

    max_diff_plot[1].set_ylabel('')
    max_diff_plot[1].locator_params(axis='y', nbins=2)
    max_diff_plot[0].set_layout_engine('tight')
    max_diff_plot[0].set_size_inches(3, 2.5)
    max_diff_plot[0].show()
    max_diff_plot[0].savefig(musc_figdir / f'max_diff_barplot_rare_vs_frequent_pupil_diff_across_sessions_musc_no_84.pdf')

    site_coords_df = pd.read_csv(ceph_dir/posix_from_win(r'X:\Dammy\anatomy\musc_sites\site_coords.csv'),index_col=0)
    site_coords_df['side'] = ['left','right']*int(len(site_coords_df)/2)
    # append side column to index
    site_coords_df.set_index('side', append=True, inplace=True)
    musc_vs_none_eff_by_name = max_diffs_by_name['none'] - max_diffs_by_name['muscimol']
    cols = matplotlib.colormaps['tab10'](np.linspace(0, 1, len(musc_vs_none_eff_by_name)))

    musc_effect_by_loc_plot = plt.subplots(3,figsize=(6,6))
    for side,m in zip(site_coords_df.index.get_level_values('side').unique(),['<','>']):
        for ci,coord, in enumerate(site_coords_df.columns):
            musc_effect_by_loc_plot[1][ci].scatter(site_coords_df.xs(side,level='side')[coord],musc_vs_none_eff_by_name,
                                                   marker=m, edgecolor=cols,facecolor='white',
                                                   label=musc_vs_none_eff_by_name.index.values)
            musc_effect_by_loc_plot[1][ci].set_title(coord)
            musc_effect_by_loc_plot[1][ci].set_xlim(site_coords_df[coord].min()-0.05,site_coords_df[coord].max()+0.05)
            # musc_effect_by_loc_plot[1][ci].legend() if coord == site_coords_df.columns[0] else None
    musc_effect_by_loc_plot[0].set_layout_engine('tight')
    musc_effect_by_loc_plot[0].show()
    musc_effect_by_loc_plot[0].savefig(musc_figdir / f'musc_effect_by_loc.pdf')

    # plot effect size by location
