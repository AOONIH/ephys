import os
import pickle
import warnings
from copy import copy

import joblib
from scipy.signal import savgol_filter

from TimeSeries_clasification.cluster_analysis import cluster_analysis, plot_clusters, plot_cluster_stats
from behviour_analysis_funcs import sync_beh2sound, get_all_cond_filts, add_datetimecol, get_drug_dates, \
    get_n_since_last, get_earlyX_trials, get_prate_block_num, filter_session, get_cumsum_columns, \
    get_lick_in_patt_trials, get_main_sess_patterns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import platform
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm
from scipy.stats import sem, ttest_ind, ttest_1samp

from decoding_funcs import Decoder
from io_utils import posix_from_win, load_pupil_data
from plot_funcs import plot_ts_var, format_axis, plot_shaded_error_ts
from reformat_dir_struct import extract_date
from save_utils import save_stats_to_tex
from sess_dataclasses import Session


def group_pupil_across_sessions(sess_dict_objs: dict,sessnames:list,event:str, cond_name:str,cond_filters=None,
                                td_df_query=None, sess_list=None,use_all=False):
    assert cond_filters or td_df_query or use_all
    all_cond_pupil = []
    if cond_filters is not None:
        cond_filter = cond_filters[cond_name]
    elif td_df_query is not None:
        cond_filter = td_df_query
    if cond_name == 'none':
        pass
    for sessname in sessnames:
        assert isinstance(sess_dict_objs[sessname],Session)
        if sess_dict_objs[sessname].pupil_obj is None:
            continue
        if sess_dict_objs[sessname].pupil_obj.aligned_pupil is None or \
                len(sess_dict_objs[sessname].pupil_obj.aligned_pupil) == 0:
            continue
        if not use_all:
            responses = sess_dict_objs[sessname].pupil_obj.aligned_pupil[event]
            trial_nums = sess_dict_objs[sessname].td_df.query(cond_filter).index.get_level_values('trial_num').values
            cond_pupil = responses.loc[responses.index.isin(trial_nums, level='trial')]
            # if sessname=='DO79_240118':
            #     print(sessname,event,cond_name,cond_filter,cond_pupil.shape)
            # cond_pupil = sess_dict_objs[sessname].pupil_obj.aligned_pupil[event].query('trial in @trial_nums')
        else:
            try:
                cond_pupil = sess_dict_objs[sessname].pupil_obj.aligned_pupil[event]
            except KeyError:
                print(f'no {event} data for {sessname}')
                continue
        all_cond_pupil.append(cond_pupil)
    cond_pupil_df = pd.concat(all_cond_pupil, axis=0)
    # cond_pupil_df['name'] = cond_pupil_df.index.get_level_values('sess').str.split('_').str[0]
    # cond_pupil_df['date'] = cond_pupil_df.index.get_level_values('sess').str.split('_').str[1]
    # # add name and date to multiindex
    # cond_pupil_df.set_index(['name', 'date'], append=True, inplace=True)

    return cond_pupil_df


def get_pupil_diff_by_session(pupil_data1:pd.DataFrame, pupil_data2:pd.DataFrame,) -> pd.DataFrame:
    unique_sess = np.unique(pupil_data1.index.get_level_values('sess'))
    pupil_diff_by_sess = []
    used_sess = []
    for sess in unique_sess:
        if not all(sess in df.index.get_level_values('sess') for df in [pupil_data1,pupil_data2]):
            print(sess)
            continue
        if any([pupil_data1.xs(sess,level='sess').shape[0]<7, pupil_data2.xs(sess,level='sess').shape[0]<7]):
            print(f'{sess} not enough data for pupil diff')
            continue
        mean_diff = (pupil_data1.xs(sess,level='sess').median(axis=0)-
                     pupil_data2.xs(sess,level='sess').median(axis=0))
        pupil_diff_by_sess.append(mean_diff)
        used_sess.append(sess)
    return pd.DataFrame(pupil_diff_by_sess,index=pd.Index(used_sess,name='sess'))


def plot_pupil_diff_across_sessions(cond_list, event_responses, sess_drug_types, drug_sess_dict, dates=None, plot=None,
                                    plt_kwargs=None,savgol_kwargs=None):
    if plot is None:
        plot = plt.subplots()
    if plt_kwargs is None:
        plt_kwargs = {}
    response_diff_df = get_pupil_diff_by_session(event_responses[cond_list[0]], event_responses[cond_list[1]])
    sess_idx_bool = np.any([response_diff_df.index.isin(drug_sess_dict[drug]) for drug in sess_drug_types], axis=0)
    response_diff_df_subset = response_diff_df[sess_idx_bool]
    if dates is not None:
        response_diff_df_subset = response_diff_df_subset.query('sess in @dates')
    mean_response = savgol_filter(response_diff_df_subset.mean(axis=0), axis=0,
                                  **savgol_kwargs if savgol_kwargs else {'window_length': 5, 'polyorder': 2})

    plot[1].plot(response_diff_df_subset.columns,mean_response, **plt_kwargs)

    # plot_ts_var(response_diff_df_subset.columns, response_diff_df_subset.values, plt_kwargs['c'], plot[1],
    #             ci_kwargs={'var_func': sem, 'confidence': 0.99})
    plot[1].fill_between(response_diff_df_subset.columns, mean_response - response_diff_df_subset.sem(axis=0),
                         mean_response + response_diff_df_subset.sem(axis=0),
                         color=plt_kwargs.get('c','k'), alpha=0.1)
    return response_diff_df_subset


def plot_pupil_ts_by_cond(responses_by_cond:dict, conditions, plot=None, sess_list=None, plot_indv_sess=False,
                          plot_mean=True, cond_line_kwargs=None, group_name=['sess'], plot_kwargs=None):
    if plot is None:
        pupil_ts_plot = plt.subplots()
    else:
        pupil_ts_plot = plot

    if plot_kwargs is None:
        plot_kwargs = {}

    print(f'{cond_line_kwargs = }')
    x_ser = np.array(responses_by_cond[conditions[0]].columns,dtype=float)

    # group_by_names = ['sess'] if group_name == 'sess' else ['sess',group_name]
    group_by_names = group_name
    lss= ['-', '--']
    # for cond_i, cond in enumerate(['alternating', 'random']):
    if sess_list is None:
        print('Using all sess_dict')
        sess_list = responses_by_cond[conditions[0]].index.get_level_values('sess').unique().tolist()
    for cond_i, cond in enumerate(conditions):
        responses_df = responses_by_cond[cond]
        if 'name' not in responses_df.index.names:
            responses_df.loc[:,'name'] = responses_df.index.get_level_values('sess').str.split('_').str[0]
            responses_df.set_index(['name'], append=True, inplace=True)
        mean_responses_by_sess = responses_df.query('sess in @sess_list').groupby(level=group_by_names).median()
        # if group_name != 'sess':
        mean_responses_by_sess = mean_responses_by_sess.groupby(level=group_name).mean()
        # add name to multiindex
        if plot_indv_sess:
            [pupil_ts_plot[1].plot(x_ser,sess_response,alpha=0.25,
                                   c=cond_line_kwargs[cond]['c'] if cond_line_kwargs else f'C{cond_i}',)
             for _,sess_response in mean_responses_by_sess.iterrows()]
        if plot_mean:
            pupil_ts_plot[1].plot(x_ser, mean_responses_by_sess.mean(axis=0),
                                  label='_'.join([cond,plot_kwargs.get('label_sffx','')]) if plot_kwargs.get('label_sffx') else cond,
                                  **cond_line_kwargs[cond] if cond_line_kwargs else {})
            # plot sems
            if len(mean_responses_by_sess) > 1:
                # sem over sess_dict
                # pupil_ts_plot[1].fill_between(x_ser,
                                            #   np.array(mean_responses_by_sess.mean(axis=0) - mean_responses_by_sess.sem(axis=0)),
                                            #   np.array(mean_responses_by_sess.mean(axis=0) + mean_responses_by_sess.sem(axis=0)),
                                            #   fc=cond_line_kwargs[cond]['c'] if cond_line_kwargs else f'C{len(pupil_ts_plot[1].lines) - 1}',
                                            #   alpha=0.1)
                plot_shaded_error_ts(pupil_ts_plot[1],x_ser, mean_responses_by_sess.mean(axis=0).values,mean_responses_by_sess.sem(axis=0).values,
                                     fc=cond_line_kwargs[cond]['c'] if cond_line_kwargs else f'C{cond_i}',alpha=0.1)
                
            else:

                # bootstrap ci for one session
                # plot_ts_var(x_ser, responses_by_cond[cond].xs(sess_list[0],level='sess').values,
                #             colour=cond_line_kwargs[cond]['c'] if cond_line_kwargs else f'C{cond_i}',plt_ax=pupil_ts_plot[1])
                # sess_bootstrap = np.array(np.array([[sess_responses[idxs,t].mean() for idxs in
                #                           [np.random.choice(sess_responses.shape[0],sess_responses.shape[0],
                #                                             replace=True) for _ in range(1000)]]
                #                             for t in range(x_ser.shape[0])])).T
                # print(sess_bootstrap.shape)
                # pupil_ts_plot[1].fill_between(x_ser,
                #                               np.quantile(sess_bootstrap, 0.025,axis=0),
                #                               np.quantile(sess_bootstrap, 0.975,axis=0),
                #                               fc=cond_line_kwargs[cond]['c'] if cond_line_kwargs else f'C{cond_i}',
                #                               alpha=0.1)
                sess_sem = responses_by_cond[cond].xs(sess_list[0], level=group_name).sem(axis=0)
                # pupil_ts_plot[1].fill_between(x_ser.tolist(),
                                            #   mean_responses_by_sess.mean(axis=0) - sess_sem,
                                            #   mean_responses_by_sess.mean(axis=0) + sess_sem,
                                            #   fc=cond_line_kwargs[cond]['c'] if cond_line_kwargs else f'C{cond_i}',
                                            #   alpha=0.1)
                plot_shaded_error_ts(pupil_ts_plot[1],x_ser, mean_responses_by_sess.mean(axis=0).values,sess_sem.values,
                                     fc=cond_line_kwargs[cond]['c'] if cond_line_kwargs else f'C{cond_i}',alpha=0.1)


    pupil_ts_plot[1].set_title('Pupil response')
    pupil_ts_plot[1].legend()
    pupil_ts_plot[0].set_layout_engine('tight')
    # pupil_ts_plot[0].show()

    if plot is None:
        return pupil_ts_plot


def plot_pupil_diff_ts_by_cond(responses_by_cond: dict, conditions, plot=None, sess_list=None, plot_kwargs=None,
                               plot_indv_group=None, cond_line_kwargs=None,group_level='sess',**kwargs):
    # Validate input
    if len(conditions) < 2:
        raise ValueError("At least two conditions must be provided.")

    if plot is None:
        pupil_diff_ts_plot = plt.subplots()
    else:
        pupil_diff_ts_plot = plot

    # Extract the x-axis values
    x_ser = responses_by_cond[conditions[0]].columns

    # If no session list is provided, use all sess_dict
    if sess_list is None:
        print('Using all sess_dict')
        sess_list = responses_by_cond[conditions[0]].index.get_level_values('sess').unique().tolist()

    # Initialize default arguments
    if plot_kwargs is None:
        plot_kwargs = {}
    if cond_line_kwargs is None:
        lss = ['-', '--', ':', '-.']
        cond_line_kwargs = {cond_i:{'c':'k','ls':lss[cond_i]} for cond_i, cond in enumerate(conditions)}

    # Group by session and calculate the median response
    responses_by_cond_by_sess = [
        responses_by_cond[cond].loc[responses_by_cond[cond].index.get_level_values('sess').isin(sess_list)].groupby(
            level='sess').median()
        for cond in conditions]
    # add name to multiindex
    for response_df in responses_by_cond_by_sess:
        response_df['name'] = response_df.index.get_level_values('sess').str.split('_').str[0]
        response_df.set_index(['name'], append=True, inplace=True)

    # Calculate differences from the baseline condition
    [print(e.shape) for e in responses_by_cond_by_sess]
    by_sess_diffs = [responses_by_cond_by_sess[0].sub(responses, level='sess') * (-1 if kwargs.get('invert') else 1)
                     for responses in responses_by_cond_by_sess[1:]]
    by_sess_diffs_arrs = [np.array(diffs) for diffs in by_sess_diffs]
    by_sess_bootstrap = [np.array([by_sess_diffs_cond[np.random.choice(len(by_sess_diffs_cond),
                                                                                  len(by_sess_diffs_cond),
                                                                                  replace=True)].mean(axis=0)
                         for _ in range(1000)])
                         for by_sess_diffs_cond in by_sess_diffs_arrs]
    print(by_sess_bootstrap[0].shape)
    # Plot the mean difference and standard error for each condition
    for cond_i, by_sess_diff in enumerate(by_sess_diffs):
        color = cond_line_kwargs.get(cond_i, {}).get('c', f'C{cond_i}')
        pupil_diff_ts_plot[1].plot(x_ser, by_sess_diff.mean(axis=0), **cond_line_kwargs.get(cond_i, {}))
        pupil_diff_ts_plot[1].fill_between(x_ser.tolist(),
                                           by_sess_diff.mean(axis=0) - by_sess_diff.sem(axis=0),
                                           # np.quantile(by_sess_bootstrap[cond_i],0.025,axis=0),
                                           by_sess_diff.mean(axis=0) + by_sess_diff.sem(axis=0),
                                           # np.quantile(by_sess_bootstrap[cond_i],0.975,axis=0),
                                           alpha=0.1, fc=color)
        if plot_indv_group is not None:
            assert plot_indv_group in by_sess_diff.index.names, f"{plot_indv_group} not in {by_sess_diff.index.names}"
            [pupil_diff_ts_plot[1].plot(x_ser, group_resp.rolling(window=50).mean(), c=color, alpha=0.1, ls='-')
             for _,group_resp in by_sess_diff.groupby(plot_indv_group).mean().iterrows()]
            print(by_sess_diff.index.get_level_values(plot_indv_group).unique())


    # Set plot title and axis labels
    pupil_diff_ts_plot[1].set_title('Pupil response', **plot_kwargs.get('title', {}))
    pupil_diff_ts_plot[1].set_xlabel(plot_kwargs.get('xlabel', 'Time'))
    pupil_diff_ts_plot[1].set_ylabel(plot_kwargs.get('ylabel', f'Difference ({conditions[0]} - {conditions[1]})'))

    # # Add shaded regions
    # for t in np.arange(0, 1, 0.25):
    #     pupil_diff_ts_plot[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1)

    # Tight layout and show plot
    plt.tight_layout()
    # plt.show()

    if plot is None:
        return pupil_diff_ts_plot


def add_name_to_response_dfs(response_dict):
    # add name to multiindex
    for response_df in response_dict.values():
        if 'name' in response_df.index.names:
            continue
        response_df['name'] = response_df.index.get_level_values('sess').str.split('_').str[0]
        response_df.set_index(['name'], append=True, inplace=True)


def get_mean_responses_by_cond(conds, response_dict, sess_list, smoothing_window=25,shuffle=False, **kwargs):
    all_responses_by_cond = {
        cond: response_dict[cond].loc[response_dict[cond].index.get_level_values('sess').isin(sess_list)].T.rolling(
            window=smoothing_window).mean().T
        for cond in conds}
    if shuffle:
        temp_list = {cond: [] for cond in conds}
        for sess in sess_list:
            sess_resps_list = [all_responses_by_cond[cond].xs(sess, level='sess',drop_level=False) for cond in conds]
            n_per_cond = np.pad(np.cumsum([resps.shape[0] for resps in sess_resps_list]),[1,0])
            all_sess_responses = pd.concat(sess_resps_list, axis=0)
            sess_responses_shuffled = all_sess_responses.sample(frac=1, replace=False)
            for cond,start,end in zip(conds,n_per_cond,n_per_cond[1:]):
                temp_list[cond].append(sess_responses_shuffled.take(np.arange(start,end)))

        all_responses_by_cond = {cond: pd.concat(temp_list[cond], axis=0) for cond in conds}
    mean_responses_by_sess_by_cond = {cond: resps.groupby(['sess', 'name']).mean()
                                      for cond, resps in all_responses_by_cond.items()}

    return mean_responses_by_sess_by_cond

def get_response_diff(response1, response2, group_level,sub_by_sess=False):
    if not sub_by_sess:
        # print(response1.groupby(group_level).mean() - response2.groupby(group_level).mean())
        return response1.groupby(group_level).mean() - response2.groupby(group_level).mean()
    else:
        return response1.groupby(['sess','name']).mean() - response2.groupby(['sess','name']).mean()


def get_max_in_window(response_df: pd.DataFrame, window: list, group_level, sub_by_sess=False, **kwargs):
    # print(responses_diff)
    if kwargs.get('smoothing_window', None) is not None:
        response_df = response_df.T.rolling(window=kwargs['smoothing_window']).mean().T
    max_method = kwargs.get('max_func', 'max')
    if max_method == 'max':
        max_resp = response_df.loc[:, window[0]:window[1]].groupby(group_level).mean().max(axis=1)
    elif max_method == 'min':
        max_resp = response_df.loc[:, window[0]:window[1]].min(axis=1).groupby(group_level).mean()
    elif max_method == 'median':
        max_resp = response_df.loc[:, window[0]:window[1]].median(axis=1).groupby(group_level).mean()
    elif max_method == 'mean':
        max_resp = response_df.loc[:, window[0]:window[1]].mean(axis=1).groupby(group_level).mean()
    elif max_method == 'sum':
        max_resp = response_df.loc[:, window[0]:window[1]].sum(axis=1).groupby(group_level).mean()
    else:
        raise ValueError(f"Invalid max_func: {max_method}")

    return max_resp


def get_max_diffs_by_condition(responses_by_cond: dict, conditions, sess_list=None,
                                group_name='name', **kwargs):
    if len(conditions) < 2:
        raise ValueError("At least two conditions must be provided.")

    if sess_list is None:
        sess_list = responses_by_cond[conditions[0]].index.get_level_values('sess').unique().tolist()

    add_name_to_response_dfs(responses_by_cond)

    window = kwargs.get('window_by_stim', (1.5, 2.5))

    max_by_group = {
        cond: get_max_in_window(responses_by_cond[cond], window, group_level=['sess','name'],
                                smoothing_window=kwargs.get('smoothing_window', 10),
                                kwargs=kwargs.get('max_window_kwargs', {}))
        for cond in conditions
    }

    mean_responses_by_cond = get_mean_responses_by_cond(
        conditions,
        max_by_group,
        sess_list,
        smoothing_window=1,
        shuffle=kwargs.get('shuffle', False)
    )

    diff_kwargs = kwargs.get('diff_kwargs', {})
    max_diffs_by_group = {
        cond: get_response_diff(mean_responses_by_cond[cond], mean_responses_by_cond[conditions[0]],
                                group_level=group_name, **diff_kwargs)
        for cond in conditions[1:]
    }

    return max_diffs_by_group


def plot_max_diffs(max_diffs_by_group: dict, plot=None, scatter=False, boxplot=True, **plot_kwargs):
    import matplotlib.pyplot as plt
    import numpy as np

    if plot is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot

    data = list(max_diffs_by_group.values())
    if 'labels' not in plot_kwargs:
        plot_kwargs['labels'] = list(max_diffs_by_group.keys())
    labels = plot_kwargs['labels']

    if 'positions' not in plot_kwargs:
        plot_kwargs['positions'] = np.arange(1, len(data) + 1)
    plot_kwargs['labels'] = labels
    positions = plot_kwargs['positions']

    if boxplot:
        ax.boxplot(data,  **plot_kwargs)

    if scatter:
        for i, values in enumerate(data):
            x_vals = np.random.normal(positions[i], 0.1, size=len(values))
            ax.scatter(x_vals, values, alpha=0.6, color='black', s=10)

    ax.set_title(plot_kwargs.get('title', 'Max Pupil Differences'))
    ax.set_ylabel(plot_kwargs.get('ylabel', 'Difference'))

    return fig, ax


def get_permuted_max_diffs(responses_by_cond: dict, conditions, sess_list=None,
                           n_permutations=1000, group_name='name', **kwargs):

    shuffled_diffs_list = []

    for _ in tqdm(range(n_permutations), desc='Permutation testing', total=n_permutations):
        max_diffs_shuffled = get_max_diffs_by_condition(
            responses_by_cond,
            conditions,
            sess_list=sess_list,
            group_name=group_name,
            shuffle=True,
            **kwargs
        )
        shuffled_diffs_list.append(max_diffs_shuffled)

    # Aggregate into one dict of lists per condition
    diff_cond_names = list(shuffled_diffs_list[0].keys())
    shuffled_diffs_by_cond = {e: pd.concat([shuff[e] for shuff in shuffled_diffs_list],axis=1).to_numpy().mean(axis=1)
                              for e in diff_cond_names}

    return shuffled_diffs_by_cond


def plot_pupil_diff_max_by_cond_non_modular(responses_by_cond: dict, conditions, plot=None, sess_list=None,
                                cond_line_kwargs=None, group_name='name',**kwargs):
    # Validate input
    if len(conditions) < 2:
        raise ValueError("At least two conditions must be provided.")

    if plot is None:
        pupil_max_diff_plot = plt.subplots()
    else:
        pupil_max_diff_plot = plot

    # If no session list is provided, use all sess_dict
    if sess_list is None:
        print('Using all sess_dict')
        sess_list = responses_by_cond[conditions[0]].index.get_level_values('sess').unique().tolist()

    # Initialize default arguments
    plot_kwargs = kwargs.get('plot_kwargs', {})
    if cond_line_kwargs is None:
        lss = ['-', '--', ':', '-.']
        cond_line_kwargs = {cond_i:{'c':'k','ls':lss[cond_i]} for cond_i, cond in enumerate(conditions)}

    # make sure name in multiindex
    add_name_to_response_dfs(responses_by_cond)

    # Group by session and calculate the median response
    mean_responses_by_cond = get_mean_responses_by_cond(conditions, responses_by_cond, sess_list,
                                                        smoothing_window=kwargs.get('smoothing_window',25))

    # get difference
    max_by_group = {
        cond: get_max_in_window(cond_diffs, kwargs.get('window_by_stim', (1.5, 2.5)), group_level=group_name)
        for cond, cond_diffs in mean_responses_by_cond.items()}

    max_diffs_by_group = {cond: get_response_diff(max_by_group[cond],
                                                  max_by_group[conditions[0]],
                                                  group_level=group_name,
                                                  **(kwargs.get('diff_kwargs', {})))
                          for cond in conditions[1:]}

    pupil_max_diff_plot[1].boxplot(list(max_diffs_by_group.values()),**plot_kwargs if plot_kwargs else {})

    if kwargs.get('permutation_test',False):
        shuffled_mean_responses_by_cond_list = [get_mean_responses_by_cond(conditions, responses_by_cond, sess_list,
                                                                     smoothing_window=kwargs.get('smoothing_window',25),
                                                                     shuffle=True)
                                           for _ in tqdm(range(kwargs.get('n_permutations',1000)),
                                                         desc='Shuffling conditions',
                                                         total=kwargs.get('n_permutations',1000))]
        shuffled_mean_responses_by_cond = {cond: pd.concat([shuffle[cond]
                                                            for shuffle in shuffled_mean_responses_by_cond_list])
                                            for cond in conditions}

        max_by_group_shuffled = {
            cond: get_max_in_window(cond_diffs, kwargs.get('window_by_stim', (1.5, 2.5)), group_level=group_name)
            for cond, cond_diffs in shuffled_mean_responses_by_cond.items()}

        max_diffs_by_group_shuffled = {cond: get_response_diff(max_by_group_shuffled[cond],
                                                               max_by_group_shuffled[conditions[0]],
                                                               group_level=group_name,
                                                               **(kwargs.get('diff_kwargs', {})))
                                       for cond in conditions[1:]}

        og_positions = plot_kwargs.get('positions',np.arange(len(max_diffs_by_group)))

        if plot_kwargs.get('positions'):
            plot_kwargs.pop('positions')
        if plot_kwargs.get('labels'):
            old_labels = plot_kwargs.get('labels')
            plot_kwargs['labels'] = [f'{lbl} shuffled' for lbl in old_labels]
        pupil_max_diff_plot[1].boxplot(list(max_diffs_by_group_shuffled.values()),
                                       positions=np.array(og_positions) +1.5,
                                       **plot_kwargs if plot_kwargs else {})


    # Set plot title and axis labels
    # pupil_max_diff_plot[1].set_title('Pupil response', **plot_kwargs.get('title', ''))
    # pupil_max_diff_plot[1].set_ylabel(plot_kwargs.get('ylabel', f'Difference ({conditions[0]} - {conditions[1]})'))

    if plot is None:
        return pupil_max_diff_plot, \
            [list(max_diffs_by_group.values()),list(max_diffs_by_group_shuffled.values()) if kwargs.get('permutation_test',False) else []]
    else:
        return list(max_diffs_by_group.values())


def plot_pupil_diff_max_by_cond(responses_by_cond: dict,
                                 conditions,
                                 plot=None,
                                 sess_list=None,
                                 cond_line_kwargs=None,
                                 group_name='name',
                                 **kwargs):

    if len(conditions) < 2:
        raise ValueError("At least two conditions must be provided.")

    if sess_list is None:
        print("Using all sess_dict")
        sess_list = responses_by_cond[conditions[0]].index.get_level_values('sess').unique().tolist()

    plot_kwargs = kwargs.get('plot_kwargs', {})
    scatter = kwargs.get('scatter', False)
    boxplot = kwargs.get('boxplot', True)

    # Get real max differences
    max_diffs_by_group = get_max_diffs_by_condition(
        responses_by_cond,
        conditions,
        sess_list=sess_list,
        group_name=group_name,
        **kwargs
    )

    # Set up plot
    if plot is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot

    # Plot real diffs
    plot_max_diffs(max_diffs_by_group,
                   plot=(fig, ax),
                   scatter=scatter,
                   boxplot=boxplot,
                   **plot_kwargs)

    max_diffs_by_group_shuffled = {}

    # Optional permutation testing
    if kwargs.get('permutation_test', False):
        shuffled_diffs = get_permuted_max_diffs(
            responses_by_cond,
            conditions,
            sess_list=sess_list,
            group_name=group_name,
            **kwargs
        )
        # Offset positions for shuffled boxplot
        og_positions = plot_kwargs.get('positions', np.arange(1, len(max_diffs_by_group) + 1))
        shuffle_plot_kwargs = plot_kwargs.copy()
        shuffle_plot_kwargs['positions'] = np.array(og_positions) + 1.5

        if 'labels' in shuffle_plot_kwargs:
            shuffle_plot_kwargs['labels'] = [f'{label} shuffled' for label in shuffle_plot_kwargs['labels']]

        # Plot shuffled
        plot_max_diffs(shuffled_diffs,
                       plot=(fig, ax),
                       scatter=scatter,
                       boxplot=boxplot,
                       **shuffle_plot_kwargs)

        max_diffs_by_group_shuffled = shuffled_diffs

    return (fig, ax), [list(max_diffs_by_group.values()),
                       list(max_diffs_by_group_shuffled.values()) if max_diffs_by_group_shuffled else []]
    # if plot is None:
    #     return (fig, ax), [list(max_diffs_by_group.values()),
    #                        list(max_diffs_by_group_shuffled.values()) if max_diffs_by_group_shuffled else []]
    # else:
    #     return list(max_diffs_by_group.values())


def decode_responses(predictors, features, model_name='logistic', n_runs=100, dec_kwargs=None):
    decoder = {dec_lbl: Decoder(np.vstack(predictors), np.hstack(features), model_name=model_name)
               for dec_lbl in ['data', 'shuffled']}
    dec_kwargs = dec_kwargs or {}
    [decoder[dec_lbl].decode(dec_kwargs=dec_kwargs | {'shuffle': shuffle_flag}, parallel_flag=False,
                             n_runs=n_runs)
     for dec_lbl, shuffle_flag in zip(decoder, [False, True])]

    return decoder


def save_responses_dicts(response_dict:dict, save_path:Path):
    pkl_dir = save_path.parent
    if not pkl_dir.is_dir():
        pkl_dir.mkdir()
    with open(save_path, 'wb') as file:
        pickle.dump(response_dict, file)


def load_pupil_sess_pkl(pupil_sess_pkl_path:Path):
    if pupil_sess_pkl_path.is_file():
        print(f'loading session pickle')
        sys_os = platform.system().lower()
        if sys_os == 'windows':
            import pathlib

            temp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
        if sys_os == 'linux':
            import pathlib

            temp = pathlib.WindowsPath
            pathlib.WindowsPath = pathlib.PosixPath

        sess_dict_dict = joblib.load(pupil_sess_pkl_path)
    else:
        sess_dict_dict = {}

    return sess_dict_dict


def init_pupil_td_obj(sess_dict: dict, sessname: str, ceph_dir:Path, all_sess_info:pd.DataFrame, td_path_pattern:str, home_dir:Path):
    sess_dict[sessname] = Session(sessname, ceph_dir)
    mouse_name = sessname.split('_')[0]
    sess_date = int(extract_date(sessname))
    sess_info = all_sess_info.query('name==@mouse_name & date==@sess_date').reset_index().query('sess_order=="main"').iloc[0]

    # get main sess pattern
    abs_td_path = None
    if 'tdata_file' in sess_info:
        td_path = home_dir.joinpath(*Path(sess_info['tdata_file']).parts[-5:])
        if td_path.is_file():
            abs_td_path = td_path
    if abs_td_path is None:
        main_sess_td_name = Path(sess_info['sound_bin'].replace('_SoundData', '_TrialData')).with_suffix(
            '.csv').name
        td_path = td_path_pattern.replace('<name>', main_sess_td_name.split('_')[0])
        abs_td_path_dir = home_dir / td_path
        abs_td_path = next(abs_td_path_dir.glob(f'{mouse_name}_TrialData_{sess_date}*.csv'))
    sess_dict[sessname].load_trial_data(abs_td_path)


def process_pupil_td_data(sesss_dict: dict, sessname:str, drug_sess_dict:dict,):
    sesss_dict[sessname].td_df['PatternPresentation_Rate'] = sesss_dict[sessname].td_df['PatternPresentation_Rate'].round(1)
    get_n_since_last(sesss_dict[sessname].td_df, 'Trial_Outcome', 0)
    [get_prate_block_num(sesss_dict[sessname].td_df, prate, rate_name) for prate, rate_name in
     zip([0.1, 0.9], ['frequent', 'rare'])]
    [get_prate_block_num(sesss_dict[sessname].td_df, prate, rate_name) for prate, rate_name in
     zip([0.1, 0.9], ['recent', 'distant'])]
    get_cumsum_columns(sesss_dict, sessname)
    times2process = ['Trial_Start', 'ToneTime', 'Trial_End', 'Gap_Time','Bonsai_Time']
    # rem if not in td_df

    [add_datetimecol(sesss_dict[sessname].td_df, col) for col in times2process
     if col in sesss_dict[sessname].td_df.columns]
    get_lick_in_patt_trials(sesss_dict[sessname].td_df, sessname)
    get_earlyX_trials(sesss_dict[sessname].td_df)
    # print(f'{sesss_dict[sessname].td_df.columns = }')
    filter_session(sesss_dict, sessname, [3, 4], drug_sess_dict)


def init_sess_pupil_obj(sess_dict: dict, sessname: str, ceph_dir: Path, all_sess_info: pd.DataFrame,
                        pupil_data_dict: dict, force_sync=True ):
    mouse_name, sess_date = sessname.split('_')[:2]
    try:
        sess_date = int(sess_date)
    except ValueError:
        sess_date = int(sess_date[:-1])
    sess_info = all_sess_info.query('name==@mouse_name & date==@sess_date').reset_index().query('sess_order=="main"').iloc[0]

    # get main sess pattern
    normal = get_main_sess_patterns(td_df=sess_dict[sessname].td_df)

    if 'beh_bin' not in sess_info or pd.isna(sess_info['beh_bin']):
        beh_events_path = None
    else:
        beh_bin_path = Path(sess_info['beh_bin'])

        beh_events_path = beh_bin_path.with_stem(f'{beh_bin_path.stem}_event_data_32').with_suffix('.csv')
        beh_events_path = ceph_dir / posix_from_win(str(beh_events_path))
    if 'sound_bin' not in sess_info or pd.isna(sess_info['sound_bin']):
        sound_write_path = None
    else:
        sound_bin_path = Path(sess_info['sound_bin'])

        sound_write_path = sound_bin_path.with_stem(f'{sound_bin_path.stem}_write_indices').with_suffix('.csv')
        sound_write_path = ceph_dir / posix_from_win(str(sound_write_path))

    sess_dict[sessname].init_pupil_obj(pupil_data_dict[sessname].pupildf, sound_write_path,
                                       beh_events_path, normal)

    if sess_date < 240101 and force_sync:
        beh_events_path_44 = beh_events_path.with_stem(f'{beh_bin_path.stem}_write_data').with_suffix('.csv')
        beh_events_path_44 = ceph_dir/posix_from_win(str(beh_events_path_44))
        try:
            sound_events_df = pd.read_csv(sound_write_path,)
            beh_events_44_df = pd.read_csv(beh_events_path_44,)
            sync_beh2sound(sess_dict[sessname].pupil_obj, beh_events_44_df, sound_events_df)
        except FileNotFoundError:
            print(f'ERROR {beh_events_path_44} not found')


def process_pupil_obj(sess_dict: dict, sessname: str, pupil_epoch_window=(-1, 3),
                          cond_filters: dict = get_all_cond_filts(),alignmethod='w_soundcard',**kwargs):
    normal = get_main_sess_patterns(td_df=sess_dict[sessname].td_df)

    main_patterns = get_main_sess_patterns(td_df=sess_dict[sessname].td_df)
    if sess_dict[sessname].td_df['Stage'].iloc[0] in [3, 5]:
        normal_patterns = main_patterns
        if len(normal_patterns) > 1 and sess_dict[sessname].td_df['Stage'].iloc[0] == 3:
            warnings.warn(f'{sessname} has more than one normal pattern for stage 3')
    elif sess_dict[sessname].td_df['Stage'].iloc[0] == 4:
        normal_patterns = get_main_sess_patterns(td_df=sess_dict[sessname].td_df.query(cond_filters['normal_exp']))
    else:
        return None
    # normal_patterns = [pattern for pattern in main_patterns if np.all(np.diff(pattern) > 0) or np.all(np.diff(pattern) < 0)]

    normal = normal_patterns[0]
    # none = idx 8 pip_diff = 0 then only times > 2 secs from t start
    base_idx = normal[0] - 2

    events = {e: {'idx': idx, 'filt': filt} for e, idx, filt in zip(['X', 'A', 'Start', 'none', ],
                                                                    [3, normal[0], base_idx, base_idx, ],
                                                                    ['', 'pip_counter==1', 'Time_diff>1',
                                                                     'Payload_diff==0&rand_n==1&d_X_times>3&d_X_times<6',

                                                                     ])}
    dev_patterns = [e for e in main_patterns if e != normal]
    dev_As = list(set([e[0] for e in dev_patterns]))
    dev_events = {f'A-{ei+1}': {'idx': idx, 'filt': 'pip_counter==1'} for ei, idx in enumerate(dev_As)} if dev_As else {}
    events.update(dev_events)

    align_kwargs = kwargs.get('align_kwargs',{})
    [sess_dict[sessname].get_pupil_to_event(e_dict['idx'], e, pupil_epoch_window,
                                            align_kwargs=dict(sound_df_query=e_dict['filt'], baseline_dur=1)|align_kwargs,
                                            alignmethod=alignmethod,)
     # alignmethod='w_td_df')
     for e, e_dict in events.items()]


def add_early_late_to_by_cond(
    by_cond_dict: dict,
    conds: list,
    early_late_dict: dict,
    sess_dict: dict,
    event: str,
    cond_filters: dict = None,
    td_df_query_template: str = "{cond_filter} & {col} {late_early_filt}",
    cumsum_suffix: str = "_cumsum"
) -> None:
    """
    Updates by_cond_dict with early/late/middle keys for each cond in conds, using early_late_dict.
    """
    filters_early_late = {}
    for cond in conds:
        col = f"{cond}{cumsum_suffix}"
        cond_filter = cond_filters[cond] if cond_filters is not None else ""
        for late_early, late_early_filt in early_late_dict.items():
            filt_expr = td_df_query_template.format(
                cond_filter=cond_filter,
                col=col,
                late_early_filt=late_early_filt.replace("<col>", col)
            )
            key = f"{cond}_{late_early}"
            filters_early_late[key] = filt_expr
    for cond_key, query in filters_early_late.items():
        by_cond_dict[cond_key] = group_pupil_across_sessions(
            sess_dict, list(sess_dict.keys()), event, cond_key, td_df_query=query
        )

def get_sliding_window_max(
    responses_by_cond: dict,
    conditions: list,
    td_df_query_template: str,
    trial_window_size: int,
    trial_stop: int,
    window_by_stim: tuple = (1.5, 2.5),
    mean_func=np.max,
    ):

    """ subsets responses by interatively adjusting query to get max over slinding window of trials. 

    """
    # Initialize a dictionary to hold the results
    sliding_window_max_mean = {cond: [] for cond in conditions}
    for cond in conditions:
        if cond not in responses_by_cond:
            raise ValueError(f"Condition {cond} not found in responses_by_cond.")
        cond_responses = responses_by_cond[cond]
        query_strings = [td_df_query_template.format(col=cond, start_i=start_i, end_i=start_i + trial_window_size)
                         for start_i in range(0, trial_stop, trial_window_size)]
        for query in query_strings:
            _df = cond_responses.query(query)
            # group by session and name
            _df = _df.groupby(['sess', 'name']).mean()
            if _df.empty:
                continue
            # Get the max over the specified window
            max_values = _df.loc[:, window_by_stim[0]:window_by_stim[1]].apply(mean_func, axis=1)


class PupilCondAnalysis:
    def __init__(self, by_cond_dict: dict, conditions: list, sess_list=None, **kwargs) -> None:
        """
        Store subset of by_cond_dict for the given conditions.
        """
        self.sess_list = sess_list

        self.max_diff_data = {}
        self.conditions = conditions
        self.by_cond = {k: by_cond_dict[k] for k in conditions if k in by_cond_dict}
        self.subset_responses_by_cond()
        self.shuffled_by_cond = {}
        self.plots = {}
        self.cluster_analysis = {}
        self.resp_diff = {}
        self.shuff_resp_diff = {}
        self.shuff_cluster_analysis = {}
        self.event_name = kwargs.get('event_name', 'pattern')

    def filter_sess_list(self,trial_threshold=5):
        sess2use = set()
        for cond, resps in self.by_cond.items():
            sess_above_thresh = [sess for sess in resps.index.get_level_values('sess').unique()
                                 if resps.xs(sess,level='sess').shape[0]>=trial_threshold and sess in self.sess_list]
            sess2use.update(sess_above_thresh)
        self.sess_list = sess2use

    def subset_responses_by_cond(self,):
        if self.sess_list is None:
            return
        for cond in self.conditions:
            self.by_cond[cond] = self.by_cond[cond].loc[self.by_cond[cond].index.get_level_values('sess').isin(self.sess_list)]

    def plot_ts_by_cond(self, cond_line_kwargs=None, group_name=['sess'], plot_kwargs=None):
        """
        Plot time series for each condition.
        """
        if plot_kwargs is None:
            plot_kwargs = {}
        if cond_line_kwargs is None:
            cond_line_kwargs = {}
        plot = plot_pupil_ts_by_cond(
            self.by_cond, self.conditions, sess_list=self.sess_list,
            cond_line_kwargs=cond_line_kwargs, group_name=group_name, plot_kwargs=plot_kwargs
        )
        if self.event_name == 'pattern':
            format_axis(plot[1],vspan=[[t, t + 0.15] for t in np.arange(0, 1, 0.25)])
            if any('dev' in cond for cond in self.conditions):
                plot[1].axvspan(0.5, 0.65, color='red', alpha=0.1)
        else:
            plot[1].axvline(0, color='k', ls='--',lw=0.5)
        self.plots['ts_by_cond'] = plot
        return plot

    def plot_diff_ts_by_cond(self, diff_conditions=None, cond_line_kwargs=None, group_level='sess', plot_kwargs=None):
        """
        Plot difference time series. diff_conditions: list, first is baseline, rest are subtracted from it.
        """
        if diff_conditions is None:
            diff_conditions = self.conditions
        if plot_kwargs is None:
            plot_kwargs = {}
        plot = plot_pupil_diff_ts_by_cond(
            self.by_cond, diff_conditions, sess_list=self.sess_list,
            cond_line_kwargs=cond_line_kwargs, group_level=group_level, plot_kwargs=plot_kwargs
        )
        self.plots['diff_ts_by_cond'] = plot
        return plot

    def plot_max_diff(self, diff_conditions=None, window_by_stim=(1.5,2.5), mean=np.max, group_name='name',
                      plot_kwargs=None, permutation_test=False, **kwargs):
        """
        Plot max difference over window. diff_conditions: list, first is baseline, rest are subtracted from it.
        Stores max_diff_data as a dict: {cond: [orig, (shuffled)]}
        """
        if diff_conditions is None:
            diff_conditions = self.conditions
        if plot_kwargs is None:
            plot_kwargs = {}
        plot, data = plot_pupil_diff_max_by_cond(
            self.by_cond, diff_conditions, sess_list=self.sess_list,
            window_by_stim=window_by_stim, mean=mean, group_name=group_name,
            plot_kwargs=plot_kwargs, permutation_test=permutation_test, **kwargs,
        )
        # data: [list of orig, list of shuffled (if permutation_test)]
        # Each is a list of arrays, one per cond (excluding baseline)
        max_diff_data = {}
        conds = diff_conditions[1:]
        for i, cond in enumerate(conds):
            if permutation_test:
                max_diff_data[f'{cond}-{diff_conditions[0]}'] = {'data': data[0][i], 'shuffled': data[1][i]}
            else:
                max_diff_data[f'{cond}-{diff_conditions[0]}'] = {'data': data[0][i]}
        self.plots['max_diff'] = plot
        self.max_diff_data = max_diff_data
        return plot, max_diff_data

    def ttest_max_diff(self, keys1, keys2,  **kwargs):
        """
        Perform independent t-test on max_diff_data between two keys (conditions).
        If perm=True, uses shuffled data (index 1), else uses original (index 0).
        Returns ttest result.
        """
        data1 = self.max_diff_data[keys1[0]][keys1[1]]
        data2 = self.max_diff_data[keys2[0]][keys2[1]]
        return ttest_ind(data1, data2, **kwargs)

    def shuffle_by_cond(self, n_permutations=500, rng_seed=None):
        sess_shuffled_resps = {}
        unique_sess = []
        for cond in self.conditions:
            unique_sess.extend(self.by_cond[cond].index.get_level_values('sess').tolist())
            sess_shuffled_resps[cond] = {}

        unique_sess = set(unique_sess)

        for sess in unique_sess:
            resps_by_cond = {cond: self.by_cond[cond].xs(sess,level='sess') for cond in self.conditions}
            n_by_cond = [resps.shape[0] for resps in resps_by_cond.values()]
            cond_break_idxs = np.cumsum(n_by_cond)
            resps_by_cond_stacked = np.vstack(list(resps_by_cond.values())).astype(np.float32)

            rng = np.random.default_rng(rng_seed)
            idxs = [rng.permutation(resps_by_cond_stacked.shape[0]) for _ in range(n_permutations)]
            shuffled_resps_by_cond = [np.split(resps_by_cond_stacked.copy()[idx],
                                               cond_break_idxs[:-1],axis=0) for idx in idxs]

            for cond_i, cond in enumerate(resps_by_cond):
                perm_resps = [shuff[cond_i] for shuff in shuffled_resps_by_cond]
                sess_shuffled_resps[cond][sess] = perm_resps

        self.shuffled_by_cond = sess_shuffled_resps

    def compute_resp_diffs(self, ref_cond=0, subset_comps=None, group_name='name', **kwargs):
        if isinstance(ref_cond, int):
            ref_cond = self.conditions[ref_cond]
        conds_4_comp = [cond for cond in self.conditions if cond != ref_cond]

        if subset_comps is not None:
            subset_comps = [e if isinstance(e,str) else self.conditions[e] for e in subset_comps]
            conds_4_comp = [cond for cond in conds_4_comp if cond in subset_comps]

        ref_cond_resp = self.by_cond[ref_cond].groupby(group_name).mean()

        resp_window = ref_cond_resp.columns.values[[0, -1]]
        if kwargs.get('resp_window') is not None:
            resp_window = kwargs.get('resp_window', resp_window)

        for cond in conds_4_comp:
            comp_name = '_vs_'.join([cond, str(ref_cond)])
            comp_cond_resp = self.by_cond[cond].groupby(group_name).mean()
            resp_diff = (comp_cond_resp.loc[:,resp_window[0]:resp_window[1]] - ref_cond_resp.loc[:,resp_window[0]:resp_window[1]]).mean(axis=0)
            self.resp_diff[comp_name] = resp_diff

    def compute_shuffled_resp_diffs(self, ref_cond=0, comp_conds=None, group_name='name', **kwargs):
        if isinstance(ref_cond, int):
            ref_cond = self.conditions[ref_cond]
        conds_4_comp = [cond for cond in self.conditions if cond != ref_cond]
        if comp_conds is not None:
            conds_4_comp = [cond for cond in conds_4_comp if cond in comp_conds]

        ref_cond_shuffs: dict = self.shuffled_by_cond[ref_cond]
        n_shuffs = len(list(ref_cond_shuffs.values())[0])
        for cond in conds_4_comp:
            comp_name = '_vs_'.join([cond, str(ref_cond)])
            ref_cond_resps = [pd.concat(
                [pd.DataFrame(shuff_resps[i], index=pd.Index([sess] * len(shuff_resps[i]), name='sess'))
                 for sess, shuff_resps in ref_cond_shuffs.items()],
                axis=0,).groupby('sess').mean()
            for i in range(n_shuffs)]
            comp_cond_resps = [pd.concat(
                [pd.DataFrame(shuff_resps[i], index=pd.Index([sess] * len(shuff_resps[i]), name='sess'))
                 for sess, shuff_resps in self.shuffled_by_cond[cond].items()],
                axis=0,).groupby('sess').mean()
            for i in range(n_shuffs)]

            resp_window = list(ref_cond_resps[0].columns.values[[0, -1]])
            if kwargs.get('resp_window') is not None:
                resp_window = kwargs.get('resp_window', resp_window)

            self.shuff_resp_diff[comp_name] = [
                comp_cond_resp.loc[:, resp_window[0]:resp_window[1]].mean(axis=0) - ref_cond_resp.loc[:, resp_window[0]:resp_window[1]].mean(axis=0)
                for ref_cond_resp, comp_cond_resp in zip(ref_cond_resps, comp_cond_resps)
            ]

    def compute_clusters(self,ref_cond=0,group_name='sess',subset_comps=None, **kwargs):

        if isinstance(ref_cond, int):
            ref_cond = self.conditions[ref_cond]
        conds_4_comp = [cond for cond in self.conditions if cond != ref_cond]
        if subset_comps is not None:
            subset_comps = [e if isinstance(e,str) else self.conditions[e] for e in subset_comps]
            conds_4_comp = [cond for cond in conds_4_comp if cond in subset_comps]

        ref_cond_resp = self.by_cond[ref_cond].loc[:,0:2.5].groupby(group_name).mean().values

        for cond in conds_4_comp:
            comp_name = '_vs_'.join([cond, str(ref_cond)])
            comp_cond_resp = self.by_cond[cond].loc[:,0:2.5].groupby(group_name).mean().values
            self.cluster_analysis[comp_name] = cluster_analysis(ref_cond_resp, comp_cond_resp, **kwargs)

    def compute_shuffled_clusters(self,ref_cond=0,comp_conds=None,group_name='sess', **kwargs):

        if isinstance(ref_cond, int):
            ref_cond = self.conditions[ref_cond]
        conds_4_comp = [cond for cond in self.conditions if cond != ref_cond]
        if comp_conds is not None:
            conds_4_comp = [cond for cond in conds_4_comp if cond in comp_conds]

        ref_cond_shuffs: dict = self.shuffled_by_cond[ref_cond]
        n_shuffs = len(list(ref_cond_shuffs.values())[0])
        for cond in conds_4_comp:
            comp_name = '_vs_'.join([cond, str(ref_cond)])
            ref_cond_resps = [pd.concat(
                [pd.DataFrame(shuff_resps[i], index=pd.Index([sess] * len(shuff_resps[i]), name='sess'))
                 for sess, shuff_resps in ref_cond_shuffs.items()],
                axis=0,).loc[:,0:2.5].groupby('sess').mean().values
            for i in range(n_shuffs)]
            comp_cond_resps = [pd.concat(
                [pd.DataFrame(shuff_resps[i], index=pd.Index([sess] * len(shuff_resps[i]), name='sess'))
                 for sess, shuff_resps in self.shuffled_by_cond[cond].items()],
                axis=0,).loc[:,0:2.5].groupby('sess').mean().values
            for i in range(n_shuffs)]
            self.shuff_cluster_analysis[comp_name] = [
                cluster_analysis(ref_cond_resp, comp_cond_resp, **kwargs)
                for ref_cond_resp, comp_cond_resp in tqdm(zip(ref_cond_resps,comp_cond_resps,),
                                                          total=len(conds_4_comp),desc='computing shuffled masses')
            ]

    def save_plot(self, plot_key: str, savename: str):
        """
        Save the plot with the given key to the specified absolute path.
        """
        plot = self.plots.get(plot_key)
        if plot is not None:
            plot[0].savefig(savename)

    def save_ttest_to_tex(self, ttest_result, tex_path):
        """
        Save ttest result to a .tex file using save_stats_to_tex from save_utils.
        """
        from save_utils import save_stats_to_tex
        save_stats_to_tex(ttest_result, tex_path)


def run_pupil_cond_analysis(
    by_cond_dict, sess_list, conditions, figdir, line_kwargs, boxplot_kwargs,
    window_by_stim=(1, 2.9), smoothing_window=25, max_window_kwargs=None,
    group_name='name', n_permutations=500, stats_dir=None,fig_savename='', tex_name=None,
    ylabel="Δ pupil size", figsize=(2.2, 1.8), ylim_ts=None, ylim_maxdiff=None, **kwargs):
    analysis = PupilCondAnalysis(
        by_cond_dict=by_cond_dict, sess_list=sess_list, conditions=conditions,
        event_name=kwargs.get('event_name', 'pattern'),
    )
    analysis.plot_ts_by_cond(cond_line_kwargs=line_kwargs)

    # analysis.filter_sess_list()
    cluster_comps = kwargs.get('cluster_comps')
    if kwargs.get('cluster_comps') is not None:
        for comp_ids in cluster_comps:
            analysis.compute_clusters(ref_cond=comp_ids[0],subset_comps=comp_ids[1:],
                                      group_name=kwargs.get('cluster_groupname', 'sess'),
                                      p_alpha=kwargs.get('p_alpha', 0.05))

            analysis.compute_resp_diffs(ref_cond=comp_ids[0],subset_comps=comp_ids[1:],
                                       group_name=kwargs.get('cluster_groupname', 'sess'),
                                       resp_window=kwargs.get('resp_window', [0,2.9]),)
    else:
        analysis.compute_clusters(group_name=kwargs.get('cluster_groupname','sess'),
                                  p_alpha=kwargs.get('p_alpha',0.05))
        analysis.compute_resp_diffs(group_name=kwargs.get('cluster_groupname','sess'),
                                    resp_window=kwargs.get('resp_window',[0,2.9]),)

    x_ser = list(analysis.by_cond.values())[0].columns.values


    if kwargs.get('inset_max_diff', True):
        axinset = analysis.plots['ts_by_cond'][0].add_axes([0.85, 0.65, 0.15, 0.35])
        axinset.patch.set_facecolor('white')
        axinset.patch.set_alpha(0.8)
        diff_quant_plot = (analysis.plots['ts_by_cond'][0], axinset)
    else:
        diff_quant_plot = None

    for comp_i,comp_name in enumerate(analysis.cluster_analysis):
        condA_quant = analysis.cluster_analysis[comp_name][0]
        condB_quant = analysis.cluster_analysis[comp_name][1]
        
        plot_clusters(x_ser,condA_quant,condB_quant,
                      analysis.plots['ts_by_cond'],p_alpha=kwargs.get('p_alpha',0.05),
                      plot_y=-0.05 - (0.025*comp_i),plot_kwargs=line_kwargs[comp_name.split('_vs_')[0]],)

        analysis.shuffle_by_cond(kwargs.get('n_permutations', 500))
        comp_conds = comp_name.split('_vs_')[::-1]
        analysis.compute_shuffled_clusters(comp_conds[0],comp_conds[1:],p_alpha=kwargs.get('p_alpha',0.05))
        analysis.compute_shuffled_resp_diffs(comp_conds[0],comp_conds[1:], 
                                             resp_window=kwargs.get('resp_window',[0,2.9]),)
        cluster_mass_from_shuff = [shuff[2] for shuff in analysis.shuff_cluster_analysis[comp_name]]
        cluster_mass_ttest = ttest_1samp([np.sum(shuff) for shuff in cluster_mass_from_shuff],
                                         np.sum(analysis.cluster_analysis[comp_name][2]),
                                         alternative='less')
        plot_cluster_stats(analysis.cluster_analysis[comp_name][2],cluster_mass_from_shuff,
                           plot=diff_quant_plot,)
        # move ylabel to right side of inset
        if diff_quant_plot is not None:
            # diff_quant_plot[1].set_ylabel(ylabel, rotation=-90, labelpad=2)`
            diff_quant_plot[1].yaxis.set_label_position("right")
            diff_quant_plot[1].yaxis.tick_right()
            #set spine visible for right and not left
            diff_quant_plot[1].spines["right"].set_visible(True)
            diff_quant_plot[1].spines["left"].set_visible(False)
            # format_axis(diff_quant_plot[1])
        # resp_diff_from_shuff = [shuff.values for shuff in analysis.shuff_resp_diff[comp_name]]
        # cluster_mass_ttest = ttest_1samp([np.sum(shuff) for shuff in resp_diff_from_shuff],
                                        #  np.sum(analysis.resp_diff[comp_name].values),
                                        #  alternative='less')
        # print(f'{analysis.resp_diff[comp_name].values.shape = }')
        # print(f'{analysis.shuff_resp_diff[comp_name] = }')
        # plot_cluster_stats(analysis.resp_diff[comp_name].values,analysis.shuff_resp_diff[comp_name],
                        #    plot=diff_quant_plot,)

        print(f'comp_name: {comp_name} cluster_mass_ttest: {cluster_mass_ttest}')
        tex_fp = stats_dir/tex_name
        save_stats_to_tex(cluster_mass_ttest,tex_fp.with_stem(tex_fp.stem+'mass_ttest'))

        # analysis.shuff_cluster_analysis[comp_name] = None  # to save memory
    analysis.shuffled_by_cond = {}  # to save memory

    # analysis.plot_max_diff(
    #     window_by_stim=window_by_stim, permutation_test=kwargs.get('permutation_test',True),
    #     smoothing_window=smoothing_window, max_window_kwargs=max_window_kwargs or {'max_func': 'max'},
    #     group_name=group_name, n_permutations=n_permutations,
    #     plot_kwargs=boxplot_kwargs,
    #     plot=diff_quant_plot,
    # )
    if kwargs.get('permutation_test', True):
        ttest = analysis.ttest_max_diff(
            [f'{conditions[1]}-{conditions[0]}', 'data'],
            [f'{conditions[1]}-{conditions[0]}', 'shuffled'],
            alternative='greater',
        )
        print(f'{tex_name} ttest: {ttest}')
        if stats_dir and tex_name:
            save_stats_to_tex(ttest, stats_dir / tex_name)
    format_axis(analysis.plots['ts_by_cond'][1])
    if ylim_ts is not None:
        analysis.plots['ts_by_cond'][1].set_ylim(*ylim_ts)
    xlim_ts = kwargs.get('xlim_ts', None)
    if xlim_ts is not None:
        analysis.plots['ts_by_cond'][1].set_xlim(xlim_ts)
    analysis.plots['ts_by_cond'][1].set_title('')
    # if ylim_maxdiff is not None:
        # analysis.plots['max_diff'][1].set_ylim(*ylim_maxdiff)
    # analysis.plots['max_diff'][1].set_title('')
    # analysis.plots['max_diff'][1].spines["right"].set_visible(True)
    # analysis.plots['max_diff'][1].spines["top"].set_visible(True)
    # analysis.plots['max_diff'][1].yaxis.set_label_position("right")
    # analysis.plots['max_diff'][1].yaxis.tick_right()
    # analysis.plots['max_diff'][1].set_ylabel(ylabel)
    if analysis.plots['ts_by_cond'][1].get_legend():
        analysis.plots['ts_by_cond'][1].get_legend().remove()
    analysis.plots['ts_by_cond'][0].set_size_inches(*figsize)

    analysis.plots['ts_by_cond'][0].show()
    # format_axis(analysis.plots['max_diff'][1])
    # analysis.plots['max_diff'][0].show()
    analysis.plots['ts_by_cond'][0].savefig(figdir / fig_savename)
    # if not kwargs.get('inset_max_diff', True):
    #     analysis.plots['max_diff'][0].savefig(figdir /fig_savename.replace('ts','maxdiff'))
    return analysis


if __name__ == "__main__":
    pass