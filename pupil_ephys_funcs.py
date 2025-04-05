"""
This module contains functions for analyzing pupil and electrophysiology data.

Functions:
- group_pupil_across_sessions: Groups pupil data across sessions based on conditions.
- get_pupil_diff_by_session: Calculates the difference in pupil data between two conditions by session.
- plot_pupil_diff_across_sessions: Plots the difference in pupil responses across sessions.
- plot_pupil_ts_by_cond: Plots pupil time series by condition.
- plot_pupil_diff_ts_by_cond: Plots the difference in pupil time series between conditions.
- add_name_to_response_dfs: Adds session names to the multi-index of response DataFrames.
- get_mean_responses_by_cond: Calculates mean responses by condition.
- get_response_diff: Computes the difference between two response conditions.
- get_max_diffs: Finds the maximum differences in responses within a specified window.
- plot_pupil_max_diff_bysess_by_cond: Plots the maximum differences in pupil responses by session and condition.
- plot_pupil_diff_max_by_cond: Plots the maximum differences in pupil responses between conditions.
- decode_responses: Decodes responses using machine learning models.
- save_responses_dicts: Saves response dictionaries to a file.
- load_pupil_sess_pkl: Loads pupil session data from a pickle file.
- init_pupil_td_obj: Initializes trial data objects for pupil analysis.
- process_pupil_td_data: Processes trial data for pupil analysis.
- init_sess_pupil_obj: Initializes session pupil objects.
- process_pupil_obj: Processes pupil data for a session.

"""

import os

from behviour_analysis_funcs import sync_beh2sound, get_all_cond_filts, add_datetimecol, get_drug_dates, \
    get_n_since_last, get_earlyX_trials, get_prate_block_num, filter_session, get_cumsum_columns, \
    get_lick_in_patt_trials
from ephys_analysis_funcs import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import platform
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm
from scipy.stats import sem, ttest_ind


def group_pupil_across_sessions(sessions_objs: dict,sessnames:list,event:str, cond_name:str,cond_filters=None,
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
        assert isinstance(sessions_objs[sessname],Session)
        if sessions_objs[sessname].pupil_obj is None:
            continue
        if sessions_objs[sessname].pupil_obj.aligned_pupil is None or \
                len(sessions_objs[sessname].pupil_obj.aligned_pupil) == 0:
            continue
        if not use_all:
            responses = sessions_objs[sessname].pupil_obj.aligned_pupil[event]
            trial_nums = sessions_objs[sessname].td_df.query(cond_filter).index.get_level_values('trial_num').values
            cond_pupil = responses.loc[responses.index.isin(trial_nums, level='trial')]
            # if sessname=='DO79_240118':
            #     print(sessname,event,cond_name,cond_filter,cond_pupil.shape)
            # cond_pupil = sessions_objs[sessname].pupil_obj.aligned_pupil[event].query('trial in @trial_nums')
        else:
            cond_pupil = sessions_objs[sessname].pupil_obj.aligned_pupil[event]
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
                          plot_mean=True, cond_line_kwargs=None, group_name=['sess']):
    if plot is None:
        pupil_ts_plot = plt.subplots()
    else:
        pupil_ts_plot = plot

    x_ser = responses_by_cond[conditions[0]].columns

    # group_by_names = ['sess'] if group_name == 'sess' else ['sess',group_name]
    group_by_names = group_name
    lss= ['-', '--']
    # for cond_i, cond in enumerate(['alternating', 'random']):
    if sess_list is None:
        print('Using all sessions')
        sess_list = responses_by_cond[conditions[0]].index.get_level_values('sess').unique().tolist()
    for cond_i, cond in enumerate(conditions):
        responses_df = responses_by_cond[cond]
        if 'name' not in responses_df.index.names:
            responses_df['name'] = responses_df.index.get_level_values('sess').str.split('_').str[0]
            responses_df.set_index(['name'], append=True, inplace=True)
        print(f'{responses_df.index.names = }')
        mean_responses_by_sess = responses_df.query('sess in @sess_list').groupby(level=group_by_names).median()
        # if group_name != 'sess':
        mean_responses_by_sess = mean_responses_by_sess.groupby(level=group_name).mean()
        # add name to multiindex
        print(f'{cond} {mean_responses_by_sess.shape} sessions')
        if plot_indv_sess:
            [pupil_ts_plot[1].plot(x_ser,sess_response,alpha=0.25,
                                   c=cond_line_kwargs[cond]['c'] if cond_line_kwargs else f'C{cond_i}',)
             for _,sess_response in mean_responses_by_sess.iterrows()]
        if plot_mean:
            pupil_ts_plot[1].plot(x_ser, mean_responses_by_sess.mean(axis=0), label=cond,
                                  **cond_line_kwargs[cond] if cond_line_kwargs else {})
            # plot sems
            if len(mean_responses_by_sess) > 1:
                # sem over sessions
                pupil_ts_plot[1].fill_between(x_ser.tolist(),
                                              mean_responses_by_sess.mean(axis=0) - mean_responses_by_sess.sem(axis=0),
                                              mean_responses_by_sess.mean(axis=0) + mean_responses_by_sess.sem(axis=0),
                                              fc=cond_line_kwargs[cond]['c'] if cond_line_kwargs else f'C{cond_i}',
                                              alpha=0.1)
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
                pupil_ts_plot[1].fill_between(x_ser.tolist(),
                                              mean_responses_by_sess.mean(axis=0) - sess_sem,
                                              mean_responses_by_sess.mean(axis=0) + sess_sem,
                                              fc=cond_line_kwargs[cond]['c'] if cond_line_kwargs else f'C{cond_i}',
                                              alpha=0.1)


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

    # If no session list is provided, use all sessions
    if sess_list is None:
        print('Using all sessions')
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


def get_mean_responses_by_cond(conds, response_dict, sess_list, smoothing_window=25,shuffle=False):
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
    mean_responses_by_sess_by_cond = {cond: resps.groupby(['sess', 'name']).median()
                                      for cond, resps in all_responses_by_cond.items()}

    return mean_responses_by_sess_by_cond

def get_response_diff(response1, response2, group_level,sub_by_sess=False):
    if not sub_by_sess:
        # print(response1.groupby(group_level).mean() - response2.groupby(group_level).mean())
        return response1.groupby(group_level).mean() - response2.groupby(group_level).mean()
    else:
        return response1.groupby(['sess','name']).mean() - response2.groupby(['sess','name']).mean()

def get_max_diffs(responses_diff: pd.DataFrame, diff_window: list,group_level,sub_by_sess=False):
    # print(responses_diff)
    max_diffs = responses_diff.loc[:, diff_window[0]:diff_window[1]].max(axis=1).groupby(group_level).mean()

    return max_diffs


def plot_pupil_max_diff_bysess_by_cond(responses_by_cond_by_stim: [dict], conditions, plot=None, sess_list=None, plot_kwargs=None,
                                cond_line_kwargs=None, window_by_stim=(1.5,2.5),mean=np.nanmean,group_name='name'):
    # Validate input
    if len(conditions) < 2:
        raise ValueError("At least two conditions must be provided.")

    if plot is None:
        pupil_max_diff_plot = plt.subplots()
    else:
        pupil_max_diff_plot = plot

    # If no session list is provided, use all sessions
    if sess_list is None:
        print('Using all sessions')
        sess_list = responses_by_cond_by_stim[0][conditions[0]].index.get_level_values('sess').unique().tolist()

    # Initialize default arguments
    if plot_kwargs is None:
        plot_kwargs = {}
    if cond_line_kwargs is None:
        lss = ['-', '--', ':', '-.']
        cond_line_kwargs = {cond_i:{'c':'k','ls':lss[cond_i]} for cond_i, cond in enumerate(conditions)}


    # Group by session and calculate the median response
    responses_by_cond_by_sess_4_stim = [[
        responses_by_cond[cond].loc[responses_by_cond[cond].index.get_level_values('sess').isin(sess_list)].T.rolling(10).mean().T.groupby(
            level='sess').median()
        for cond in conditions]
    for responses_by_cond in responses_by_cond_by_stim]

    # add name to multiindex
    for responses_by_cond_by_sess in responses_by_cond_by_sess_4_stim:
        for response_df in responses_by_cond_by_sess:
            response_df['name'] = response_df.index.get_level_values('sess').str.split('_').str[0]
            response_df.set_index(['name'], append=True, inplace=True)

    # Calculate differences from the baseline condition
    by_sess_diffs_4_stim =[ [responses_by_cond_by_sess[0].sub(responses, level='sess')
                     for responses in responses_by_cond_by_sess[1:]]
                     for responses_by_cond_by_sess in responses_by_cond_by_sess_4_stim]
    for by_sess_diffs in by_sess_diffs_4_stim:
        for by_sess_diff in by_sess_diffs:
            by_sess_diff.columns = by_sess_diff.columns.astype(float)

    # print(by_sess_bootstrap[0].shape)
    # Plot the mean difference and standard error for each condition
    window_means_by_cond = [[[mean(group_resp[window[0]:window[1]])
                            for _, group_resp in by_sess_diff.groupby(group_name).mean().iterrows()]
                            for by_sess_diff in by_sess_diffs]
                            for by_sess_diffs,window in zip(by_sess_diffs_4_stim,window_by_stim)]
    pupil_max_diff_plot[1].boxplot(sum([e for e in window_means_by_cond],[]),**plot_kwargs if plot_kwargs else {})


    # Set plot title and axis labels
    pupil_max_diff_plot[1].set_title('Pupil response', **plot_kwargs.get('title', {}))
    pupil_max_diff_plot[1].set_ylabel(plot_kwargs.get('ylabel', f'Difference ({conditions[0]} - {conditions[1]})'))

    # # Add shaded regions
    # for t in np.arange(0, 1, 0.25):
    #     pupil_max_diff_plot[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1)

    # Tight layout and show plot
    plt.tight_layout()
    # plt.show()

    if plot is None:
        return pupil_max_diff_plot, sum([e for e in window_means_by_cond],[])
    else:
        return sum([e for e in window_means_by_cond],[])


def plot_pupil_diff_max_by_cond(responses_by_cond: dict, conditions, plot=None, sess_list=None,
                                cond_line_kwargs=None, group_name='name',**kwargs):
    # Validate input
    if len(conditions) < 2:
        raise ValueError("At least two conditions must be provided.")

    if plot is None:
        pupil_max_diff_plot = plt.subplots()
    else:
        pupil_max_diff_plot = plot

    # If no session list is provided, use all sessions
    if sess_list is None:
        print('Using all sessions')
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
    mean_diff_by_group = {cond: get_response_diff(mean_responses_by_cond[cond],
                                                  mean_responses_by_cond[conditions[0]],
                                                  group_level=group_name,
                                                  **(kwargs.get('diff_kwargs', {})))
                                for cond in conditions[1:]}
    max_diffs_by_group = {cond: get_max_diffs(cond_diffs,kwargs.get('window_by_stim',(1.5,2.5)),group_level=group_name)
                          for cond,cond_diffs in mean_diff_by_group.items()}

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

        mean_diff_by_group_shuffled = {cond: get_response_diff(shuffled_mean_responses_by_cond[cond],
                                                                shuffled_mean_responses_by_cond[conditions[0]],
                                                                group_level=group_name,
                                                               **(kwargs.get('diff_kwargs', {})))
                                       for cond in conditions[1:]}
        max_diffs_by_group_shuffled = {cond: get_max_diffs(cond_diffs,kwargs.get('window_by_stim',(1.5,2.5)),group_level=group_name)
                                        for cond,cond_diffs in mean_diff_by_group_shuffled.items()}
        og_positions = plot_kwargs.get('positions',np.arange(len(max_diffs_by_group)))
        if plot_kwargs.get('positions'):
            plot_kwargs.pop('positions')
        pupil_max_diff_plot[1].boxplot(list(max_diffs_by_group_shuffled.values()),
                                       positions=np.array(og_positions)+len(max_diffs_by_group),
                                       **plot_kwargs if plot_kwargs else {})


    # Set plot title and axis labels
    pupil_max_diff_plot[1].set_title('Pupil response', **plot_kwargs.get('title', {}))
    pupil_max_diff_plot[1].set_ylabel(plot_kwargs.get('ylabel', f'Difference ({conditions[0]} - {conditions[1]})'))

    if plot is None:
        return pupil_max_diff_plot, \
            [list(max_diffs_by_group.values()),list(max_diffs_by_group_shuffled.values()) if kwargs.get('permutation_test',False) else []]
    else:
        return list(max_diffs_by_group.values())


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
        with open(pupil_sess_pkl_path, 'rb') as f:
            sessions_dict = pickle.load(f)
    else:
        sessions_dict = {}

    return sessions_dict


def init_pupil_td_obj(sess_dict: dict, sessname: str, ceph_dir:Path, all_sess_info:pd.DataFrame, td_path_pattern:str, home_dir:Path):
    sess_dict[sessname] = Session(sessname, ceph_dir)
    mouse_name, sess_date = sessname.split('_')[:2]
    sess_date = int(sess_date)
    sess_info = all_sess_info.query('name==@mouse_name & date==@sess_date').reset_index().query('sess_order=="main"').iloc[0]

    # get main sess pattern
    main_sess_td_name = Path(sess_info['sound_bin'].replace('_SoundData', '_TrialData')).with_suffix(
        '.csv').name
    td_path = td_path_pattern.replace('<name>', main_sess_td_name.split('_')[0])
    abs_td_path_dir = home_dir / td_path
    abs_td_path = next(abs_td_path_dir.glob(f'{mouse_name}_TrialData_{sess_date}*.csv'))
    sess_dict[sessname].load_trial_data(abs_td_path)


def process_pupil_td_data(sesss_dict: dict, sessname:str, drug_sess_dict:dict,):
    get_n_since_last(sesss_dict[sessname].td_df, 'Trial_Outcome', 0)
    [get_prate_block_num(sesss_dict[sessname].td_df, prate, rate_name) for prate, rate_name in
     zip([0.1, 0.9], ['frequent', 'rare'])]
    [get_prate_block_num(sesss_dict[sessname].td_df, prate, rate_name) for prate, rate_name in
     zip([0.1, 0.9], ['recent', 'distant'])]
    get_cumsum_columns(sesss_dict, sessname)
    [add_datetimecol(sesss_dict[sessname].td_df, col) for col in ['Trial_Start', 'ToneTime', 'Trial_End', 'Gap_Time',
                                                                'Bonsai_Time']]
    get_lick_in_patt_trials(sesss_dict[sessname].td_df, sessname)
    get_earlyX_trials(sesss_dict[sessname].td_df)
    # print(f'{sesss_dict[sessname].td_df.columns = }')
    filter_session(sesss_dict, sessname, [3, 4], drug_sess_dict)


def init_sess_pupil_obj(sess_dict: dict, sessname: str, ceph_dir: Path, all_sess_info: pd.DataFrame,
                        pupil_data_dict: dict, ):
    mouse_name, sess_date = sessname.split('_')[:2]
    sess_date = int(sess_date)
    sess_info = all_sess_info.query('name==@mouse_name & date==@sess_date').reset_index().query('sess_order=="main"').iloc[0]
    sound_bin_path = Path(sess_info['sound_bin'])
    beh_bin_path = Path(sess_info['beh_bin'])

    # get main sess pattern
    normal = get_main_sess_patterns(td_df=sess_dict[sessname].td_df)

    sound_write_path = sound_bin_path.with_stem(f'{sound_bin_path.stem}_write_indices').with_suffix('.csv')
    sound_write_path = ceph_dir / posix_from_win(str(sound_write_path))
    beh_events_path = beh_bin_path.with_stem(f'{beh_bin_path.stem}_event_data_32').with_suffix('.csv')
    beh_events_path = ceph_dir / posix_from_win(str(beh_events_path))

    if not beh_events_path.is_file():
        print(f'ERROR {beh_events_path} not found')
        return None
    sess_dict[sessname].init_pupil_obj(pupil_data_dict[sessname].pupildf, sound_write_path,
                                       beh_events_path, normal)


def process_pupil_obj(sess_dict: dict, sessname: str, pupil_epoch_window=(-1, 3),
                          cond_filters: dict = get_all_cond_filts(),alignmethod='w_td_df'):
    normal = get_main_sess_patterns(td_df=sess_dict[sessname].td_df)

    [sess_dict[sessname].get_pupil_to_event(e_idx, e_name, [-1, 3], alignmethod=alignmethod,
                                            align_kwargs=dict(baseline_dur=1, ))  # size_col='canny_raddi_a_zscored'
     for e_idx, e_name in tqdm(zip([3, normal[0][0]], ['X', 'A']), total=2, desc='pupil data processing')]

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

    events = {e: {'idx': idx, 'filt': filt} for e, idx, filt in zip(['X', 'A', 'Start', 'none', 'deviant_A'],
                                                                    [3, normal[0], base_idx, base_idx, normal[0]],
                                                                    ['', 'pip_counter==1', 'Time_diff>1',
                                                                     'Payload_diff==0&rand_n==1&d_X_times>3&d_X_times<6',
                                                                     'ptype==1 & pip_counter==1'
                                                                     ])}
    [sess_dict[sessname].get_pupil_to_event(e_dict['idx'], e, pupil_epoch_window,
                                            align_kwargs=dict(sound_df_query=e_dict['filt'], baseline_dur=1),
                                            alignmethod='w_soundcard')
     # alignmethod='w_td_df')
     for e, e_dict in events.items()]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])

    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['axes.spines.top'] = False
    matplotlib.rcParams['legend.fontsize'] = 13
    matplotlib.rcParams['axes.titlesize'] = 16
    matplotlib.rcParams['axes.labelsize'] = 14
    matplotlib.rcParams['xtick.labelsize'] = 13
    matplotlib.rcParams['ytick.labelsize'] = 13
    matplotlib.rcParams['figure.labelsize'] = 18


    # sess_topology_path = ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_ephys_2401.csv')
    sess_topology_path = ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_musc_2406.csv')
    # sess_topology_path = ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_musc_may23.csv')
    # sess_topology_path = ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_musc_sept23.csv')
    # sess_topology_path = ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_fam_jan23.csv')

    # try:
    #     gen_metadata(sess_topology_path,ceph_dir,col_name='beh_bin',harp_bin_dir='')
    # except OSError:
    #     pass
    session_topology = pd.read_csv(sess_topology_path)
    sessions = {}
    all_sess_info = session_topology.query('sess_order=="main"')

    cohort_tag = "_".join(sess_topology_path.stem.split(('_'))[-2:])

    figdir = ceph_dir / 'Dammy' / 'figures' / 'jan_23_animals'
    if not figdir.is_dir():
        figdir.mkdir()

    pkls = [ceph_dir / posix_from_win(p) for p in [
        # r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_ephys_2024_allsess_fam_2d_90Hz_hpass01_lpass4hanning015_TOM.pkl',
        # r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_ephys_2024_allsess_v3_fam_2d_90Hz_hpass01_lpass4_TOM.pkl',
        r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_ephys_2024_allsess_new_out_method_fam_2d_90Hz_hpass01_lpass4_TOM.pkl',
        # r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_musc_2406_allsess_fam_2d_90Hz_hpass01_lpass4hanning015_TOM.pkl',
        # r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_musc_2406_allsess_v2_fam_2d_90Hz_hpass01_lpass4_TOM.pkl',
        r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_musc_2406_v2408_fam_2d_90Hz_hpass01_lpass4_TOM.pkl',
        r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_sept23_w_canny_fam_2d_90Hz_hpass01_lpass4_TOM.pkl',
        r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_musc_may23_fam_2d_90Hz_hpass01_lpass4_TOM.pkl',
        r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_fam_jan23_fam_2d_90Hz_hpass01_lpass4_TOM.pkl', ]]

    cohort_pkl_dict = {k:v for k,v in zip(['ephys_2401','musc_2406','musc_sept23','musc_may23','fam_jan23'],pkls)}

    # all_pupil_data = load_pupil_data(Path(ceph_dir/posix_from_win(
    #     r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_ephys_2024_allsess_fam_2d_90Hz_hpass01_lpass4hanning015_TOM.pkl',
    #     # r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_musc_2406_allsess_fam_2d_90Hz_hpass01_lpass4hanning015_TOM.pkl'
    #     # r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_musc_2406_allsess_v2_fam_2d_90Hz_hpass01_lpass4_TOM.pkl'
    #     # r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_sept23_w_canny_fam_2d_90Hz_hpass01_lpass4_TOM.pkl'
    #     # r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_musc_may23_fam_2d_90Hz_hpass01_lpass4_TOM.pkl'
    # )))
    all_pupil_data = load_pupil_data(cohort_pkl_dict[cohort_tag])
    print(all_pupil_data.keys())
    home_dir = Path(config[f'home_dir_{sys_os}'])
    td_path_pattern = 'data/Dammy/<name>/TrialData'

    cohort_config_path = home_dir/'gd_analysis'/'config'/f'{cohort_tag}.yaml'
    # cohort_config_path = home_dir/'gd_analysis'/'config'/'musc_sept23.yaml'
    # cohort_config_path = home_dir/'gd_analysis'/'config'/'musc_may23.yaml'
    with open(cohort_config_path, 'r') as file:
        cohort_config = yaml.safe_load(file)
    drug_sess_dict = {}
    get_drug_dates(cohort_config,session_topology,drug_sess_dict, date_start=None, date_end=240830)  # 240717

    cond_filters = get_all_cond_filts()
    # common_filters = 'Tone_Position==0 & Stage==3 & N_TonesPlayed==4 & Trial_Outcome==1'
    # common_filters = 'Trial_Outcome==1 & Session_Block==0'
    good_trial_filt = 'n_since_last_Trial_Outcome <=5'

    [cond_filters.update({k:' & '.join([v,good_trial_filt])}) for k,v in cond_filters.items()]

    for sessname in tqdm(list(all_pupil_data.keys()), desc='processing sessions'):
        print(sessname)
        if sessname in ['DO85_240625','DO79_240215','DO60_230224'] or 'DO76' in sessname:  # issue with getting correct trial nums in trial dict
            continue
        sessions[sessname] = Session(sessname, ceph_dir)
        name, date = sessname.split('_')[:2]
        date = int(date)
        sess_info = all_sess_info.query('name==@name & date==@date').reset_index().query('sess_order=="main"').iloc[0]

        # get main sess pattern
        main_sess_td_name = Path(sess_info['sound_bin'].replace('_SoundData', '_TrialData')).with_suffix(
            '.csv').name

        td_path = td_path_pattern.replace('<name>', main_sess_td_name.split('_')[0])
        abs_td_path_dir = home_dir / td_path
        abs_td_path = next(abs_td_path_dir.glob(f'{name}_TrialData_{date}*.csv'))
        sessions[sessname].load_trial_data(abs_td_path)
        get_n_since_last(sessions[sessname].td_df, 'Trial_Outcome', 0)
        [get_prate_block_num(sessions[sessname].td_df, prate,rate_name) for prate,rate_name in zip([0.1,0.9],['frequent', 'rare'])]
        get_cumsum_columns(sessions, sessname)
        filter_session(sessions, sessname, [3,4], drug_sess_dict)

    for sessname in tqdm(list(sessions.keys()), desc='processing sessions'):
        print(sessname)
        name, date = sessname.split('_')[:2]
        date = int(date)
        sess_info = all_sess_info.query('name==@name & date==@date').reset_index().query('sess_order=="main"').iloc[0]
        sound_bin_path = Path(sess_info['sound_bin'])
        beh_bin_path = Path(sess_info['beh_bin'])

        # get main sess pattern
        normal = get_main_sess_patterns(td_df=sessions[sessname].td_df)
        [add_datetimecol(sessions[sessname].td_df, col) for col in ['Trial_Start', 'ToneTime', 'Trial_End', 'Gap_Time',
                                                                    'Bonsai_Time']]
        get_earlyX_trials(sessions[sessname].td_df)

        sound_write_path = sound_bin_path.with_stem(f'{sound_bin_path.stem}_write_indices').with_suffix('.csv')
        sound_write_path = ceph_dir/posix_from_win(str(sound_write_path))
        sound_events_path = sound_bin_path.with_stem(f'{sound_bin_path.stem}_event_data_81').with_suffix('.csv')
        sound_events_path = ceph_dir/posix_from_win(str(sound_events_path))
        beh_events_path = beh_bin_path.with_stem(f'{beh_bin_path.stem}_event_data_32').with_suffix('.csv')
        beh_events_path = ceph_dir/posix_from_win(str(beh_events_path))
        beh_events_path_44 = beh_bin_path.with_stem(f'{beh_bin_path.stem}_event_data_44').with_suffix('.csv')
        beh_events_path_44 = ceph_dir/posix_from_win(str(beh_events_path_44))
        beh_writes_path = beh_bin_path.with_stem(f'{beh_bin_path.stem}_write_data').with_suffix('.csv')
        beh_writes_path = ceph_dir/posix_from_win(str(beh_writes_path))

        if not beh_events_path.is_file():
            print(f'ERROR {beh_events_path} not found')
            continue
        sessions[sessname].init_pupil_obj(all_pupil_data[sessname].pupildf,sound_write_path,
                                          beh_events_path,normal)
        if date < 240101:
            try:
                sound_events_df = pd.read_csv(sound_events_path,nrows=1)
                beh_events_44_df = pd.read_csv(beh_events_path_44,nrows=1)
            except FileNotFoundError:
                print(f'ERROR {beh_events_path_44} not found')
                continue
            sync_beh2sound(sessions[sessname].pupil_obj, beh_events_44_df, sound_events_df)

        # [sessions[sessname].get_pupil_to_event(e_idx,e_name,[-1,3], alignmethod='w_td_df',
        #                                        align_kwargs=dict(baseline_dur=1,))  # size_col='canny_raddi_a_zscored'
        #  for e_idx,e_name in tqdm(zip([3,normal[0][0]],['X','A']),total=2,desc='pupil data processing')]

        main_patterns = get_main_sess_patterns(td_df=sessions[sessname].td_df)
        normal_patterns = [pattern for pattern in main_patterns if np.all(np.diff(pattern) > 0) or np.all(np.diff(pattern) < 0)]

        normal = normal_patterns[0]
        # none = idx 8 pip_diff = 0 then only times > 2 secs from t start
        base_idx = normal[0] - 2
        events = {e: {'idx': idx, 'filt': filt} for e, idx, filt in zip(['X', 'A', 'Start', 'none','deviant_A'],
                                                                        [3, normal[0], base_idx, base_idx,normal[0]],
                                                                        ['', 'pip_counter==1', 'Time_diff>1',
                                                                         'Payload_diff==0&rand_n==1&d_X_times>3&d_X_times<6',
                                                                         'ptype==1 & pip_counter==1'
                                                                         ])}
        [sessions[sessname].get_pupil_to_event(e_dict['idx'], e,[-1,4],
                                               align_kwargs=dict(sound_df_query=e_dict['filt'],baseline_dur=1),
                                               alignmethod='w_soundcard')
                                               # alignmethod='w_td_df')
         for e, e_dict in events.items()]


        # X_raster_plot = plot_2d_array_with_subplots(sessions[sessname].pupil_obj.aligned_pupil['X'],plot_cbar=False,
        #                                             cmap='Reds')
        # X_raster_plot[0].show()


        # trial_nums_by_cond = [sessions[sessname].td_df.query(cond_filter).index.values for cond_filter in cond_filters]

        # for event in ['X','A']:

        #
        # A_raster_plot = plot_2d_array_with_subplots(sessions[sessname].pupil_obj.aligned_pupil['A'],plot_cbar=False)
        # A_raster_plot[0].show()
        # # histofram of pupil diff
        # diff_hist_plot = plt.subplots()
        # diff_hist_plot[1].hist(sessions[sessname].pupil_obj.aligned_pupil['X'].diff(axis=1).values.flatten(),bins=100,
        #                        density=True)
        # diff_hist_plot[0].show()
        # e_diff = sessions[sessname].pupil_obj.aligned_pupil['X'].diff(axis=1)
        # A_pdr = e_diff > np.nanmean(e_diff)+np.nanstd(e_diff)*1.96
        # A_raster_pdr_plot = plt.subplots(2)
        # plot_2d_array_with_subplots(A_pdr.values,plot_cbar=False,cmap='Reds',
        #                             plot=(A_raster_pdr_plot[0],A_raster_pdr_plot[1][0]))
        # A_raster_pdr_plot[1][1].plot(A_pdr.mean(axis=0).loc[-0.75:],color='k')
        # # A_raster_pdr_plot[1][1].set_ylim(np.quantile(A_pdr.mean(axis=0),[0.01,1])*[0.95,1.1])
        # A_raster_pdr_plot[0].show()
    # [get_n_since_last(sess.td_df,'Trial_Outcome',1) for sess in sessions.values()]


    # A_by_cond_none = {cond: group_pupil_across_sessions(sessions, [e for e in drug_sess_dict['none'] if e in sessions], 'A', cond, cond_filters, )
    #                   for cond in ['rare', 'frequent','recent','distant','normal','deviant_C']}
    A_by_cond = {cond: group_pupil_across_sessions(sessions,list(sessions.keys()),'A',cond,cond_filters,)
                 for cond in ['rare','frequent','recent','distant','normal','deviant_C' ]}
    A_by_cond['none'] = group_pupil_across_sessions(sessions,list(sessions.keys()),'A','none',cond_filters)
    A_dev_by_cond = {'deviant_C': group_pupil_across_sessions(sessions,list(sessions.keys()),'deviant_A','deviant_C',cond_filters)}
    X_by_cond = {cond: group_pupil_across_sessions(sessions,list(sessions.keys()),'X',cond,cond_filters)
                 for cond in ['rare','frequent','recent','distant','normal','deviant_C' ]}

    assert pd.concat([A_by_cond['rare'],A_by_cond['frequent']]).drop_duplicates().shape[0] == pd.concat([A_by_cond['rare'],A_by_cond['frequent']]).shape[0]
    # pickle session dict
    pkl_dir = ceph_dir/'Dammy'/'pupil_data'
    if not pkl_dir.is_dir():
        pkl_dir.mkdir(parents=True)
    with open(pkl_dir/f'{cohort_tag}_sessions.pkl', 'wb') as f:
        pickle.dump(sessions, f)
    # pick A and X by cond
    pickle.dump(A_by_cond, open(pkl_dir/f'{cohort_tag}_A_by_cond.pkl', 'wb'))
    pickle.dump(X_by_cond, open(pkl_dir/f'{cohort_tag}_X_by_cond.pkl', 'wb'))

    rare_vs_frequent_drug_plots = plt.subplots(2,len(drug_sess_dict),sharey='row',figsize=(6*len(drug_sess_dict),5*2))
    for ei,(event,event_response) in enumerate(zip(['A','X'],[A_by_cond,X_by_cond])):
        for drug_i,drug in enumerate(['none','muscimol','saline']):
            for cond_i, cond_name in enumerate(event_response):
                plot = rare_vs_frequent_drug_plots[1][ei,drug_i]
                if cond_name in ['mid1','mid2']:
                    continue
                cond_pupil: pd.DataFrame = event_response[cond_name]
                if cond_pupil.empty:
                    continue
                sess_idx_bool = cond_pupil.index.get_level_values('sess').isin(drug_sess_dict[drug])
                cond_pupil_control_sess = cond_pupil[sess_idx_bool]
                mean_means_pupil = cond_pupil_control_sess.groupby(level='sess').mean().mean(axis=0)
                plot.plot(mean_means_pupil, label=cond_name)
                cis = sem(cond_pupil_control_sess.groupby(level='sess').mean())
                plot.fill_between(cond_pupil_control_sess.columns.tolist(), mean_means_pupil-cis, mean_means_pupil+cis, alpha=0.1)
                # plot_ts_var(cond_pupil_control_sess.columns,cond_pupil_control_sess.values,f'C{cond_i}',plot)
                plot.set_xlabel(f'time from {event.replace("A","pattern")} onset(s)')
                plot.set_ylabel('pupil size')
                plot.legend()
                plot.axvline(0, color='k',ls='--')
                if event == 'A':
                    [plot.axvspan(t,t+0.15,fc='grey',alpha=0.1) for t in np.arange(0,1,0.25)]

    [plot.set_title(drug) for drug,plot in zip(['none','muscimol','saline'],rare_vs_frequent_drug_plots[1][0])]
    rare_vs_frequent_drug_plots[0].suptitle('Pupil response to rare vs. frequent')
    rare_vs_frequent_drug_plots[0].set_layout_engine('tight')
    # rare_vs_frequent_drug_plots[0].set_size_inches(8*2,3*len(drug_sess_dict))
    rare_vs_frequent_drug_plots[0].show()
    rare_vs_frequent_drug_plots[0].savefig(figdir / f'rare_vs_frequent_pupil_response_all_sessions_{cohort_tag}.svg')

    # plot rare vs freq none drug separate figure
    line_colours = ['darkblue','darkgreen','darkred','darkorange','darkcyan']
    rare_freq_plots = {}
    for ei,(event,event_response) in enumerate(zip(['A','X'],[A_by_cond,X_by_cond])):
        pupil_to_event_plot = plt.subplots()
        for cond_i, cond_name in enumerate(['rare','frequent','distant', 'recent']): # enumerate(event_response):
            if cond_name in ['mid1','mid2']:
                continue
            cond_pupil: pd.DataFrame = event_response[cond_name]
            sess_idx_bool = cond_pupil.index.get_level_values('sess').isin(drug_sess_dict['none'])
            cond_pupil_control_sess = cond_pupil[sess_idx_bool]
            mean_means_pupil = cond_pupil_control_sess.groupby(level='sess').mean().mean(axis=0)
            plot = pupil_to_event_plot[1]
            plot.plot(mean_means_pupil, label=cond_name, color=line_colours[cond_i] if cond_name not in ['none','base'] else 'darkgrey')
            cis = sem(cond_pupil_control_sess.groupby(level='sess').mean())
            plot.fill_between(cond_pupil_control_sess.columns, mean_means_pupil-cis, mean_means_pupil+cis, alpha=0.1,
                              fc=line_colours[cond_i] if cond_name not in ['none','base'] else 'darkgrey', )
            plot.set_xlabel(f'time from {event.replace("A","pattern")} onset(s)')
            plot.set_ylabel('pupil size')
            plot.legend()
            plot.axvline(0, color='k',ls='--')
            if event == 'A':
                [plot.axvspan(t,t+0.15,fc='grey',alpha=0.1) for t in np.arange(0,1,0.25)]

        pupil_to_event_plot[1].set_title('Pupil response to rare vs. frequent')
        pupil_to_event_plot[0].set_layout_engine('tight')
        # pupil_to_event_plot[0].set_size_inches(8,3)
        pupil_to_event_plot[0].show()
        pupil_to_event_plot[0].savefig(figdir / f'rare_vs_frequent_pupil_response_none_drug_{event}_{cohort_tag}.svg')
        rare_freq_plots[event] = copy(pupil_to_event_plot)

    plot_indv_sessions = False
    # plot indv sessions for none drug
    if plot_indv_sessions:
        for drug in drug_sess_dict:
            if not drug_sess_dict[drug]:
                continue
            n_plotcolumns = 6
            rare_vs_freq_by_sess_plot = plt.subplots(ncols=n_plotcolumns,nrows=len(drug_sess_dict[drug])//n_plotcolumns,
                                                     figsize=(6*4, 4*len(drug_sess_dict[drug])//n_plotcolumns))
            for sess,ax in tqdm(zip(drug_sess_dict[drug],rare_vs_freq_by_sess_plot[1].flatten()),
                                 total=len(drug_sess_dict[drug]),desc='plotting sessions'):
                for cond_i, cond_name in enumerate(['rare','frequent']):
                    sess_idx_bool = A_by_cond[cond_name].index.get_level_values('sess')==sess
                    cond_pupil = A_by_cond[cond_name][sess_idx_bool]
                    ax.plot(cond_pupil.mean(axis=0), label=cond_name)
                    plot_ts_var(cond_pupil.columns,cond_pupil.values,f'C{cond_i}',plt_ax=ax)
                    ax.set_title(f'{drug} drug session {sess}')
                    ax.set_xlabel(f'time from pattern onset (s)')
                    ax.set_ylabel('pupil size')
                    ax.axhline(0,color='k',ls='--')
            rare_vs_freq_by_sess_plot[0].set_layout_engine('tight')
            rare_vs_freq_by_sess_plot[0].show()

    rare_freq_diff_plot = plt.subplots(ncols=len(drug_sess_dict),sharey='row', figsize=(12, 5))
    rare_freq_sessions = np.intersect1d(
        *[A_by_cond[cond].index.get_level_values('sess').values for cond in ['rare', 'frequent']])
    # rare_freq_sessions = [sess for sess in rare_freq_sessions if sess in drug_sess_dict['none']]
    for i,sess_type in enumerate(['none','saline','muscimol']):
        for event,event_response,col in zip(['Pattern','X'][:],[A_by_cond,X_by_cond],['g','b']):
            plot_pupil_diff_across_sessions(['distant','recent'],event_response,[sess_type],drug_sess_dict,
                                            plot=[rare_freq_diff_plot[0],rare_freq_diff_plot[1][i]],
                                            dates=rare_freq_sessions,
                                            plt_kwargs=dict(c=col,label=event))

            # rare_freq_diff_plot[1][i].set_xlabel(f'time from pattern onset (s))
            [rare_freq_diff_plot[1][i].axvspan(t, t+0.15,fc='grey',alpha=0.1) for t in np.arange(0,1,0.25)]
            rare_freq_diff_plot[1][i].set_title(f'{sess_type} sessions')
            rare_freq_diff_plot[1][i].axhline(0,color='k',ls='--')
            rare_freq_diff_plot[1][i].axvline(0,color='k',ls='--')


    # rare_freq_diff_plot[0].suptitle(f'Pupil difference between rare and frequent',fontsize=18)
    # [ax.tick_params(axis='both', labelsize=13) for ax in rare_freq_diff_plot[1]]
    [ax.locator_params(axis='both',nbins=4) for ax in rare_freq_diff_plot[1]]
    # rare_freq_diff_plot[1][0].set_ylabel('pupil size (rare - frequent)',fontsize=14)
    rare_freq_diff_plot[1][-1].legend()
    rare_freq_diff_plot[0].set_layout_engine('tight')
    rare_freq_diff_plot[0].show()

    rare_freq_diff_plot[0].savefig(figdir / f'rare_vs_frequent_pupil_diff_across_sessions_{cohort_tag}.svg')

    rare_freq_diff_all_drugs = plt.subplots()
    all_pupil_diff_data = {}
    for i,(sess_type,col) in enumerate(zip(['none','saline','muscimol'],['darkgreen','darkblue','darkred'])):
        all_pupil_diff_data[sess_type] = plot_pupil_diff_across_sessions(['distant','recent'], A_by_cond, [sess_type], drug_sess_dict,
                                                                         plot=rare_freq_diff_all_drugs,
                                                                         plt_kwargs=dict(c=col,label=sess_type),
                                                                         savgol_kwargs=dict(window_length=25, polyorder=5))

    rare_delta_ts_data_pkl = ceph_dir / posix_from_win(r'X:\Dammy\figures\figure_muscimol') / f'{cohort_tag}_rare_freq_delta.pkl'
    with open(rare_delta_ts_data_pkl, 'wb') as pklfile:
        pickle.dump(all_pupil_diff_data, pklfile)

    rare_freq_diff_all_drugs[1].set_title(f'Pupil difference between rare and frequent')
    rare_freq_diff_all_drugs[1].set_xlabel(f'Time from pattern onset (s)')
    rare_freq_diff_all_drugs[1].set_ylabel('pupil size (rare - frequent)')
    # rare_freq_diff_all_drugs[0].set_layout_engine('tight')
    box = rare_freq_diff_all_drugs[1].get_position()
    rare_freq_diff_all_drugs[1].set_position([box.x0, box.y0, box.width * 0.9, box.height])
    rare_freq_diff_all_drugs[0].legend(bbox_to_anchor=(.85, .88))
    rare_freq_diff_all_drugs[1].axhline(0, color='k', ls='--')
    rare_freq_diff_all_drugs[1].locator_params(axis='both', nbins=4)
    rare_freq_diff_all_drugs[1].tick_params(axis='both')
    [rare_freq_diff_all_drugs[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
    rare_freq_diff_all_drugs[0].set_size_inches(8, 6)
    rare_freq_diff_all_drugs[0].show()
    rare_freq_diff_all_drugs[0].savefig(figdir / f'rare_vs_frequent_pupil_diff_across_sessions_all_drugs_{cohort_tag}.svg')

    # decode cond from pupil response to A use session means
    decode_pupil = False
    if decode_pupil:
        A_predictors = []
        A_labels = []
        # t_start = -0.75
        t_start = 1.5
        t_end = 2.5
        x_ser = A_by_cond['rare'].loc[:, t_start:].columns
        for cond_i,cond in enumerate(['normal_exp','dev_ABBA1']):
        # for cond_i,cond in enumerate(['recent','distant']):
            cond_pupil = X_by_cond[cond]
            sess_idx_bool = cond_pupil.index.get_level_values('sess').isin(abstraction_sessions)
            # A_predictors.append(cond_pupil.diff(axis=1).loc[sess_idx_bool, t_start:].values) # time series
            A_predictors.append(np.array([mean(cond_pupil.loc[sess_idx_bool, t_start:t_end],axis=1)
                                          # for mean in [np.mean,np.median,np.max]]).T)
                                          for mean in [np.max]]).T)
            A_labels.append([cond_i]*cond_pupil[sess_idx_bool].shape[0])
        A_vs_X_predictors = []
        A_vs_X_labels = []
        for cond_i, cond_pupil in enumerate([A_by_cond['frequent'], X_by_cond['frequent']]):
            sess_idx_bool = cond_pupil.index.get_level_values('sess').isin(drug_sess_dict['none'])
            # A_vs_X_predictors.append(cond_pupil.loc[sess_idx_bool, t_start:].values) # time series
            A_vs_X_predictors.append(np.array([mean(cond_pupil.loc[sess_idx_bool, t_start:t_end],axis=1)
                                               for mean in [np.mean,np.median,np.max]]).T) # time series
            # A_vs_X_predictors.append(cond_pupil.diff(axis=1).loc[sess_idx_bool, t_start:].values)
            A_vs_X_labels.append([cond_i] * cond_pupil[sess_idx_bool].shape[0])
        decoders_dict = {}
        for xys,lbl in zip([[A_predictors,A_labels],[A_vs_X_predictors,A_vs_X_labels]],['normal vs deviant','Pattern vs X']):
            decoders_dict[lbl] = decode_responses(xys[0], xys[1],dec_kwargs=dict(cv_folds=0, penalty='l1',solver='saga',
                                                                                 n_jobs=os.cpu_count(),),
                                                  n_runs=100)

        for lbl,decoder in decoders_dict.items():
            # plot accuracy

            decoder_acc_plot = plt.subplots()
            decoder_acc_plot[1].boxplot([np.array(dec.accuracy) for dec in decoder.values()],
                                        labels=decoder.keys())
            decoder_acc_plot[1].set_title(f'Accuracy of {lbl}')
            decoder_acc_plot[0].set_layout_engine('tight')
            decoder_acc_plot[0].show()

            # ttest on fold accuracy
            ttest = ttest_ind(*[np.array(dec.fold_accuracy).flatten() for dec in decoder.values()],
                              alternative='greater', equal_var=True)
            print(f'ttest on {lbl} = {ttest} p = {ttest.pvalue}')
            dec_coefs_plot = plt.subplots()
            [dec_coefs_plot[1].plot(
                # x_ser,
                                    np.mean([e.coef_[0] for e in decoder['data'].models[run]], axis=0),
                                    c='gray',alpha=0.25)
             for run in range(len(decoder['data'].models))]
            # dec_coefs_plot[1].plot(x_ser,
            #                         np.mean([[e.coef_[0] for e in rare_vs_freq_decoder['data'].models[run]]], axis=0),
            #                        c='r')
            dec_coefs_plot[1].set_title(f'Coefficients time series of {lbl}')
            dec_coefs_plot[1].set_xticks(np.arange(3))
            dec_coefs_plot[1].set_xticklabels(['mean','median','max'])
            # dec_coefs_plot[1].axvline(0, color='k', ls='--')
            dec_coefs_plot[0].set_layout_engine('tight')

            dec_coefs_plot[0].show()

    # assert False
    # alt vs rand analysis
    [A_by_cond.update({cond: group_pupil_across_sessions(sessions, list(sessions.keys()), 'A', cond, cond_filters, )})
     for cond in ['alternating', 'random']]
    [X_by_cond.update({cond: group_pupil_across_sessions(sessions, list(sessions.keys()), 'X', cond, cond_filters, )})
     for cond in ['alternating', 'random']]
    alt_rand_sessions = np.intersect1d(*[A_by_cond[cond].index.get_level_values('sess').values
                                         for cond in ['normal', 'deviant_C']])
    alt_rand_sessions = [sess for sess in alt_rand_sessions if '240805' in sess]
    # plot alt vs rand pupil response
    alt_rand_pupil_plot = plt.subplots()
    x_ser = A_by_cond['normal'].columns
    lss= ['-', '--']
    # for cond_i, cond in enumerate(['alternating', 'random']):
    for cond_i, cond in enumerate(['normal', 'deviant_C']):
        mean_responses_by_sess = A_by_cond[cond].query('sess in @alt_rand_sessions').groupby(level='sess').mean()
        # [alt_rand_pupil_plot[1].plot(x_ser,sess_response,c=f'C{cond_i}',alpha=0.25)
        #  for _,sess_response in mean_responses_by_sess.iterrows()]
        alt_rand_pupil_plot[1].plot(x_ser, mean_responses_by_sess.mean(axis=0), label=cond)
        # plot sems
        alt_rand_pupil_plot[1].fill_between(x_ser,mean_responses_by_sess.mean(axis=0) - mean_responses_by_sess.sem(axis=0),
                                             mean_responses_by_sess.mean(axis=0) + mean_responses_by_sess.sem(axis=0), alpha=0.1)
    alt_rand_pupil_plot[1].set_title('Pupil response')
    alt_rand_pupil_plot[1].legend()
    alt_rand_pupil_plot[0].set_layout_engine('tight')
    alt_rand_pupil_plot[0].show()

    # get alt vs rand diff
    alt_rand_sessions = np.intersect1d(*[A_by_cond[cond].index.get_level_values('sess').values
                                         for cond in ['normal', 'deviant_C']])
    # alt_rand_sessions = [sess for sess in alt_rand_sessions if '240805' in sess]
    # get alt rand diff
    # responses_by_cond_by_sess = [A_by_cond[cond].groupby(level='sess').mean() for cond in ['rare', 'frequent']]
    none_sessions = list(sessions.keys())
    # responses_by_cond_by_sess = [A_by_cond[cond].query('sess in @alt_rand_sessions').groupby(level='sess').median() for cond in ['alternating', 'random']]
    responses_by_cond_by_sess = [A_by_cond[cond].query('sess in @alt_rand_sessions').groupby(level='sess').median() for cond in ['deviant_C', 'normal']]
    alt_rand_diff = responses_by_cond_by_sess[0].sub(responses_by_cond_by_sess[1],level='sess')
    alt_rand_diff_plot = plt.subplots()
    alt_rand_diff_plot[1].plot(x_ser, alt_rand_diff.mean(axis=0),c='k')
    alt_rand_diff_plot[1].fill_between(x_ser,alt_rand_diff.mean(axis=0) - alt_rand_diff.sem(axis=0),
                                             alt_rand_diff.mean(axis=0) + alt_rand_diff.sem(axis=0), alpha=0.1,
                                       fc='k')
    alt_rand_diff_plot[1].set_title('Pupil response')
    alt_rand_diff_plot[1].set_ylabel('alternating - random')
    [alt_rand_diff_plot[1].axvspan(t,t+0.15,fc='grey',alpha=0.1) for t in np.arange(0,1,0.25)]
    alt_rand_diff_plot[0].set_layout_engine('tight')
    alt_rand_diff_plot[0].show()

    # rolling
    # responses_by_cond_by_sess = [A_by_cond[cond].groupby(level='sess').rolling(10).mean() for cond in ['alternating']]

    norm_dev_figdir = ceph_dir/'Dammy'/'figures'/'jan_23_animals_norm_dev'
    if not norm_dev_figdir.is_dir():
        norm_dev_figdir.mkdir(parents=True)
    # effects over time
    # effects over trial nums
    early_late_dict = {'early':'<=7', 'late': '>= <col>.max()-8'}
    filters_early_late = {}
    [[filters_early_late.update({f'{cond}_{late_early}': f'{cond_filters[cond]} & {cond}_num_cumsum {late_early_filt.replace("<col>", f"{cond}_num_cumsum")}'})
     for late_early,late_early_filt in early_late_dict.items()]
     for cond in ['normal', 'deviant_C','alternating']]

    [A_by_cond.update({cond: group_pupil_across_sessions(sessions, list(sessions.keys()), 'A', cond, td_df_query=query)})
     for cond, query in filters_early_late.items()]

    norm_dev_sessions = np.intersect1d(*[A_by_cond[cond].index.get_level_values('sess').values
                                         for cond in ['normal', 'deviant_C']])
    cond_line_kwargs = {cond:{'c':'blue' if 'normal' in cond else 'red',
                              'ls': '--' if 'late' in cond else '-'}
                        for cond in ['normal_early', 'normal_late', 'deviant_C_early', 'deviant_C_late', 'normal', 'deviant_C'] }

    norm_dev_early_late_plot = plot_pupil_ts_by_cond(A_by_cond, ['normal_early', 'normal_late', 'deviant_C_early', 'deviant_C_late'],
                                                     sess_list=norm_dev_sessions, cond_line_kwargs=cond_line_kwargs)

    norm_dev_plot = plot_pupil_ts_by_cond(A_by_cond, ['normal', 'deviant_C'], sess_list=norm_dev_sessions,
                                          cond_line_kwargs=cond_line_kwargs)
    for plot,savename in zip([norm_dev_early_late_plot, norm_dev_plot],['norm_dev_early_late_plot', 'norm_dev_plot']):
        plot[1].set_title('')
        # plot[1].set_ylim(-0.75,1.25)
        ylim = copy(plot[1].get_ylim())
        plot[1].set_yticks(np.arange(np.round(ylim[0]*2)/2, ylim[1], 1/2))
        plot[1].locator_params(axis='x', nbins=4)
        # plot[1].axvline(0,c='k',ls='--')
        [plot[1].axvspan(t,t+0.15,fc='grey',alpha=0.1) for t in np.arange(0,1,0.25)]
        plot[0].set_layout_engine('tight')
        plot[0].set_size_inches(3,3)
        plot[0].show()
        plot[0].savefig(norm_dev_figdir/f'{savename}.svg')


    norm_dev_diff_plot = plot_pupil_diff_ts_by_cond(A_by_cond, [ 'deviant_C', 'normal'], sess_list=norm_dev_sessions )
    # norm_dev_diff_plot = plot_pupil_diff_ts_by_cond(A_by_cond, ['normal_early', 'normal_late', 'deviant_C_early', 'deviant_C_late'],
    #                                                 sess_list=norm_dev_sessions )
    norm_dev_diff_plot[1].locator_params(axis='both', nbins=4)
    norm_dev_diff_plot[1].set_title('')
    [norm_dev_diff_plot[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
    norm_dev_diff_plot[1].axhline(0, c='k', ls='--')
    norm_dev_diff_plot[0].set_layout_engine('tight')
    norm_dev_diff_plot[0].set_size_inches(3,3)
    norm_dev_diff_plot[0].show()

    norm_dev_diff_plot[0].savefig(norm_dev_figdir/'norm_dev_diff_plot.svg')

    early_late_dict = {'early':'<=10', 'late': '>= <col>.max()-11'}
    filters_early_late = {}
    [[filters_early_late.update({f'{cond}_{late_early}': f'{cond_filters[cond]} & {cond}_num_cumsum {late_early_filt.replace("<col>", f"{cond}_num_cumsum")}'})
     for late_early,late_early_filt in early_late_dict.items()]
     for cond in ['alternating', 'random']]

    [A_by_cond.update({cond: group_pupil_across_sessions(sessions, list(sessions.keys()), 'A', cond, td_df_query=query)})
     for cond, query in filters_early_late.items() if cond in ['alternating_early', 'alternating_late', 'random_early', 'random_late']]

    altenating_sessions =  np.intersect1d(*[A_by_cond[cond].index.get_level_values('sess').values
                                         for cond in ['alternating', 'random']])
    alternating_figdir = norm_dev_figdir.parent/'alternating_new_out_method'
    if not alternating_figdir.is_dir():
        alternating_figdir.mkdir()

    for condition in ['alternating', 'random']:
        alternating_early_late_plot = plot_pupil_ts_by_cond(A_by_cond, [f'{condition}_early', f'{condition}_late'],
                                                            sess_list=altenating_sessions )
        alternating_early_late_plot[1].locator_params(axis='both', nbins=4)
        alternating_early_late_plot[1].set_title('')
        [alternating_early_late_plot[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
        alternating_early_late_plot[1].axhline(0, c='k', ls='--')
        alternating_early_late_plot[0].set_layout_engine('tight')
        # alternating_early_late_plot[0].set_size_inches(3,3)
        alternating_early_late_plot[0].show()
        alternating_early_late_plot[0].savefig(alternating_figdir/f'{condition}_early_late_plot.svg')

        # diff plot
        sess_list=altenating_sessions
        diff_plot = plot_pupil_diff_ts_by_cond(A_by_cond, [f'{condition}_early', f'{condition}_late'], sess_list=altenating_sessions )
        diff_plot[1].locator_params(axis='both', nbins=4)
        diff_plot[1].set_title('')
        [diff_plot[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
        diff_plot[1].axhline(0, c='k', ls='--')
        diff_plot[0].set_layout_engine('tight')
        # diff_plot[0].set_size_inches(3,3)
        diff_plot[0].show()
        diff_plot[0].savefig(alternating_figdir/f'{condition}_diff_plot.svg')

    [[A_by_cond.update(
        {f'{cond}_{block_i}': group_pupil_across_sessions(sessions, list(sessions.keys()), 'A', cond,
                                                          td_df_query=cond_filters[cond] + f' & {cond}_block_num== {block_i}')})  # {"0.9" if cond == "rare" else "0.1"}
        for block_i in range(1,4)]
     for cond in ['rare', 'frequent',]]

    # plot rare vs frequent
    rare_freq_sessions = np.intersect1d(
        *[A_by_cond[cond].index.get_level_values('sess').values for cond in ['rare', 'frequent']])
    rare_freq_figdir = norm_dev_figdir.parent/'rare_freq'
    if not rare_freq_figdir.is_dir():
        rare_freq_figdir.mkdir()
    rare_freq_plot = plot_pupil_ts_by_cond(A_by_cond, ['rare', 'frequent'],sess_list=rare_freq_sessions )
    rare_freq_plot[1].locator_params(axis='both', nbins=4)
    rare_freq_plot[1].set_title('')
    [rare_freq_plot[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
    rare_freq_plot[1].axhline(0, c='k', ls='--')
    rare_freq_plot[0].set_layout_engine('tight')
    rare_freq_plot[0].show()
    rare_freq_plot[0].savefig(rare_freq_figdir/'rare_freq_plot.svg')

    # plot rare by block num
    rare_by_block_figdir = norm_dev_figdir.parent/'rare_by_block'
    if not rare_by_block_figdir.is_dir():
        rare_by_block_figdir.mkdir()
    for rare_freq in ['rare', 'frequent']:
        rare_freq_sessions = np.intersect1d(*[A_by_cond[cond].index.get_level_values('sess').values for cond in [f'{rare_freq}_{i}' for i in [1,3]]])
        rare_by_block_plot = plot_pupil_ts_by_cond(A_by_cond, [f'{rare_freq}_{i}' for i in [1,2,3]], sess_list=rare_freq_sessions)
        rare_by_block_plot[1].locator_params(axis='x', nbins=4)
        ylim = copy(rare_by_block_plot[1].get_ylim())
        # rare_by_block_plot[1].set_yticks(np.arange(np.round(ylim[0] * 2) / 2,1))  # ylim[1], 1 / 2)
        rare_by_block_plot[1].set_yticks(np.arange(0,1.25,0.5))  # ylim[1], 1 / 2)
        rare_by_block_plot[1].set_ylim(-0.5,1)
        rare_by_block_plot[1].set_title('')
        [rare_by_block_plot[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
        rare_by_block_plot[1].axhline(0, c='k', ls='--')
        rare_by_block_plot[0].set_layout_engine('tight')
        # rare_by_block_plot[0].set_size_inches(3,3)
        rare_by_block_plot[0].show()
        rare_by_block_plot[0].savefig(rare_by_block_figdir/f'{rare_freq}_by_block_plot.svg')


    # rare freq diff by drug
    rare_freq_diff_plot = plt.subplots()
    drug_line_kwargs = {'none':{'c': 'darkgreen', 'ls': '-'},'saline': {'c': 'darkblue', 'ls': '-'},'muscimol': {'c': 'darkred', 'ls': '-'}}
    for di,date in enumerate(np.unique([sess.split('_')[1] for sess in drug_sess_dict['saline']])) :
        # line_kwargs = {0:drug_line_kwargs[drug]}
        line_kwargs = None
        plot_pupil_diff_ts_by_cond(A_by_cond, ['distant', 'recent'],
                                   # sess_list=[sess for sess in rare_freq_sessions if sess in drug_sess_dict[drug]],
                                   sess_list=[sess for sess in rare_freq_sessions if date in sess],
                                   plot=rare_freq_diff_plot,
                                   cond_line_kwargs={0:{'c':f'C{di}','label':date}})

    rare_freq_diff_plot[1].locator_params(axis='both', nbins=4)
    rare_freq_diff_plot[1].set_title('')
    rare_freq_diff_plot[1].set_ylabel('')
    rare_freq_diff_plot[1].set_ylim(-.75,.75)
    rare_freq_diff_plot[1].set_xlabel('')
    rare_freq_diff_plot[1].legend(loc='upper left')
    [rare_freq_diff_plot[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
    rare_freq_diff_plot[1].axhline(0, c='k', ls='--')
    rare_freq_diff_plot[0].set_layout_engine('tight')
    rare_freq_diff_plot[0].show()