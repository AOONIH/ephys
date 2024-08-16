import argparse
import pickle
import platform
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as tck
from matplotlib.lines import Line2D
from scipy.stats import ttest_ind
from tqdm import tqdm

from ephys_analysis_funcs import posix_from_win, Session, get_main_sess_patterns

import yaml

from behviour_analysis_funcs import get_sess_name_date_idx, sync_beh2sound, get_all_cond_filts, \
    group_td_df_across_sessions, get_n_since_last, add_datetimecol, group_licks_across_sessions

if __name__ == "__main__":

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
        ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_musc_may23.csv'),
        ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_musc_sept23.csv'),
        ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_musc_2406.csv'),
    ]
    session_topology_dfs = [pd.read_csv(sess_topology_path) for sess_topology_path in session_topology_paths]
    # [df.set_index([cohort_i]*len(df),) for cohort_i, df in enumerate(session_topology_dfs)]
    [pd.concat({cohort_i: df}, names=['cohort']) for cohort_i, df in enumerate(session_topology_dfs)]
    all_sess_info = pd.concat(session_topology_dfs, axis=0).query('sess_order=="main"')

    # matplotlib settings
    plt.style.use('figure_stylesheet.mplstyle')
    line_colours = ['darkblue','darkgreen','darkred','darkorange','darkcyan']
    boxplot_colours = ['#335C67','#9e2a2b','#e09f3e','#540b0e','#fff3b0','#dbd3c9']

    # A4 page variables
    fig_width = 8.3
    # fig_height = 11.7
    fig_height = fig_width

    # init session obj for td_df and sound
    sessions = {}
    for sess_i, sess_info in tqdm(all_sess_info.iterrows(), total=len(all_sess_info), desc='Getting sessions objs'):
        sessname = Path(sess_info['sound_bin']).stem
        sessname = sessname.replace('_SoundData','')
        sessions[sessname] = Session(sessname, ceph_dir)
        date = sess_info['date']

        abs_td_path = Path(sess_info['tdata_file'])
        abs_td_path = home_dir/Path(*abs_td_path.parts[-5:])
        sessions[sessname].load_trial_data(abs_td_path)
        if sessions[sessname].td_df.shape[0] < 100:
            print(f'{sessname} has too few trials: {sessions[sessname].td_df.shape[0]}')
            sessions.pop(sessname)

    # format td_df
    sessnames = list(sessions.keys())
    [get_n_since_last(sess.td_df, 'Trial_Outcome', 0) for sess in sessions.values()]
    all_sess_td_df = group_td_df_across_sessions(sessions, sessnames)
    [add_datetimecol(all_sess_td_df, col) for col in ['Trial_Start', 'ToneTime', 'Trial_End', 'Gap_Time']]


    #  figure 1: performance
    fig1_dir = ceph_dir / 'Dammy' / 'figures' / 'figure1_performance'
    fig1_pkl_dir = ceph_dir / 'Dammy' / 'figures' / 'figure1_performance' / 'pickles'
    if not fig1_dir.is_dir():
        fig1_dir.mkdir()
    if not fig1_pkl_dir.is_dir():
        fig1_pkl_dir.mkdir()

    # plot
    #
    # h_ratio = 0.4
    # w_ratio = 0.5

    h_ratio = 0.33
    w_ratio = 0.33

    all_sess_performance_plot = plt.subplots()
    good_trial_filt = 'Stage>=3 & n_since_last_Trial_Outcome <=5'
    td_df_by_tone_position = all_sess_td_df.query(good_trial_filt).groupby('Tone_Position')
    # all_sess_performance_plot[1].boxplot([df[1].groupby('sess')['Trial_Outcome'].mean()
    #                                       for df in td_df_by_tone_position],
    #                                      labels=td_df_by_tone_position.groups.keys())
    bxplot = all_sess_performance_plot[1].boxplot(
        all_sess_td_df.query(good_trial_filt).groupby(['sess', 'Tone_Position'])[
            'Trial_Outcome'].mean().unstack()[[1, 0]],
        labels=['Pattern', 'Non pattern'])
    for patch, color in zip(bxplot['boxes'], boxplot_colours):
        patch.set_facecolor(color)
    all_sess_performance_plot[1].set_title('pattern vs non pattern trials')
    all_sess_performance_plot[1].set_ylabel('Performance')

    all_sess_performance_plot[1].locator_params(axis='y', nbins=3)
    all_sess_performance_plot[0].set_layout_engine('tight')

    all_sess_performance_plot[0].show()

    all_sess_performance_plot[0].set_size_inches(fig_width * w_ratio, fig_height * h_ratio)
    all_sess_performance_plot[0].savefig(fig1_dir / 'performance_by_tone_position.svg')

    # save plot data
    with open(fig1_pkl_dir / 'performance_by_tone_position_plot.pkl', 'wb') as f:
        pickle.dump(all_sess_performance_plot, f)
    all_sess_td_df.query(good_trial_filt).groupby(['sess', 'Tone_Position'])[
        'Trial_Outcome'].mean().unstack()[[1, 0]].to_pickle(fig1_pkl_dir / 'performance_by_tone_position_data.pkl')

    # ttest on performance by tone position
    ttest = ttest_ind(*[df[1].groupby('sess')['Trial_Outcome'].mean() for df in td_df_by_tone_position],
                      alternative='two-sided')
    print(ttest)

    # performance by tone onset time
    # add datetime column for Trial_Start and Tone_Time
    all_pattern_trials_df = all_sess_td_df.query(
        'Stage>=3 & Tone_Position==0 & n_since_last_Trial_Outcome<=5').copy()
    time_to_metrics = {k:v for k,v in zip(['time_to_tone', 'time_to_gap','start_to_gap'],
                                          [['Trial_Start_dt', 'ToneTime_dt'],['ToneTime_dt', 'Gap_Time_dt'],['Trial_Start_dt', 'Gap_Time_dt']])}

    for k,v in time_to_metrics.items():
        all_pattern_trials_df.loc[:, k] = all_pattern_trials_df[v[1]] - all_pattern_trials_df[v[0]]
    for metric,title,x_label in zip(time_to_metrics,['pattern onset','pattern to X time','start to X time'],
                                    ['Time to pattern (s)','Time to X (s)','Time to X (s)']):
        print(title)
        all_pattern_trials_df.loc[:, metric] = all_pattern_trials_df[metric].dt.total_seconds()
        all_pattern_trials_df.loc[:, f'{metric}_rounded'] = np.round(all_pattern_trials_df[metric])
        pattern_trials_by_time_to = all_pattern_trials_df.groupby(['sess', f'{metric}_rounded'])[
            'Trial_Outcome'].mean().unstack()
        times2use = all_pattern_trials_df[f'{metric}_rounded'].value_counts().where(
            lambda x: x > 0.03*len(all_pattern_trials_df)).dropna().index.sort_values().values
        # times2use = sorted(times2use)
        # times2use = np.arange(2, 7, 1)
        pattern_trials_by_time_to = {t: pattern_trials_by_time_to[t].dropna() for t in times2use}
        patt_perf_by_time_to_plot = plt.subplots()
        bxplot = patt_perf_by_time_to_plot[1].boxplot(pattern_trials_by_time_to.values(),
                                                           labels=times2use.astype(int))
        for patch, color in zip(bxplot['boxes'], boxplot_colours):
            patch.set_facecolor(color)
        # x_label = ' '.join(metric.split('_')).replace('time','Time').replace('gap','X').replace('tone','pattern')
        patt_perf_by_time_to_plot[1].set_title(title)
        patt_perf_by_time_to_plot[1].set_ylabel('Performance')
        patt_perf_by_time_to_plot[1].set_xlabel(x_label)
        patt_perf_by_time_to_plot[1].locator_params(axis='y', nbins=4)
        patt_perf_by_time_to_plot[0].set_layout_engine('tight')
        # patt_perf_by_time_to_plot[0].show()
        patt_perf_by_time_to_plot[0].set_size_inches(fig_width * w_ratio, fig_height * h_ratio)
        patt_perf_by_time_to_plot[0].savefig(fig1_dir / f'{"_".join(title.split(" "))}.svg')

        # save plot data
        with open(fig1_pkl_dir / f'{"_".join(title.split(" "))}_plot.pkl', 'wb') as f:
            pickle.dump(patt_perf_by_time_to_plot, f)
        with open(fig1_pkl_dir / f'{"_".join(title.split(" "))}_data.pkl', 'wb') as f:
            pickle.dump(pattern_trials_by_time_to, f)

    all_correct_trials_df = all_sess_td_df.query('Stage>=3 & Trial_Outcome==1').copy()
    all_correct_trials_df.loc[:, 'reaction_time'] = all_correct_trials_df['Trial_End_dt'] - \
                                                    all_correct_trials_df['Gap_Time_dt']
    all_correct_trials_df.loc[:, 'reaction_time'] = all_correct_trials_df['reaction_time'].dt.total_seconds()

    for grouping,lbls,idxs ,title in zip(['Tone_Position', 'N_TonesPlayed'],[['Pattern','Non pattern'],['AB','ABC','ABCD']],
                                  [[1,0],[2,3,4]],['Pattern vs non pattern trials',
                                                   'Number of tones played']):
        all_correct_trials_df_plot = plt.subplots()
        means_by_group = all_correct_trials_df.groupby(['sess', grouping])['reaction_time'].mean().unstack()[idxs]
        # remove nans
        means_by_group = means_by_group.dropna()
        bxplot = all_correct_trials_df_plot[1].boxplot(means_by_group,labels=lbls)
        for patch, color in zip(bxplot['boxes'], boxplot_colours):
            patch.set_facecolor(color)
        all_correct_trials_df_plot[1].set_ylabel('Reaction Time (s)')
        all_correct_trials_df_plot[1].set_ylim((-0.05, 1))
        all_correct_trials_df_plot[1].set_title(title)
        all_correct_trials_df_plot[1].locator_params(axis='y', nbins=4)
        all_correct_trials_df_plot[0].set_layout_engine('tight')
        all_correct_trials_df_plot[0].show()
        all_correct_trials_df_plot[0].set_size_inches(fig_width * w_ratio, fig_height * h_ratio)
        all_correct_trials_df_plot[0].savefig(fig1_dir / f'reaction_time_by_{grouping.lower()}.svg')

        # save plot data
        with open(fig1_pkl_dir / f'reaction_time_by_{grouping.lower()}_plot.pkl', 'wb') as f:
            pickle.dump(all_correct_trials_df_plot, f)
        with open(fig1_pkl_dir / f'reaction_time_by_{grouping.lower()}_data.pkl', 'wb') as f:
            pickle.dump(means_by_group, f)

    # figure 2:lick analysis
    # for sess_i, sess_info in tqdm(all_sess_info.iterrows(), total=len(all_sess_info), desc='Getting sessions objs'):
    get_sess_objs = False
    sess_pkl_path = fig1_pkl_dir / 'sessions.pickle'
    if get_sess_objs:
        for sessname in tqdm(sessions, total=len(sessions), desc='Getting sessions lick objs'):
            if sessions[sessname].lick_obj is not None:
                continue
            name, date, sess_idx, sess_info = get_sess_name_date_idx(sessname, all_sess_info)

            if sessname not in sessions:
                continue
            if sessions[sessname].td_df.shape[0] < 100:
                print(f'{sessname} has too few trials: {sessions[sessname].td_df.shape[0]}')
                sessions.pop(sessname)
                continue
            if 3 not in sessions[sessname].td_df['Stage'].values or sessions[sessname].td_df.query('Tone_Position==0').empty:
                continue
            sound_bin_path = Path(sess_info['sound_bin'])
            beh_bin_path = Path(sess_info['beh_bin'])

            sound_write_path = sound_bin_path.with_stem(f'{sound_bin_path.stem}_write_indices').with_suffix('.csv')
            sound_write_path = ceph_dir / posix_from_win(str(sound_write_path))
            sound_events_path = sound_bin_path.with_stem(f'{sound_bin_path.stem}_event_data_81').with_suffix('.csv')
            sound_events_path = ceph_dir / posix_from_win(str(sound_events_path))
            beh_events_path = beh_bin_path.with_stem(f'{beh_bin_path.stem}_event_data_32').with_suffix('.csv')
            beh_events_path = ceph_dir / posix_from_win(str(beh_events_path))
            beh_events_path_44 = beh_bin_path.with_stem(f'{beh_bin_path.stem}_event_data_44').with_suffix('.csv')
            beh_events_path_44 = ceph_dir / posix_from_win(str(beh_events_path_44))
            beh_writes_path = beh_bin_path.with_stem(f'{beh_bin_path.stem}_write_data').with_suffix('.csv')
            beh_writes_path = ceph_dir / posix_from_win(str(beh_writes_path))

            if not beh_events_path.is_file():
                print(f'ERROR {beh_events_path} not found')
                continue
            main_patterns = get_main_sess_patterns(td_df=sessions[sessname].td_df)
            normal_patterns = [pattern for pattern in main_patterns if np.all(np.diff(pattern) > 0)]
            non_normal_patterns = [pattern for pattern in main_patterns if not np.all(np.diff(pattern) > 0)]
            if non_normal_patterns:
                main_patterns = list(sum(zip(normal_patterns, non_normal_patterns), ()))
            else:
                main_patterns = normal_patterns
            if not main_patterns:
                print(f'no main patterns for {sessname}')
                continue
            sessions[sessname].init_lick_obj(beh_events_path, sound_write_path,normal_patterns)
            if date < 240101:
                sound_events_df = pd.read_csv(sound_events_path,nrows=1)
                beh_events_44_df = pd.read_csv(beh_events_path_44,nrows=1)
                sync_beh2sound(sessions[sessname].lick_obj, beh_events_44_df,sound_events_df)

            sessions[sessname].init_sound_event_dict(sound_write_path,patterns=main_patterns)

            # lick analysis
            normal = normal_patterns[0]
            # none = idx 8 pip_diff = 0 then only times > 2 sexs from t start
            base_idx = normal[0] - 2
            events = {e: {'idx':idx, 'filt':filt} for e, idx, filt in zip(['X','A','Start','none'][:2],
                                                                          [3,normal[0],base_idx,base_idx],
                                                                          ['','','Time_diff>1','Payload_diff==0&rand_n==2'])}
            [sessions[sessname].get_licks_to_event(e_dict['idx'], e, align_kwargs=dict(sound_df_query=e_dict['filt'],
                                                                                       kernel_width=100))
             for e, e_dict in events.items()]
    else:
        if sess_pkl_path.is_file():
            with open(sess_pkl_path, 'rb') as f:
                sessions = pickle.load(f)
        else:
            sessions = {}
            warnings.warn('Sessions pickle not found and not getting sessions.')

    # get all session licks into a single df
    # pickle sessions dict
    # with open(fig1_pkl_dir / 'sessions.pickle', 'wb') as f:
    #     pickle.dump(sessions, f)
    sessnames = list(sessions.keys())
    all_sess_licks_df_by_cond = {k: {cond: group_licks_across_sessions(sessions, sessnames, k, cond,
                                                                       get_all_cond_filts())
                                     for cond in ['rare','frequent']}
                                 for k in ['X','A','Start']}

    all_lick_pkl_path = fig1_pkl_dir / 'all_sess_licks_df.pkl'
    if all_lick_pkl_path.is_file():
        all_sess_licks_df = pd.read_pickle(all_lick_pkl_path)
    else:
        all_sess_licks_df = pd.concat({event: pd.concat({cond: cond_df for cond, cond_df in event_dfs.items()},axis=0,
                                                        names=['cond'])
                                       for event, event_dfs in all_sess_licks_df_by_cond.items()},
                                      axis=0,names=['event'])

        with open(fig1_pkl_dir / 'all_sess_licks_df.pkl', 'wb') as f:
            pickle.dump(all_sess_licks_df, f)

    patt_non_patt_pkl_path = fig1_pkl_dir / 'patt_non_patt_licks_df.pkl'
    if patt_non_patt_pkl_path.is_file():
        patt_non_patt_licks_df = pd.read_pickle(patt_non_patt_pkl_path)
    else:
        patt_non_patt_licks_df = (pd.concat(
            {'non_patt': group_licks_across_sessions(sessions, sessnames, 'none', 'none', get_all_cond_filts()),
             'patt': group_licks_across_sessions(sessions, sessnames, 'A', 'pattern', get_all_cond_filts())},
            axis=0, names=['trial_type']))
        with open(fig1_pkl_dir / 'patt_non_patt_licks_df.pkl', 'wb') as f:
            pickle.dump(patt_non_patt_licks_df, f)

    patt_non_patt2X_pkl_path = fig1_pkl_dir / 'patt_non_patt2X_licks_df.pkl'
    if patt_non_patt2X_pkl_path.is_file():
        patt_non_patt2X_licks_df = pd.read_pickle(patt_non_patt2X_pkl_path)
    else:
        patt_non_patt2X_licks_df = (pd.concat(
            {'non_patt': group_licks_across_sessions(sessions, sessnames, 'X', 'none', get_all_cond_filts()),
             'patt': group_licks_across_sessions(sessions, sessnames, 'X', 'pattern', get_all_cond_filts())},
            axis=0, names=['trial_type']))
        with open(fig1_pkl_dir / 'patt_non_patt2X_licks_df.pkl', 'wb') as f:
            pickle.dump(patt_non_patt2X_licks_df, f)

    # plot
    patt_nonpatt_cols = ['dimgray','indigo']
    x_ser = patt_non_patt_licks_df.columns.total_seconds()
    y_lim = patt_non_patt2X_licks_df.mean(level='trial_type').max().max()*1.1 # .quantile(0.99) # .quantile(0.98)
    y_lim = np.round(y_lim,1)
    x_lim = (-1,1.5)
    # patt vs non-patt 2 patt
    patt_non_patt_plot = plt.subplots()
    patt_non_patt_licks_by_type = patt_non_patt_licks_df.groupby(['trial_type','sess']).mean()
    [patt_non_patt_plot[1].plot(x_ser, patt_non_patt_licks_by_type.loc[trial_type].mean(axis=0),
                                label=lbls,c=patt_nonpatt_cols[type_i])
     for type_i,(trial_type,lbls) in enumerate(zip(['non_patt','patt'],['non pattern','pattern']))]
    # fill between sem
    mean_by_trial_type = patt_non_patt_licks_df.groupby(['trial_type','sess']).mean()
    sem_by_trial_type = patt_non_patt_licks_df.groupby(['trial_type','sess']).sem()
    [patt_non_patt_plot[1].fill_between(x_ser, mean_by_trial_type.loc[trial_type].mean(axis=0) - sem_by_trial_type.loc[trial_type].mean(axis=0),
                                        mean_by_trial_type.loc[trial_type].mean(axis=0) + sem_by_trial_type.loc[trial_type].mean(axis=0), color=patt_nonpatt_cols[type_i], alpha=0.1)
     for type_i,(trial_type,lbls) in enumerate(zip(['non_patt','patt'],['non pattern','pattern']))]
    # patt_non_patt_plot[1].plot(x_ser, patt_non_patt_licks_by_type.mean(, label=['non-patt','patt'])
    patt_non_patt_plot[1].set_title('licks to event onset ')
    patt_non_patt_plot[1].legend()
    patt_non_patt_plot[1].locator_params(axis='both', nbins=4)
    patt_non_patt_plot[1].set_ylabel('lick rate')
    patt_non_patt_plot[1].set_ylim(0,y_lim)
    patt_non_patt_plot[1].set_xlim(*x_lim)
    patt_non_patt_plot[1].xaxis.set_major_locator(tck.MultipleLocator())
    patt_non_patt_plot[1].axvline(0, ls='--', c='k')
    patt_non_patt_plot[1].set_xlabel('time from event onset (s)')
    patt_non_patt_plot[0].set_layout_engine('tight')
    patt_non_patt_plot[0].show()
    patt_non_patt_plot[0].set_size_inches(fig_width * w_ratio, fig_height * h_ratio)
    patt_non_patt_plot[0].savefig(fig1_dir / 'licks2A_by_trial_type.svg', format='svg')

    # save plot data
    with open(fig1_pkl_dir / 'licks2A_by_trial_type_plot.pkl', 'wb') as f:
        pickle.dump(patt_non_patt_plot, f)
    with open(fig1_pkl_dir / 'licks2A_by_trial_type_data.pkl', 'wb') as f:
        pickle.dump(patt_non_patt_licks_df.groupby(['trial_type','sess']), f)

    # patt non patt 2x
    patt_non_patt2X_plot = plt.subplots()
    patt_non_patt2X_licks_by_type = patt_non_patt2X_licks_df.groupby(['trial_type','sess']).mean()
    [patt_non_patt2X_plot[1].plot(x_ser, patt_non_patt2X_licks_by_type.loc[trial_type].mean(axis=0),
                                  label=lbl,c=patt_nonpatt_cols[type_i])
     for type_i,(trial_type,lbl) in enumerate(zip(['non_patt','patt'],['non pattern','pattern']))]
    # fill between sem
    mean_by_trial_type = patt_non_patt2X_licks_df.groupby(['trial_type','sess']).mean()
    sem_by_trial_type = patt_non_patt2X_licks_df.groupby(['trial_type','sess']).sem()
    [patt_non_patt2X_plot[1].fill_between(x_ser, mean_by_trial_type.loc[trial_type].mean(axis=0) - sem_by_trial_type.loc[trial_type].mean(axis=0),
                                          mean_by_trial_type.loc[trial_type].mean(axis=0) + sem_by_trial_type.loc[trial_type].mean(axis=0), color=patt_nonpatt_cols[type_i], alpha=0.1)
     for type_i,(trial_type,lbl) in enumerate(zip(['non_patt','patt'],['non pattern','pattern']))]
    # patt_non_patt_plot[1].plot(x_ser, patt_non_patt_licks_by_type.mean(, label=['non-patt','patt'])
    patt_non_patt2X_plot[1].set_title('licks to X onset')
    # patt_non_patt2X_plot[1].legend()
    patt_non_patt2X_plot[1].locator_params(axis='both', nbins=4)
    patt_non_patt2X_plot[1].set_ylabel('lick rate')
    patt_non_patt2X_plot[1].set_ylim(0,y_lim)
    patt_non_patt2X_plot[1].set_xlim(*x_lim)
    patt_non_patt2X_plot[1].xaxis.set_major_locator(tck.MultipleLocator())
    patt_non_patt2X_plot[1].set_xlabel('time from X onset (s)')
    patt_non_patt2X_plot[1].axvline(0, ls='--', c='k')
    patt_non_patt2X_plot[0].set_layout_engine('tight')
    patt_non_patt2X_plot[0].show()
    patt_non_patt2X_plot[0].set_size_inches(fig_width * w_ratio, fig_height * h_ratio)
    patt_non_patt2X_plot[0].savefig(fig1_dir / 'licks_by_trial_type_2X.svg', format='svg')

    # save plot data
    with open(fig1_pkl_dir / 'licks_by_trial_type_2X_plot.pkl', 'wb') as f:
        pickle.dump(patt_non_patt2X_plot, f)
    with open(fig1_pkl_dir / 'licks_by_trial_type_2X_data.pkl', 'wb') as f:
        pickle.dump(patt_non_patt2X_licks_df.groupby(['trial_type','sess']), f)

    for ei, event in enumerate(all_sess_licks_df.index.levels[0]):
    # for event in ['X']:
        all_licks2event_by_cond = all_sess_licks_df.loc[event].groupby(['sess','cond']).mean()
        conds = all_licks2event_by_cond.index.get_level_values('cond').unique().tolist()
        # plot mean lick across grouped by condition
        lick_plot = plt.subplots()
        [lick_plot[1].plot(x_ser,all_licks2event_by_cond.xs(cond,level='cond').mean(axis=0),label=cond,
                           c=line_colours[cond_i])
         for cond_i, cond in enumerate(['rare','frequent'])]
        # fill between sem
        mean_by_cond = [all_licks2event_by_cond.xs(cond,level='cond').mean(axis=0) for cond in conds]
        sem_by_cond = [all_licks2event_by_cond.xs(cond,level='cond').sem(axis=0) for cond in conds]
        [lick_plot[1].fill_between(x_ser, mean_by_cond[cond_i] - sem_by_cond[cond_i],
                                    mean_by_cond[cond_i] + sem_by_cond[cond_i], color=line_colours[cond_i], alpha=0.1)
         for cond_i, cond in enumerate(conds)]

        lick_plot[1].set_title(f'lick rate to {event} onset')
        if ei == len(all_sess_licks_df.index.levels[0]) - 1:
            lick_plot[1].legend(loc='upper right')
        # lick_plot[1].legend(handler_map={Line2D:MyHandlerLine2D()})
        lick_plot[1].locator_params(axis='both', nbins=4)
        lick_plot[1].set_ylabel('lick rate')
        lick_plot[1].set_ylim(0,y_lim)
        lick_plot[1].set_xlim(*x_lim)
        lick_plot[1].xaxis.set_major_locator(tck.MultipleLocator())
        lick_plot[1].set_xlabel('time from event (s)')
        lick_plot[1].axvline(0, c='k', ls='--')
        lick_plot[0].set_layout_engine('tight')
        lick_plot[0].show()
        lick_plot[0].set_size_inches(fig_width * w_ratio, fig_height * h_ratio)
        lick_plot[0].savefig(fig1_dir / f'licks2{event}_{"_".join(conds)}.svg', format='svg')

        # save plot data
        with open(fig1_pkl_dir / f'licks2{event}_{"_".join(conds)}_plot.pkl', 'wb') as f:
            pickle.dump(lick_plot, f)
        with open(fig1_pkl_dir / f'licks2{event}_{"_".join(conds)}_data.pkl', 'wb') as f:
            pickle.dump([all_licks2event_by_cond.xs(cond,level='cond') for cond in conds], f)
