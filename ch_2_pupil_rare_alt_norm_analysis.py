import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from behviour_analysis_funcs import get_lick_in_patt_trials, group_td_df_across_sessions
from scipy.stats import ttest_1samp
from pupil_ephys_funcs import *

def configure_plot(ax, title='', xlabel='', ylabel='', nbins=4, hline=0, vspans=None):
    """
    Configures a matplotlib axis with common settings.

    Parameters:
        ax (matplotlib.axes.Axes): The axis to configure.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        nbins (int): Number of bins for axis ticks.
        hline (float): Horizontal line value.
        vspans (list of tuples): List of (start, end) tuples for vertical spans.
    """
    ax.locator_params(axis='both', nbins=nbins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(hline, c='k', ls='--')
    if vspans:
        for t_start, t_end in vspans:
            ax.axvspan(t_start, t_end, fc='grey', alpha=0.1)

def plot_pupil_diff(A_by_cond, conditions, sess_list, plot):
    """
    Plots pupil differences by condition.

    Parameters:
        A_by_cond (dict): Data grouped by condition.
        conditions (list): List of condition names.
        sess_list (list): List of sessions.
        plot (list): List of matplotlib figure and axes.
    """
    plot_pupil_diff_ts_by_cond(A_by_cond, conditions, sess_list=sess_list, plot=plot)
    configure_plot(plot[1], nbins=4, vspans=[(t, t + 0.15) for t in np.arange(0, 1, 0.25)])
    plot[0].set_layout_engine('tight')
    plot[0].show()

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
    td_path_pattern = 'data/Dammy/<name>/TrialData'

    plt.style.use('figure_stylesheet.mplstyle')


    session_topology_paths  = [
        ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_ephys_2401.csv'),
        ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_musc_2406.csv'),
    ]
    pupil_pkl_paths = [ceph_dir / posix_from_win(p) for p in [
        # r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_ephys_2024_allsess_v3_fam_2d_90Hz_hpass01_lpass4_TOM.pkl',
        r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_ephys_2024_allsess_v2409_walt_fam_2d_90Hz_hpass01_lpass4_TOM.pkl',
        # r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_ephys_2024_allsess_v3_fam_2d_90Hz_hpass01_lpass4_TOM.pkl',
        # r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_musc_2406_allsess_v2_fam_2d_90Hz_hpass01_lpass4_TOM.pkl',
        r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_musc_2406_v2409_walt_fam_2d_90Hz_hpass01_lpass4_TOM.pkl',
        # r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_musc_2406_v2408_fam_2d_90Hz_hpass01_lpass4_TOM.pkl',
    ]]

    all_pupil_data = [load_pupil_data(pkl_path) for pkl_path in pupil_pkl_paths]
    pupil_data_dict = all_pupil_data[0]
    pupil_data_dict = {**pupil_data_dict, **all_pupil_data[1]}

    # load config and get all drug dates
    cohort_tags = ["_".join(sess_topology_path.stem.split(('_'))[-2:]) for sess_topology_path in session_topology_paths]
    config_paths = [home_dir/'gd_analysis'/'config'/f'{cohort_tag}.yaml' for cohort_tag in cohort_tags]
    cohort_configs = []
    for config_path in config_paths:
        with open(config_path, 'r') as file:
            cohort_configs.append(yaml.safe_load(file))

    session_topology_dfs = [pd.read_csv(sess_topology_path) for sess_topology_path in session_topology_paths]
    [pd.concat({cohort_i: df}, names=['cohort']) for cohort_i, df in enumerate(session_topology_dfs)]
    all_sess_info = pd.concat(session_topology_dfs, axis=0).query('sess_order=="main"')

    drug_sess_dict = {}
    [get_drug_dates(cohort_config, session_topology, drug_sess_dict, date_end=None)
     for cohort_config, session_topology in zip(cohort_configs, session_topology_dfs)]
    cond_filters = get_all_cond_filts()


    sess_pkl_path = ceph_dir / posix_from_win(r'X:\Dammy\pupil_data') / 'ephys_2401_musc_2401_cohort_sess_dicts.pkl'


    existing_sessions = list(sessions.keys())

    window = [-1, 4]
    for sessname in tqdm(list(pupil_data_dict.keys()), desc='processing sessions'):
        print(sessname)
        if sessname in ['DO85_240625','DO79_240215'] or 'DO76' in sessname:  # issue with getting correct trial nums in trial dict
            continue
        if sessname in existing_sessions or args.ow:
            continue


        init_pupil_td_obj(sessions, sessname, ceph_dir, all_sess_info, td_path_pattern, home_dir)
        process_pupil_td_data(sessions,sessname,drug_sess_dict)


    for sessname in tqdm(list(sessions.keys()), desc='processing sessions'):
        if sessions[sessname].pupil_obj is not None or args.ow:
            continue
        print(sessname)

        init_sess_pupil_obj(sessions, sessname, ceph_dir, all_sess_info, pupil_data_dict)
        process_pupil_obj(sessions,sessname)
    for sessname in tqdm(list(sessions.keys()), desc='lick data processing', total=len(list(sessions.keys()))):
        if sessions[sessname].lick_obj is None:
            name, date = sessname.split('_')
            date = int(date)
            sess_info = all_sess_info.query('name==@name & date==@date').reset_index().query('sess_order=="main"').iloc[
                0]
            sound_bin_path = Path(sess_info['sound_bin'])
            beh_bin_path = Path(sess_info['beh_bin'])
            sound_write_path = sound_bin_path.with_stem(f'{sound_bin_path.stem}_write_indices').with_suffix('.csv')
            sound_write_path = ceph_dir / posix_from_win(str(sound_write_path))
            beh_events_path = beh_bin_path.with_stem(f'{beh_bin_path.stem}_event_data_32').with_suffix('.csv')
            beh_events_path = ceph_dir / posix_from_win(str(beh_events_path))
            if not beh_events_path.is_file():
                print(f'ERROR {beh_events_path} not found skipping for lick')
                continue

            main_patterns = get_main_sess_patterns(td_df=sessions[sessname].td_df)
            if sessions[sessname].td_df['Stage'].iloc[0] in [3, 5]:
                normal_patterns = main_patterns
                if len(normal_patterns) > 1 and sessions[sessname].td_df['Stage'].iloc[0] == 3:
                    warnings.warn(f'{sessname} has more than one normal pattern for stage 3')
            elif sessions[sessname].td_df['Stage'].iloc[0] == 4:
                normal_patterns = get_main_sess_patterns(
                    td_df=sessions[sessname].td_df.query(cond_filters['normal_exp']))
            else:
                continue
            sessions[sessname].init_lick_obj(beh_events_path, sound_write_path, normal_patterns)
        if 'none_X_licks' in sessions[sessname].lick_obj.event_licks:
            # continue
            pass
        X_times = sessions[sessname].lick_obj.sound_writes.query('Payload==3')['Timestamp'].values
        lick_times = sessions[sessname].lick_obj.spike_times
        lick_dist_to_X = np.array((np.mat(lick_times).T - X_times))
        non_X_licks = [t for t, dist_t in zip(lick_times, lick_dist_to_X)
                       if not any(np.logical_and(dist_t>-window[1],dist_t<-window[0]))]
        non_X_licks_dist = np.array(np.mat(non_X_licks).T - non_X_licks)
        non_X_licks_pure = [t for t, dist_t in zip(non_X_licks, non_X_licks_dist[~np.eye(non_X_licks_dist.shape[0],
                                                                                         dtype=bool)].reshape(non_X_licks_dist.shape[0],-1))
                            if not any(np.logical_and(dist_t>-window[1],dist_t<-window[0]))]
        sessions[sessname].pupil_obj.align2times(non_X_licks_pure, 'dlc_radii_a_zscored',window=window,sessname=sessname,
                                                 event_name='none_X_licks',baseline_dur=1)

    # save sessions
    # if list(sessions.keys()) != existing_sessions or args.ow:
    #     print('saving sessions')
    #     with open (sess_pkl_path, 'wb') as f:
            # pickle.dump(sessions, f)

    cond_filters = get_all_cond_filts()
    # good_trial_filt = 'n_since_last_Trial_Outcome <=5 & lick_in_patt==0'
    good_trial_filt = 'n_since_last_Trial_Outcome <=5 '

    [cond_filters.update({k: ' & '.join([v, good_trial_filt])}) for k, v in cond_filters.items()]

    A_by_cond = {cond: group_pupil_across_sessions(sessions,list(sessions.keys()),'A',cond,cond_filters,)
                 for cond in ['rare','frequent','recent','distant','normal','deviant_C','normal_exp',
                              'normal_exp_midlate','no_early',]}
    # A_by_cond['none'] = group_pupil_across_sessions(sessions,list(sessions.keys()),'none','none',cond_filters)
    # A_dev_by_cond = {'deviant_C': group_pupil_across_sessions(sessions,list(sessions.keys()),'deviant_A','deviant_C',cond_filters)}
    X_by_cond = {cond: group_pupil_across_sessions(sessions,list(sessions.keys()),'X',cond,cond_filters)
                 for cond in ['rare','frequent','recent','distant','normal','deviant_C','normal_exp' ]}

    A_by_cond['all'] = pd.concat([A_by_cond[cond]
                                  for cond in ['rare','frequent','recent','distant','normal','deviant_C']])
    [A_by_cond.update({cond: group_pupil_across_sessions(sessions, list(sessions.keys()), 'A', cond, cond_filters, )})
     for cond in ['hit_all', 'miss_all', 'hit_pattern', 'miss_pattern','hit_none','miss_none']]
    [X_by_cond.update({cond: group_pupil_across_sessions(sessions, list(sessions.keys()), 'X', cond, cond_filters, )})
     for cond in ['earlyX_1tones','earlyX_2tones','earlyX_3tones', ]]
    [X_by_cond.update({cond: group_pupil_across_sessions(sessions, list(sessions.keys()), 'X', cond, cond_filters, )})
     for cond in ['hit_all', 'miss_all', 'hit_pattern', 'miss_pattern','hit_none','miss_none']]
    X_by_cond['all'] = pd.concat([X_by_cond[cond]
                                  for cond in ['rare','frequent','recent','distant','normal','deviant_C']])

    assert pd.concat([A_by_cond['rare'],A_by_cond['frequent']]).drop_duplicates().shape[0] == pd.concat([A_by_cond['rare'],A_by_cond['frequent']]).shape[0]

    # set figsize for pupil ts plots
    matplotlib.rcParams['figure.figsize'] = 4,3
    # assert False

    norm_dev_figdir = ceph_dir / 'Dammy' / 'figures' / f'norm_dev_plots_{"_".join(cohort_tags)}'
    rare_freq_figdir = norm_dev_figdir.parent/f'rare_freq_new_out_method_{"_".join(cohort_tags)}'
    pupil2features_figdir = norm_dev_figdir.parent/f'pupil2features_{"_".join(cohort_tags)}'

    for figdir in [norm_dev_figdir,rare_freq_figdir,pupil2features_figdir]:
        if not figdir.is_dir():
            figdir.mkdir()

    # pupil responses by task features
    AX_plot = plot_pupil_ts_by_cond(X_by_cond, ['all'],)
    plot_pupil_ts_by_cond(A_by_cond, ['all'],plot=AX_plot)
    AX_plot[0].show()


    non_X_lick_pupil = {'none_X_licks':group_pupil_across_sessions(sessions, list(sessions.keys()), 'none_X_licks', 'all',
                                                    use_all=True)}
    non_X_lick_pupil['trial_start'] = group_pupil_across_sessions(sessions, list(sessions.keys()), 'Start',None,
                                                    use_all=True)

    # hit vs miss
    hit_miss_keys = [k for k in X_by_cond.keys() if 'hit' in k or 'miss' in k]
    hit_miss_line_kwargs = {cond: {'c': '#403f4c' if 'hit' in cond else '#9a998c',
                                   'ls': '--' if 'none' in cond else '-'}
                           for cond in hit_miss_keys}
    hit_miss_all_plot = plot_pupil_ts_by_cond(X_by_cond, ['hit_all', 'miss_all'], cond_line_kwargs=hit_miss_line_kwargs)
    hit_miss_all_plot[0].show()
    hit_miss_patt_none_plot = plot_pupil_ts_by_cond(X_by_cond, ['hit_pattern', 'miss_pattern', 'hit_none', 'miss_none'],
                                                    cond_line_kwargs=hit_miss_line_kwargs)
    hit_miss_patt_none_plot[1].locator_params(axis='both', nbins=4)
    hit_miss_patt_none_plot[1].set_title('')
    hit_miss_patt_none_plot[1].axvline(0,c='k',ls='--')

    hit_miss_patt_none_plot[0].show()
    # hit_miss_patt_none_plot[0].savefig(pupil2features_figdir / 'hit_miss_plot.svg')
    # hit vs miss vs patt vs base
    hit_miss_patt_base_plot = plt.subplots(ncols=1,sharey=True,squeeze=False)
    plot_pupil_ts_by_cond(X_by_cond, ['hit_all', 'miss_all'],
                          plot=(hit_miss_patt_base_plot[0],hit_miss_patt_base_plot[1][0,0]),)
    plot_pupil_ts_by_cond(A_by_cond, ['all','none'],
                          plot=(hit_miss_patt_base_plot[0],hit_miss_patt_base_plot[1][0,0]))
    plot_pupil_ts_by_cond(non_X_lick_pupil, ['trial_start', ],
                          plot=(hit_miss_patt_base_plot[0],hit_miss_patt_base_plot[1][0,0]))
    for ax in hit_miss_patt_base_plot[1].flatten():
        ax.axvline(0,c='k',ls='--')
        ax.set_title('')
        ax.locator_params(axis='both', nbins=4)

    hit_miss_patt_base_plot[0].set_layout_engine('tight')
    hit_miss_patt_base_plot[0].set_size_inches(5,3.8)
    hit_miss_patt_base_plot[0].show()
    # hit_miss_patt_base_plot[0].savefig(pupil2features_figdir / 'hit_miss_base_plot.svg')

    # group all lick pupil responses
    none_X_licks_plot = plot_pupil_ts_by_cond(non_X_lick_pupil, ['none_X_licks'])
    plot_pupil_ts_by_cond(X_by_cond, ['all'],plot=none_X_licks_plot)
    # redo legend
    none_X_licks_plot[1].get_legend().remove()
    # none_X_licks_plot[1].legend(['licking','X:hits'], loc='upper center')
    none_X_licks_plot[1].locator_params(axis='both', nbins=4)
    none_X_licks_plot[1].set_title('')
    none_X_licks_plot[1].axvline(0,c='k',ls='--')
    none_X_licks_plot[0].show()
    none_X_licks_plot[0].set_size_inches(2.5,2.2)
    none_X_licks_plot[0].set_layout_engine('tight')
    # none_X_licks_plot[0].savefig(pupil2features_figdir / 'none_X_licks_plot.svg')

    # plot early X pupil
    earlyX_plot = plot_pupil_ts_by_cond(X_by_cond, ['all','earlyX_2tones','earlyX_3tones'])
    # earlyX_plot = plt.subplots()
    # [earlyX_plot[1].plot(X_by_cond[cond].mean(axis=0),label=cond) for cond in['all','earlyX_2tones','earlyX_3tones'] ]
    earlyX_plot[1].legend()
    earlyX_plot[0].show()
    # format
    earlyX_plot[1].set_title('')
    earlyX_plot[1].locator_params(axis='both', nbins=4)
    earlyX_plot[0].set_layout_engine('tight')
    earlyX_plot[1].axvline(0,c='k',ls='--')
    earlyX_plot[0].set_size_inches(4,3)
    earlyX_plot[0].show()
    # earlyX_plot[0].savefig(pupil2features_figdir / 'earlyX_plot.svg')

    # get stats on peak and peak time across sessions
    X_A_max = [response_dict['all'].loc[:,0:].max(axis=1) for response_dict in [X_by_cond, A_by_cond]]
    X_A_peak_t = [response_dict['all'].loc[:,0:].idxmax(axis=1) for response_dict in [X_by_cond, A_by_cond]]

    X_A_max_t_hist = plt.subplots(nrows=2)
    [X_A_max_t_hist[1][0].hist(X_A_max[i],label=lbl,alpha=0.5,density=True, bins='fd')
     for i,(data,lbl) in enumerate(zip(X_A_max,['X','A']))]
    [X_A_max_t_hist[1][1].hist(X_A_peak_t[i],label=lbl,alpha=0.5,density=True,bins='fd')
     for i,(data,lbl) in enumerate(zip(X_A_peak_t,['X','A']))]
    # X_A_max_t_hist[1][0].legend()
    X_A_max_t_hist[1][1].legend()
    [ax.locator_params(axis='both', nbins=4) for ax in X_A_max_t_hist[1]]
    X_A_max_t_hist[0].set_layout_engine('tight')
    X_A_max_t_hist[0].set_size_inches(3,4)
    X_A_max_t_hist[0].show()
    # X_A_max_t_hist[0].savefig(pupil2features_figdir / 'X_A_max_t_hist.svg')
    # # add line for mean and add text with mean
    # [X_A_max_t_hist[1][0].axvline(np.median(X_A_max[i]),ls='--',c='k') for i in range(2)]
    # [X_A_max_t_hist[1][0].text(0.5, 0.8, f'{np.median(X_A_max[i]):.2f}',
    #                            ha='center', va='center', transform=X_A_max_t_hist[1][0].transAxes) for i in range(2)]
    # [X_A_max_t_hist[1][1].axvline(np.median(X_A_peak_t[i]),ls='--',c='k') for i in range(2)]
    # [X_A_max_t_hist[1][1].text(0.5, 0.5, f'{np.median(X_A_peak_t[i]):.2f}',
    #                            ha='center', va='center', transform=X_A_max_t_hist[1][1].transAxes) for i in range(2)]
    #


    # norm vs deviant

    if not norm_dev_figdir.is_dir():
        norm_dev_figdir.mkdir(parents=True)
    # effects over time
    # effects over trial nums
    early_late_dict = {'early': '<=7', 'late': '> <col>.max()-7'}
    filters_early_late = {}
    [[filters_early_late.update({
                                    f'{cond}_{late_early}': f'{cond_filters[cond]} & {cond}_num_cumsum {late_early_filt.replace("<col>", f"{cond}_num_cumsum")}'})
      for late_early, late_early_filt in early_late_dict.items()]
     for cond in ['normal', 'normal_exp','deviant_C']]

    [A_by_cond.update(
        {cond: group_pupil_across_sessions(sessions, list(sessions.keys()), 'A', cond, td_df_query=query)})
     for cond, query in filters_early_late.items()]

    norm_dev_sessions = np.intersect1d(*[A_by_cond[cond].index.get_level_values('sess').values
                                         for cond in ['normal_exp', 'deviant_C']])
    normdev_musc_session = [sess for sess in norm_dev_sessions if sess in drug_sess_dict['muscimol']]
    norm_dev_sessions = [sess for sess in norm_dev_sessions if sess not in drug_sess_dict['muscimol']
                         and sess not in ['DO83_240726'] and A_by_cond['deviant_C'].xs(sess,level='sess').shape[0] > 4]
    norm_dev_sessions = [sess for sess in norm_dev_sessions
                         if len(get_main_sess_patterns(td_df=sessions[sess].td_df,))>=2]
    normdev_line_kwargs = {cond: {'c': 'darkblue' if 'normal' in cond else 'darkred',
                               'ls': '--' if 'late' in cond else '-'}
                           for cond in
                           ['normal_early', 'normal_late', 'deviant_C_early', 'deviant_C_late', 'normal', 'deviant_C',
                            'normal_exp_early', 'normal_exp_late','normal_exp']}

    # plot indvidual sessions
    n_plot_cols = 6
    n_plot_rows = int(np.ceil(len(norm_dev_sessions) / n_plot_cols))
    indv_normdev_plots = plt.subplots(nrows=n_plot_rows, ncols=n_plot_cols, squeeze=False,sharey='all',sharex='col')
    for sess, plot in zip(norm_dev_sessions, indv_normdev_plots[1].flatten()):
        plot_pupil_ts_by_cond(A_by_cond, ['normal', 'deviant_C'], sess_list=[sess], plot=(indv_normdev_plots[0], plot))
        plot.set_title(sess)
        [plot.axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
        plot.axvspan(0.5, 0.5+0.15, fc='red', alpha=0.1)
        # plot.get_legend().remove()
    indv_normdev_plots[0].set_size_inches((n_plot_cols * 3, n_plot_rows * 3))
    indv_normdev_plots[0].set_layout_engine('tight')
    indv_normdev_plots[0].show()
# #     # indv_normdev_plots[0].savefig(norm_dev_figdir / 'indv_sessions.svg',)

    # plot example sesssion
    eg_normdev_sess = 'DO82_240729'
    eg_normdev_plot = plot_pupil_ts_by_cond(A_by_cond, ['normal', 'deviant_C'], sess_list=[eg_normdev_sess],
                                            cond_line_kwargs=normdev_line_kwargs)
    [eg_normdev_plot[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
    eg_normdev_plot[1].axvspan(0.5, 0.5 + 0.15, fc='red', alpha=0.1)
    eg_normdev_plot[1].locator_params(axis='both', nbins=4)
    eg_normdev_plot[1].set_title('')
    eg_normdev_plot[1].get_legend().remove()
    eg_normdev_plot[0].set_layout_engine('tight')
    # eg_normdev_plot[0].set_size_inches((4, 4))
    eg_normdev_plot[0].show()
    # indv_normdev_plots[0].set_title(eg_sess)
#     # eg_normdev_plot[0].savefig(norm_dev_figdir / f'example_norm_dev_sess_{eg_normdev_sess}.svg')

    eg_rarefreq_sess = 'DO85_240711'
    rare_freq_cond_line_kwargs = {cond: {'c': 'darkblue' if 'distant' in cond else 'darkgreen',
                                         'ls': '--' if 'late' in cond else '-'}
                                  for cond in
                                  ['distant', 'recent']}
    eg_rarefreq_plot = plot_pupil_ts_by_cond(A_by_cond, ['distant', 'recent'], sess_list=[eg_rarefreq_sess],
                                             cond_line_kwargs=rare_freq_cond_line_kwargs)
    [eg_rarefreq_plot[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
    eg_rarefreq_plot[1].locator_params(axis='both', nbins=4)
    eg_rarefreq_plot[1].set_title('')
    eg_rarefreq_plot[1].get_legend().remove()
    eg_rarefreq_plot[0].set_layout_engine('tight')
    eg_rarefreq_plot[0].set_size_inches((3.8, 2.850))
    eg_rarefreq_plot[0].show()
    # indv_normdev_plots[0].set_title(eg_sess)
#     # eg_rarefreq_plot[0].savefig(rare_freq_figdir / f'example_rare_freq_sess_{eg_rarefreq_sess}.svg')

    norm_dev_early_late_plots = [plot_pupil_ts_by_cond(A_by_cond, [f'{cond}_early', f'{cond}_late'],
                                                       sess_list=norm_dev_sessions,
                                                       cond_line_kwargs=normdev_line_kwargs)
                                 for cond in ['normal', 'deviant_C']]

    norm_exp_vs_norm_late_plot = plot_pupil_ts_by_cond(A_by_cond, ['normal_exp_early','normal_exp_late','normal_late'],
                                                       sess_list=norm_dev_sessions,)
    norm_exp_vs_norm_late_plot[1].get_legend().remove()
    norm_exp_vs_norm_late_plot[1].legend(ncols=1, loc='upper right')

    [norm_exp_vs_norm_late_plot[1].axvspan(t, t + 0,.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
    # norm_exp_vs_norm_late_plot[0].set_size_inches()
    norm_exp_vs_norm_late_plot[0].show()
    norm_exp_vs_norm_late_plot[0].set_layout_engine('tight')

    # norm_exp_vs_norm_late_plot[0].savefig(norm_dev_figdir / 'norm_exp_vs_norm_late_plot.svg')

    norm_dev_plot = plot_pupil_ts_by_cond(A_by_cond, ['normal_exp', 'deviant_C'], sess_list=norm_dev_sessions,
                                          cond_line_kwargs=normdev_line_kwargs)
    norm_dev_plot[0].show()
    ylim = [min([plot[1].get_ylim()[0] for plot in [*norm_dev_early_late_plots, norm_dev_plot]]),
            max([plot[1].get_ylim()[1] for plot in [*norm_dev_early_late_plots, norm_dev_plot]])]
    for plot, savename in zip([*norm_dev_early_late_plots, norm_dev_plot],
                              ['norm_early_late_plot', 'dev_early_late_plot', 'norm_dev_plot'],):
        plot[1].set_title('')
        plot[1].set_ylim(ylim)
        # ylim = copy(plot[1].get_ylim())
        plot[1].set_yticks(np.arange(np.round(ylim[0] * 2) / 2, ylim[1], 1 / 2))
        plot[1].locator_params(axis='x', nbins=4)
        # plot[1].axvline(0,c='k',ls='--')
        [plot[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
        plot[1].axvspan(0.5, 0.5 + 0.15, fc='red', alpha=0.1)
        # plot[1].get_legend().remove()
        plot[0].set_layout_engine('tight')
        plot[0].set_size_inches(3.6,2.7)
        plot[0].show()
        # plot[0].savefig(norm_dev_figdir / f'{savename}.svg')

    norm_dev_diff_plot = plot_pupil_diff_ts_by_cond(A_by_cond, ['deviant_C','normal'], sess_list=norm_dev_sessions,
                                                    plot_indv_group=None)

    # norm_dev_diff_plot = plot_pupil_diff_ts_by_cond(A_by_cond, ['normal_early', 'normal_late', 'deviant_C_early', 'deviant_C_late'],
    #                                                 sess_list=norm_dev_sessions )
    norm_dev_diff_plot[1].locator_params(axis='both', nbins=4)
    norm_dev_diff_plot[1].set_title('')
    norm_dev_diff_plot[1].set_ylabel('')
    # norm_dev_diff_plot[1].set_ylim(-0.5, .8)
    norm_dev_diff_plot[1].set_xlabel('')
    [norm_dev_diff_plot[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
    norm_dev_diff_plot[1].axvspan(0.5,0.5+0.15,fc='red',alpha=0.1)
    norm_dev_diff_plot[1].axhline(0, c='k', ls='--')
    norm_dev_diff_plot[0].set_layout_engine('tight')
    # norm_dev_diff_plot[0].set_size_inches(5,4)
    # # norm_dev_diff_plot[0].set_size_inches(3, 3)
    norm_dev_diff_plot[0].show()
#     # norm_dev_diff_plot[0].savefig(norm_dev_figdir / 'norm_dev_diff_plot_indv_animal_lines.svg')

    # plot max diff over window
    norm_dev_max_diff_plot,normdev_diff_data = plot_pupil_diff_max_by_cond(A_by_cond, ['normal_exp_midlate', 'deviant_C'],
                                                                           sess_list=norm_dev_sessions,
                                                                           window_by_stim=[(1.5,2.5),(1.5,2.5)][0],
                                                                           mean=np.max,
                                                                           plot_kwargs={'showfliers':False,
                                                                                        'labels':['Pattern'],
                                                                                        'showmeans':False,
                                                                                        'widths':0.3},
                                                                           group_name='name',
                                                                           diff_kwargs={'sub_by_sess':False},
                                                                           permutation_test=True,
                                                                           n_permutations=1000)

    norm_dev_max_diff_plot[1].set_ylabel('')
    norm_dev_max_diff_plot[1].axhline(0, c='k', ls='--')
    norm_dev_max_diff_plot[1].set_title('')
    norm_dev_max_diff_plot[1].locator_params(axis='y', nbins=4)
    norm_dev_max_diff_plot[0].set_layout_engine('tight')
    norm_dev_max_diff_plot[0].set_size_inches(2.7,2.7)
    norm_dev_max_diff_plot[0].show()
    norm_dev_max_diff_plot[0].savefig(norm_dev_figdir / 'norm_dev_max_diff_plot_w_shuffle.svg')

    # hist on data
    norm_dev_diff_hist = plt.subplots()
    norm_dev_diff_hist[1].hist(normdev_diff_data[0],label='A',alpha=0.5,density=True,bins='fd')
    # norm_dev_diff_hist[1].hist(normdev_diff_data[1],label='X',alpha=0.5,density=True)
    norm_dev_diff_hist[0].show()
    # norm_dev_diff_hist[0].savefig(norm_dev_figdir / 'norm_dev_diff_hist.svg')

    # ttest on diff data
    ttest_ind(normdev_diff_data[0][0].values,normdev_diff_data[1][0].values,alternative='greater')
    ttest_1samp(normdev_diff_data[0],0,alternative='greater')

    # alt vs rand analysis
    [A_by_cond.update({cond: group_pupil_across_sessions(sessions, list(sessions.keys()), 'A', cond, cond_filters, )})
     for cond in ['alternating', 'random']]
    [X_by_cond.update({cond: group_pupil_across_sessions(sessions, list(sessions.keys()), 'X', cond, cond_filters, )})
     for cond in ['alternating', 'random']]

    early_late_dict = {'early': '<=10', 'late': '> <col>.max()-10'}
    filters_early_late = {}
    [[filters_early_late.update({
                                    f'{cond}_{late_early}': f'{cond_filters[cond]} & {cond}_num_cumsum {late_early_filt.replace("<col>", f"{cond}_num_cumsum")}'})
      for late_early, late_early_filt in early_late_dict.items()]
     for cond in ['alternating', 'random','normal_exp']]

    [A_by_cond.update(
        {cond: group_pupil_across_sessions(sessions, list(sessions.keys()), 'A', cond, td_df_query=query)})
     for cond, query in filters_early_late.items() if
     cond in ['alternating_early', 'alternating_late', 'random_early', 'random_late', 'normal_exp_early', 'normal_exp_late']]

    alternating_sessions = np.intersect1d(*[A_by_cond[cond].index.get_level_values('sess').values
                                            for cond in ['alternating', 'random']])
    alternating_sessions = [sess for sess in alternating_sessions if sess in drug_sess_dict['none'] and 'DO84' not in sess]
    alternating_sessions = [sess for sess in alternating_sessions if
                            A_by_cond['alternating'].xs(sess, level='sess').shape[0] > 50 or
                            A_by_cond['random'].xs(sess, level='sess').shape[0] > 50]
    alternating_figdir = norm_dev_figdir.parent / f'alternating_{"_".join(cohort_tags)}'
    if not alternating_figdir.is_dir():
        alternating_figdir.mkdir()

    # alternating vs random ts plot
    alt_rand_plot = plot_pupil_ts_by_cond(A_by_cond, ['alternating', 'random'],
                                          sess_list=alternating_sessions)
    alt_rand_plot[1].locator_params(axis='both', nbins=4)
    alt_rand_plot[1].set_title('')
    [alt_rand_plot[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
    alt_rand_plot[1].axhline(0, c='k', ls='--')
    alt_rand_plot[0].set_layout_engine('tight')
    alt_rand_plot[0].set_size_inches(4,3)
    alt_rand_plot[0].show()
    # alt_rand_plot[0].savefig(alternating_figdir / 'alt_rand_plot.svg')

    # early late comp
    for cond in ['alternating', 'random']:
        early_late_plot = plot_pupil_ts_by_cond(A_by_cond, [f'{cond}_early', f'{cond}_late'],
                                                 sess_list=alternating_sessions)
        early_late_plot[1].locator_params(axis='both', nbins=4)
        early_late_plot[1].set_title('')
        [early_late_plot[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
        early_late_plot[1].axhline(0, c='k', ls='--')
        early_late_plot[1].set_ylim(-.5,1)
        early_late_plot[0].set_layout_engine('tight')
        early_late_plot[0].set_size_inches(4,3)
        early_late_plot[0].show()
        # early_late_plot[0].savefig(alternating_figdir / f'{cond}_early_late_plot.svg')

    # diff plot for alternating vs random
    alternating_diff_plot = plot_pupil_diff_ts_by_cond(A_by_cond, ['random', 'alternating'],
                                                       sess_list=alternating_sessions,
                                                       # plot_indv_group='name',
                                                       group_level='name'
                                                       )
    alternating_diff_plot[1].set_ylabel('')
    alternating_diff_plot[1].set_title('')
    [alternating_diff_plot[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
    alternating_diff_plot[1].axhline(0, c='k', ls='--')
    alternating_diff_plot[1].locator_params(axis='y', nbins=4)
    alternating_diff_plot[0].set_layout_engine('tight')
    alternating_diff_plot[0].set_size_inches(4,3)
    alternating_diff_plot[0].show()
    # alternating_diff_plot[0].savefig(alternating_figdir / 'alternating_diff_plot.svg')

    # max diff plot
    alt_rand_max_diff_plot,alt_rand_max_diff = plot_pupil_diff_max_by_cond(A_by_cond,
                                                                           ['alternating', 'random'],
                                                                           sess_list=alternating_sessions,
                                                                           window_by_stim=[(1.5,2.5),(1.5,2.5)][0],
                                                                           mean=np.max,
                                                                           plot_kwargs={'showfliers':False,
                                                                                        'labels':['Pattern'],
                                                                                        },
                                                                           group_name='name')

    alt_rand_max_diff_plot[1].set_ylabel('')
    alt_rand_max_diff_plot[1].set_title('')
    alt_rand_max_diff_plot[1].axhline(0, c='k', ls='--')
    alt_rand_max_diff_plot[1].locator_params(axis='y', nbins=4)
    alt_rand_max_diff_plot[0].set_layout_engine('tight')
    alt_rand_max_diff_plot[0].set_size_inches(3,3)
    alt_rand_max_diff_plot[0].show()
    # alt_rand_max_diff_plot[0].savefig(alternating_figdir / 'alt_rand_max_diff_plot.svg')

    # ttest
    ttest = ttest_1samp(alt_rand_max_diff[0],0,alternative='greater')
    print(ttest)
    # hist pot on data
    alt_rand_diff_hist = plt.subplots()
    alt_rand_diff_hist[1].hist(alt_rand_max_diff,alpha=0.5,density=True, bins='fd')
    alt_rand_diff_hist[0].show()

    names = list(A_by_cond['alternating'].index.get_level_values('name').unique())
    names.remove('DO84')
    alt_diff_by_name_plot = plt.subplots(len(names),sharey=True)
    # [plot_pupil_diff_ts_by_cond(A_by_cond, ['alternating', 'random'], plot=alt_diff_by_name_plot,
    #                       sess_list=[s for s in alternating_sessions if n in s]) for n in names]
    [plot_pupil_diff_ts_by_cond(A_by_cond, ['alternating', 'random'],
                                plot=(alt_diff_by_name_plot[0],ax),
                                sess_list=[s for s in alternating_sessions if n in s])
     for n,ax in zip(names,alt_diff_by_name_plot[1])]
    # alt_diff_by_name_plot[1].set_ylabel('')
    # alt_diff_by_name_plot[1].set_title('')
    # [alt_diff_by_name_plot[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
    # alt_diff_by_name_plot[1].axhline(0, c='k', ls='--')
    # alt_diff_by_name_plot[1].locator_params(axis='y', nbins=4)
    alt_diff_by_name_plot[0].set_size_inches(3, 3*len(names))
    alt_diff_by_name_plot[0].set_layout_engine('tight')
    alt_diff_by_name_plot[0].show()
#     # alternating_diff_plot[0].savefig(alternating_figdir / 'alternating_diff_plot.svg')

    for condition, _sessions in zip(['alternating', 'random','normal_exp'][:2],
                                    [alternating_sessions,alternating_sessions,norm_dev_sessions]):
        alt_rand_plot = plot_pupil_ts_by_cond(A_by_cond, [f'{condition}_early', f'{condition}_late'],
                                              sess_list=_sessions, group_name='sess',
                                              plot_indv_sess=False)
        alt_rand_plot[1].locator_params(axis='both', nbins=4)
        alt_rand_plot[1].set_title('')
        [alt_rand_plot[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
        alt_rand_plot[1].axhline(0, c='k', ls='--')
        alt_rand_plot[0].set_layout_engine('tight')
#         # alternating_early_late_plot[0].set_size_inches(3,3)
        alt_rand_plot[0].show()
#         # alternating_early_late_plot[0].savefig(alternating_figdir / f'{condition}_early_late_plot.svg')

        # diff plot
        diff_plot = plot_pupil_diff_ts_by_cond(A_by_cond, [f'{condition}_early', f'{condition}_late'],
                                               sess_list=_sessions)
        diff_plot[1].locator_params(axis='both', nbins=4)
        diff_plot[1].set_title('')
        [diff_plot[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
        diff_plot[1].axhline(0, c='k', ls='--')
        diff_plot[0].set_layout_engine('tight')
#         # diff_plot[0].set_size_inches(3,3)
        diff_plot[0].show()
#         # diff_plot[0].savefig(alternating_figdir / f'{condition}_diff_plot.svg')

    # plot indv alternating sessions
    all_alt_sess_plot = plt.subplots(ncols=6,nrows=len(alternating_sessions)//6, sharex=True, sharey=True)
    [plot_pupil_ts_by_cond(A_by_cond, ['alternating','random'], sess_list=[s],plot_indv_sess=True,
                           plot=(all_alt_sess_plot[0],ax))
     for ax,s in zip(all_alt_sess_plot[1].flatten(),alternating_sessions)]
    [ax.set_title(f'{s}') for ax,s in zip(all_alt_sess_plot[1].flatten(),alternating_sessions)]
    # [ax.set_xlabel('') for ax in all_alt_sess_plot[1].flatten()[:-1]
    all_alt_sess_plot[0].set_layout_engine('tight')
    all_alt_sess_plot[0].set_size_inches(3*6,3*len(alternating_sessions)//6)
    all_alt_sess_plot[0].show()


    # rare vs frequent analysis
    [[A_by_cond.update(
        {f'{cond}_{block_i}': group_pupil_across_sessions(sessions, list(sessions.keys()), 'A', cond,
                                                          td_df_query=cond_filters[
                                                                          cond] + f' & {cond}_block_num== {block_i}')})
        # {"0.9" if cond == "rare" else "0.1"}
        for block_i in range(1, 4)]
        for cond in ['distant', 'recent', 'rare', 'frequent']]

    # plot rare vs frequent
    rare_freq_cond_line_kwargs = {cond: {'c': 'darkblue' if any([filt in cond for filt in ['rare', 'distant']]) else 'darkgreen',
                                  'ls': '--' if 'late' in cond else '-'}
                                  for cond in
                                  ['distant_1', 'recent_1', 'distant_2', 'recent_2', 'distant_3', 'recent_3']+
                                  ['recent','distant','rare','frequent']}
    rare_freq_sessions = np.intersect1d(
        *[A_by_cond[cond].index.get_level_values('sess').values for cond in ['recent', 'distant']])
    rare_freq_sessions = [sess for sess in rare_freq_sessions if sess in drug_sess_dict['none']
                          and all([A_by_cond[cond].xs(sess, level='sess').shape[0]>6
                                   for cond in ['recent', 'distant']])]
    rare_freq_figdir = norm_dev_figdir.parent / f'rare_freq_{"_".join(cohort_tags)}'
    if not rare_freq_figdir.is_dir():
        rare_freq_figdir.mkdir()
    rare_freq_plot = plot_pupil_ts_by_cond(A_by_cond, ['frequent', 'rare',], sess_list=rare_freq_sessions,
                                           group_name='sess', plot_indv_sess=False,
                                           cond_line_kwargs=rare_freq_cond_line_kwargs)
    rare_freq_plot[1].locator_params(axis='both', nbins=4)
    rare_freq_plot[1].set_title('')
    [rare_freq_plot[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
    rare_freq_plot[1].axhline(0, c='k', ls='--')
    rare_freq_plot[0].set_layout_engine('tight')
    rare_freq_plot[0].show()
# #     # rare_freq_plot[0].savefig(rare_freq_figdir / 'rare_freq_plot.svg')

    rare_freq_diff_by_block_plot = plt.subplots()
    for i in range(3):
        plot_pupil_diff(A_by_cond, [f'rare_{i+1}', f'frequent_{i+1}'], rare_freq_sessions, rare_freq_diff_by_block_plot)

    # Plot rare frequent diff
    rare_freq_diff_plot = plot_pupil_diff_ts_by_cond(A_by_cond, ['distant', 'recent'], sess_list=rare_freq_sessions, cond_line_kwargs=None)
    configure_plot(rare_freq_diff_plot[1], nbins=4, vspans=[(t, t + 0.15) for t in np.arange(0, 1, 0.25)])
    rare_freq_diff_plot[0].set_layout_engine('tight')
    rare_freq_diff_plot[0].show()

    # plot max diff over window
    rare_freq_max_diff_plot, rare_freq_diff_data = plot_pupil_diff_max_by_cond(A_by_cond, ['recent', 'distant' ],
                                                                               sess_list=rare_freq_sessions,
                                                                               window_by_stim=[(1.5, 2.5)][0],
                                                                               mean=np.max,
                                                                               plot_kwargs={'showfliers': False,
                                                                                            'showmeans': False,
                                                                                            'labels': ['Pattern'],
                                                                                            'widths': 0.3
                                                                                            },
                                                                               permutation_test=True,
                                                                               group_name='name')

    rare_freq_max_diff_plot[1].set_ylabel('')
    rare_freq_max_diff_plot[1].set_title('')
    rare_freq_max_diff_plot[1].axhline(0, c='k', ls='--')
    rare_freq_max_diff_plot[1].locator_params(axis='y', nbins=4)
    rare_freq_max_diff_plot[0].set_layout_engine('tight')
    rare_freq_max_diff_plot[0].set_size_inches(2.7, 2.7)
    rare_freq_max_diff_plot[0].show()
    rare_freq_max_diff_plot[0].savefig(rare_freq_figdir / 'rare_freq_max_diff_plot_w_shuffle.svg')

    # ttest
    rare_freq_max_diff_ttest = ttest_ind(rare_freq_diff_data[0][0].values, rare_freq_diff_data[1][0].values,
                                         alternative='greater')
    # rare_freq_max_diff_ttest = ttest_1samp(rare_freq_diff_data[0], 0.0, alternative='greater')
    print(rare_freq_max_diff_ttest)

    # plot rare by block num
    rare_by_block_figdir = norm_dev_figdir.parent / f'rare_by_block_{"_".join(cohort_tags)}'
    if not rare_by_block_figdir.is_dir():
        rare_by_block_figdir.mkdir()
    for rare_freq in ['rare', 'frequent']:
        rare_by_block_plot = plt.subplots()
        # rare_freq_sessions = np.intersect1d(
        #     *[A_by_cond[cond].index.get_level_values('sess').values for cond in [f'{rare_freq}_{i}' for i in [1, 3]]])

        plot_pupil_ts_by_cond(A_by_cond, [f'{rare_freq}_{i}' for i in [1, 2, 3]],
                                                   sess_list=rare_freq_sessions,
                                                   plot=rare_by_block_plot,
                                                   cond_line_kwargs=None)
        rare_by_block_plot[1].locator_params(axis='x', nbins=4)
        ylim = copy(rare_by_block_plot[1].get_ylim())
        # rare_by_block_plot[1].set_yticks(np.arange(np.round(ylim[0] * 2) / 2,1))  # ylim[1], 1 / 2)
        rare_by_block_plot[1].set_yticks(np.arange(0, 1.25, 0.5))  # ylim[1], 1 / 2)
        rare_by_block_plot[1].set_ylim(-0.25, 1)
        rare_by_block_plot[1].set_title('')
        [rare_by_block_plot[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
        rare_by_block_plot[1].axhline(0, c='k', ls='--')
        rare_by_block_plot[0].set_layout_engine('tight')
#         # rare_by_block_plot[0].set_size_inches(3,3)
        rare_by_block_plot[0].show()
#         # rare_by_block_plot[0].savefig(rare_by_block_figdir / f'{rare_freq}_by_block_plot.svg')


    # rare freq within block adaptation

    # mega mouse by cond
    conds = ['alternating_early', 'alternating_late', 'random_early', 'random_late']
    conds = ['distant', 'frequent']
    mega_mouse_by_cond_plot = plt.subplots()
    for ci, cond in enumerate(conds[:]):
        # cond_pupil = A_by_cond[cond].query('sess in @alternating_sessions').groupby('name').mean()
        cond_pupil = A_by_cond[cond].query('sess in @rare_freq_sessions').groupby(['sess']).median()
        mega_mouse_by_cond_plot[1].plot(cond_pupil.mean(axis=0),label=cond)
        mega_mouse_by_cond_plot[1].fill_between(cond_pupil.columns.tolist(),
                                                cond_pupil.mean(axis=0) - cond_pupil.sem(axis=0),
                                                cond_pupil.mean(axis=0) + cond_pupil.sem(axis=0),
                                                alpha=0.1,fc=f'C{ci}')

    mega_mouse_by_cond_plot[1].legend()
    mega_mouse_by_cond_plot[1].axvline(0,c='k',ls='--')
    mega_mouse_by_cond_plot[0].show()


    # rare freq interstim dist
    # use block 1 only
    for metric in ['time']:
        interstim_by_cond = [[A_by_cond[f'{cond}_1'].xs(sess,level='sess').index.get_level_values(metric).to_series().diff().tolist()
                              for sess in A_by_cond[f'{cond}_1'].index.get_level_values('sess').unique()]
                             for cond in ['recent','distant']]
        interstim_by_cond = [np.hstack(interstim_by_cond[i]) for i in range(len(interstim_by_cond))]
        interstim_by_cond = [arr[~np.isnan(arr)] for arr in interstim_by_cond]
        interstim_by_cond_plot = plt.subplots()
        bxplt = interstim_by_cond_plot[1].boxplot(interstim_by_cond,labels=['recent','distant'],showfliers=False,)
        [patch.set_facecolor(rare_freq_cond_line_kwargs[cond]['c'])
         for patch, cond in zip(bxplt['boxes'], ['recent','distant'])]
        interstim_by_cond_plot[0].show()
        interstim_by_cond_plot[0].set_layout_engine('tight')
        interstim_by_cond_plot[0].set_size_inches(2.7,2.7)
        # interstim_by_cond_plot[0].savefig(rare_by_block_figdir / f'interstim_{metric}_by_cond.svg')

    all_sess_td_df = group_td_df_across_sessions(sessions, list(sessions.keys()))
    all_dists_df = all_sess_td_df.query(cond_filters['distant'])
    all_dist_patts = []
    for sess in all_dists_df.index.get_level_values('sess').unique():
        all_dist_patts.append([all_dists_df.xs(sess, level='sess').index.get_level_values('trial_num').to_series(),
                               A_by_cond['distant'].xs(sess, level='sess').index.get_level_values('trial').to_series()])

    d_trials_plot = plt.subplots()
    for cond in ['rare_prate','frequent_prate']:
        cond_td_df_all_sess = all_sess_td_df.query(cond_filters[cond])
        d_trials = np.hstack([cond_td_df_all_sess.xs(sess, level='sess').index.get_level_values('trial_num').to_series().diff().dropna()
                              for sess in cond_td_df_all_sess.index.get_level_values('sess').unique()])
        d_trials_plot[1].hist(d_trials,bins=np.arange(min(d_trials),max(d_trials)+1,1),density=True,alpha=0.5,label=cond,
                              color=rare_freq_cond_line_kwargs[cond.replace('_prate','')]['c'])
    d_trials_plot[1].legend(['rare','frequent'])
    d_trials_plot[1].set_xlim(0.25,25)
    d_trials_plot[1].axvline(3,c='k',ls='--')
    d_trials_plot[1].locator_params(axis='both', nbins=3)
    d_trials_plot[0].set_layout_engine('tight')
    d_trials_plot[0].set_size_inches(2,2.7)
    d_trials_plot[0].show()
    d_trials_plot[0].savefig(rare_by_block_figdir / 'd_trials.pdf')

    d_trials_plot = plt.subplots()
    d_trals_by_cond = {}
    for cond in ['rare_prate','frequent_prate']:
        cond_td_df_all_sess = all_sess_td_df.query(cond_filters[cond])
        d_trials = np.hstack([cond_td_df_all_sess.xs(sess, level='sess')['ToneTime_dt'].diff().dropna().dt.total_seconds()
                              for sess in cond_td_df_all_sess.index.get_level_values('sess').unique()])
        d_trals_by_cond[cond] = d_trials
    all_dt_times = np.hstack(list(d_trals_by_cond.values()))
    min_dt,max_dt = np.percentile(all_dt_times,[1,97.25])
    bins2use,_ = np.histogram(all_dt_times[np.logical_and(all_dt_times>min_dt,all_dt_times<max_dt)], bins='fd')
    for cond in ['rare_prate', 'frequent_prate']:
        dt_tone_2plot = d_trals_by_cond[cond][np.logical_and(d_trals_by_cond[cond]>min_dt,d_trals_by_cond[cond]<max_dt)]
        d_trials_plot[1].hist(dt_tone_2plot,bins=bins2use.sort(),density=False,alpha=0.5,label=cond,
                              color=rare_freq_cond_line_kwargs[cond.replace('_prate','')]['c'])
        d_trials_plot[1].axvline(dt_tone_2plot.mean(),c='k',ls='--',lw=1)
        print(f'{cond}: {dt_tone_2plot.mean():.2f} s')
    d_trials_plot[1].legend(['rare','frequent'])
    # d_trials_plot[1].set_xlim(0,)
    # d_trials_plot[1].axvline(3,c='k',ls='--')
    d_trials_plot[1].locator_params(axis='both', nbins=3)
    d_trials_plot[0].set_layout_engine('tight')
    d_trials_plot[0].set_size_inches(2,2.7)
    d_trials_plot[0].show()
    d_trials_plot[0].savefig(rare_by_block_figdir / 'dt_pattern_onset.pdf')

    # distribution of tone time
    all_tones_trials = all_sess_td_df.query('Tone_Position==0 & Early_Licks==0 & N_TonesPlayed==4')
    all_pretone_durs = (all_tones_trials['ToneTime_dt'] - all_tones_trials['Trial_Start_dt']).dt.total_seconds().values
    all_posttone_durs = (all_tones_trials['Gap_Time_dt'] - all_tones_trials['ToneTime_dt']).dt.total_seconds().values
    tone_time_plot = plt.subplots()
    tone_time_plot[1].hist(all_pretone_durs, bins=np.arange(np.floor(all_pretone_durs.min()), np.ceil(all_pretone_durs.max())),
                           density=True, alpha=0.2, label='pre-tone')
    tone_time_plot[1].hist(all_posttone_durs, bins=np.arange(np.floor(all_posttone_durs.min()), np.ceil(all_posttone_durs.max())),
                           density=True, alpha=0.2, label='post-tone')
    tone_time_plot[1].locator_params(axis='both', nbins=4)
    tone_time_plot[0].set_layout_engine('tight')
    # tone_time_plot[0].set_size_inches(2, 2.7)
    tone_time_plot[0].show()
    # tone_time_plot[0].savefig(rare_by_block_figdir / 'tone_time.svg')

    # ABCD0 vs ABCD1 vs ABBA1
    all_abstr_df = all_sess_td_df.query(cond_filters['dev_ABCD1'])
    for sessname in all_abstr_df.index.get_level_values('sess').unique():
        main_patterns = get_main_sess_patterns(td_df=sessions[sessname].td_df)
        events = {e: {'idx': idx, 'filt': filt} for e, idx, filt in zip(['A_dev', ],
                                                                        [main_patterns[1][0]],
                                                                        ['pip_counter==1'])}
        [sessions[sessname].get_pupil_to_event(e_dict['idx'], e, window,
                                               align_kwargs=dict(sound_df_query=e_dict['filt'], baseline_dur=1),
                                               alignmethod='w_soundcard')
         for e, e_dict in events.items()]

    [A_by_cond.update({cond: group_pupil_across_sessions(sessions, all_abstr_df.index.get_level_values('sess').unique(),
                                                         'A_dev',cond_name=cond,
                                                         cond_filters=cond_filters)})
     for cond in ['dev_ABCD1','dev_ABBA1']]
    [X_by_cond.update({cond: group_pupil_across_sessions(sessions, all_abstr_df.index.get_level_values('sess').unique(),
                                                         'X',cond_name=cond,
                                                         cond_filters=cond_filters)})
     for cond in ['dev_ABCD1','dev_ABBA1']]
    abstraction_sessions = np.intersect1d(
        *[A_by_cond[cond].index.get_level_values('sess').values for cond in ['normal_exp', 'dev_ABCD1']])
    abstraction_sessions = [sess for sess in abstraction_sessions if sess in drug_sess_dict['none']
                            and A_by_cond['dev_ABCD1'].xs(sess, level='sess').shape[0] > 3]

    # plot pupil_ts
    abstr_figdir = norm_dev_figdir.parent / 'abstraction'
    if not abstr_figdir.is_dir():
        abstr_figdir.mkdir()
    abstr_cond_line_kwargs = {cond: {'c': c,'lw':2} for cond, c in zip(['normal','dev_ABCD1', 'dev_ABBA1'],
                                                                ['darkblue', 'lightcoral','peru'])}
    abstraction_plot = plot_pupil_ts_by_cond(A_by_cond,['normal','dev_ABCD1','dev_ABBA1'],
                                             sess_list=abstraction_sessions,
                                             cond_line_kwargs=abstr_cond_line_kwargs,
                                             )
    abstraction_plot[0].set_layout_engine('tight')
    # abstraction_plot[0].set_size_inches(3,3)
    # format plot
    abstraction_plot[1].axhline(0, c='k', ls='--')
    abstraction_plot[1].locator_params(axis='both', nbins=4)
    [abstraction_plot[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
    abstraction_plot[1].set_title('')
    abstraction_plot[0].set_layout_engine('tight')
    abstraction_plot[0].show()
    # abstraction_plot[0].savefig(abstr_figdir / 'abstraction.svg')
    # plot pupil_diff
    abstraction_diff_plot = plot_pupil_diff_ts_by_cond(A_by_cond,['normal','dev_ABCD1','dev_ABBA1'],
                                                       sess_list=abstraction_sessions,invert=True,
                                                       )
    abstraction_diff_plot[1].set_ylabel('')
    abstraction_diff_plot[1].set_title('')
    abstraction_diff_plot[1].axhline(0, c='k', ls='--')
    abstraction_diff_plot[1].locator_params(axis='both', nbins=4)
    [abstraction_diff_plot[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
    abstraction_diff_plot[0].set_layout_engine('tight')
    # abstraction_diff_plot[0].set_size_inches(3,3)
    abstraction_diff_plot[0].show()
    # abstraction_diff_plot[0].savefig(abstr_figdir / 'abstraction_diff.svg')

    # plot max diff
    abstraction_max_diff_plot,abst_data = plot_pupil_diff_max_by_cond(A_by_cond,['normal','dev_ABCD1','dev_ABBA1'],
                                                            sess_list=abstraction_sessions,
                                                            window_by_stim=[(1, 2.5), (1.5, 2.5)][0],
                                                            mean=np.max,
                                                            plot_kwargs={'showfliers': False,
                                                                         'showmeans': False,
                                                                         'labels': ['dev_ABCD1', 'dev_ABBA1'],
                                                                         'bootstrap':1000,
                                                                         # 'title':'',
                                                                         'widths': 0.3
                                                                         },
                                                            group_name='name',
                                                            permutation_test=True
                                                            )
    # format plot
    abstraction_max_diff_plot[1].set_ylabel('')
    abstraction_max_diff_plot[1].set_title('')
    abstraction_max_diff_plot[1].locator_params(axis='y', nbins=4)
    abstraction_max_diff_plot[0].set_layout_engine('tight')
    # abstraction_max_diff_plot[0].set_size_inches(3,3)
    abstraction_max_diff_plot[0].show()
    abstraction_max_diff_plot[0].savefig(abstr_figdir / 'abstraction_max_diff_w_shuffle.svg')


    ttest_ind(abst_data[0][0],abst_data[1][0],alternative='greater')
    ttest_ind(abst_data[0][1],abst_data[1][1],alternative='greater')
    # ttest_1samp(abst_data[0],0,alternative='greater')

    # dump data
    processed_pkl_dir = ceph_dir/ r'X:\Dammy\Xdetection_mouse_hf_test\processed_data'
    save_responses_dicts(A_by_cond,processed_pkl_dir/f'A_by_cond_{"_".join(cohort_tags)}.pkl')
    save_responses_dicts(X_by_cond,processed_pkl_dir/f'X_by_cond_{"_".join(cohort_tags)}.pkl')
    # save_responses_dicts(sessions,processed_pkl_dir/f'sessions_no_licks_{"_".join(cohort_tags)}.pkl')
    save_responses_dicts(sessions,processed_pkl_dir/f'sessions_w_licks_{"_".join(cohort_tags)}.pkl')