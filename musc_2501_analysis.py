from pupil_analysis_funcs import *

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
        ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_musc_2501.csv'),
    ]
    pupil_pkl_paths = [ceph_dir / posix_from_win(p) for p in [
        r'X:\Dammy\mouse_pupillometry\pickles\mouse_hf_musc_2501_allsess_250301_fam_2d_90Hz_hpass01_lpass4_TOM.pkl',
    ]]

    all_pupil_data = [load_pupil_data(pkl_path) for pkl_path in pupil_pkl_paths]
    pupil_data_dict = all_pupil_data[0]
    if len(all_pupil_data) > 1:
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

    sess_pkl_path = ceph_dir / posix_from_win(r'D:\Dammy\sess_obj_pkls') / 'musc_2501_cohort_sess_dicts.pkl'

    sessions = load_pupil_sess_pkl(sess_pkl_path)

    existing_sessions = list(sessions.keys())

    window = [-1, 4]

    for sessname in tqdm(list(pupil_data_dict.keys()), desc='processing sessions'):
        print(sessname)
        if sessname in existing_sessions or args.ow:
            continue

        init_pupil_td_obj(sessions, sessname, ceph_dir, all_sess_info, td_path_pattern, home_dir)
        process_pupil_td_data(sessions,sessname,drug_sess_dict)

    for sessname in tqdm(list(sessions.keys()), desc='processing sessions'):
        if sessions[sessname].pupil_obj is not None or args.ow:
            continue
        print(sessname)

        init_sess_pupil_obj(sessions, sessname, ceph_dir, all_sess_info, pupil_data_dict)
        process_pupil_obj(sessions,sessname,alignmethod='w_soundcard')

    # save pkl
    with open(sess_pkl_path, 'wb') as pklfile:
        pickle.dump(sessions, pklfile)

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

    # plot rare vs frequent
    rare_freq_cond_line_kwargs = {
        cond: {'c': 'darkblue' if any([filt in cond for filt in ['rare', 'distant']]) else 'darkgreen',
               'ls': '--' if 'late' in cond else '-'}
        for cond in
        ['distant_1', 'recent_1', 'distant_2', 'recent_2', 'distant_3', 'recent_3'] +
        ['recent', 'distant', 'rare', 'frequent']}
    rare_freq_sessions = np.intersect1d(
        *[A_by_cond[cond].index.get_level_values('sess').values for cond in ['recent', 'distant']])
    rare_freq_sessions = [sess for sess in rare_freq_sessions if sess in drug_sess_dict['none']
                          and all([A_by_cond[cond].xs(sess, level='sess').shape[0] > 6
                                   for cond in ['recent', 'distant']])]
    # rare_freq_figdir = norm_dev_figdir.parent / f'rare_freq_{"_".join(cohort_tags)}'
    # if not rare_freq_figdir.is_dir():
    #     rare_freq_figdir.mkdir()
    rare_freq_plot = plot_pupil_ts_by_cond(A_by_cond, ['frequent', 'rare', ], sess_list=rare_freq_sessions,
                                           group_name='sess', plot_indv_sess=False,
                                           cond_line_kwargs=rare_freq_cond_line_kwargs)
    rare_freq_plot[1].locator_params(axis='both', nbins=4)
    rare_freq_plot[1].set_title('')
    [rare_freq_plot[1].axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
    rare_freq_plot[1].axhline(0, c='k', ls='--')
    rare_freq_plot[0].set_layout_engine('tight')
    rare_freq_plot[0].show()

    # plot rare vs frequent by drug
    drugs = ['none','saline','muscimol']
    assert all([drug in drug_sess_dict for drug in drugs])
    all_drugs_rare_freq_plot = plt.subplots(ncols=len(drugs), sharey=True,figsize=(4*len(drugs),4))
    for drug,ax in zip(['none','saline','muscimol'],all_drugs_rare_freq_plot[1]):
        rare_freq_sessions = np.intersect1d(
            *[A_by_cond[cond].index.get_level_values('sess').values for cond in ['recent', 'distant']])
        rare_freq_drug_sessions = [sess for sess in rare_freq_sessions if sess in drug_sess_dict[drug]
                                   and all([A_by_cond[cond].xs(sess, level='sess').shape[0] > 6
                                            and 'DO91' not in sess
                                            for cond in ['recent', 'distant']])]
        plot_pupil_ts_by_cond(A_by_cond, ['frequent', 'rare', ], sess_list=rare_freq_drug_sessions,
                              group_name='sess', plot_indv_sess=False,
                              cond_line_kwargs=rare_freq_cond_line_kwargs,
                              plot=(all_drugs_rare_freq_plot[0], ax))
        ax.locator_params(axis='both', nbins=4)
        ax.set_title(drug)
        [ax.axvspan(t, t + 0.15, fc='grey', alpha=0.1) for t in np.arange(0, 1, 0.25)]
        ax.axhline(0, c='k', ls='--')
    all_drugs_rare_freq_plot[0].set_layout_engine('tight')
    all_drugs_rare_freq_plot[0].show()

    # plot max diff over window by drug
    all_drugs_max_diff_plot = plt.subplots(ncols=len(drugs), sharey=True,figsize=(4*len(drugs),4))
    for drug,ax in zip(['none','saline','muscimol'],all_drugs_max_diff_plot[1]):
        max_diff_sessions = np.intersect1d(
            *[X_by_cond[cond].index.get_level_values('sess').values for cond in ['recent', 'distant']])
        max_diff_drug_sessions = [sess for sess in max_diff_sessions if sess in drug_sess_dict[drug]
                                  and all([X_by_cond[cond].xs(sess, level='sess').shape[0] > 6
                                           and 'DO91' not in sess
                                           for cond in ['recent', 'distant']])]
        # plot max diff over window
        plot_pupil_diff_max_by_cond(A_by_cond, ['frequent', 'rare',],sess_list=max_diff_drug_sessions,
                                    window_by_stim=[(1.5, 2.5)][0],
                                    mean=np.max,plot_kwargs={'showfliers': False,
                                                 'showmeans': False,
                                                 'labels': ['data'],
                                                 'widths': 0.3,
                                                 },
                                    positions=[0,1],
                                    permutation_test=True, n_permutations=50,
                                    plot=(all_drugs_max_diff_plot[0], ax),
                                    group_name='name')

        ax.set_ylabel('delta pupil')
        ax.set_title(drug)
        ax.axhline(0, c='k', ls='--')
        ax.locator_params(axis='y', nbins=4)
    all_drugs_max_diff_plot[0].set_layout_engine('tight')
    # all_drugs_max_diff_plot[0].set_size_inches(2.7, 2.7)
    all_drugs_max_diff_plot[0].show()
        # # rare_freq_max_diff_plot[0].savefig(rare_freq_figdir / 'rare_freq_max_diff_plot_w_shuffle.svg')
        #
        # # ttest
        # rare_freq_max_diff_ttest = ttest_ind(rare_freq_diff_data[0][0].values, rare_freq_diff_data[1][0].values,
        #                                      alternative='greater')
        # # rare_freq_max_diff_ttest = ttest_1samp(rare_freq_diff_data[0], 0.0, alternative='greater')
        # print(rare_freq_max_diff_ttest)

