from pupil_analysis_funcs import *
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

    pupil_pkl_paths = [ceph_dir / posix_from_win(p) for p in [
        r'X:\Dammy\mouse_pupillometry\pickles\aligned_normdev_human_normdev_w_pupilsense_normdev_2d_90Hz_hpass00_lpass0_TOM.pkl',
    ]]
    all_pupil_data = [load_pupil_data(pkl_path) for pkl_path in pupil_pkl_paths]
    pupil_data_dict = all_pupil_data[0]

    all_human_td_files = list(Path(r'H:\data\Hilde\Human\TrialData').iterdir())
    all_sess_info_dict = {}
    for sess in pupil_data_dict:
        name, date = sess.split('_')
        date = int(date)
        sess_tdfile = [td for td in all_human_td_files if name in td.stem and str(date) in td.stem]
        if not sess_tdfile:
            continue
        sess_tdfile = sess_tdfile[0]
        all_sess_info_dict[sess] = {
            'name': name,
            'date': date,
            'tdata_file': sess_tdfile,
            'sess_order': 'main',
        }
    all_sess_info = pd.DataFrame(all_sess_info_dict).T.reset_index(drop=True)

    sess_pkl_path = Path(r'D:') / 'human_fam_sess_dicts_no_filt.pkl'
    # sessions = load_pupil_sess_pkl(sess_pkl_path)
    sessions = {}
    loaded_sess_dict_sig = copy(tuple(sorted(sessions.items())))

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

    cond_filters = get_all_cond_filts()
    # Optionally add a good trial filter here

    # Group by normal and deviant
    A_by_cond = {cond: group_pupil_across_sessions(
        sessions, list(sessions.keys()), 'A', cond, cond_filters)
        for cond in ['normal_exp_human', 'deviant_C_human']}
    A_by_cond['normal'] = A_by_cond['normal_exp_human']
    A_by_cond['deviant_C'] = A_by_cond['deviant_C_human']

    for cond in A_by_cond:
        A_by_cond[cond] = A_by_cond[cond].T.rolling(window=25).mean().T

    # Find sessions with both normal and deviant_C data
    norm_dev_sessions = np.intersect1d(
        A_by_cond['normal'].index.get_level_values('sess').values,
        A_by_cond['deviant_C'].index.get_level_values('sess').values
    )
    norm_dev_sessions = [sess for sess in norm_dev_sessions
                         if all(A_by_cond[cond].xs(sess, level='sess').shape[0] > 4 for cond in ['normal', 'deviant_C'])
                         and 'Human32' not in sess]  # remove session with bad sync

    # Output directory
    norm_dev_figdir = ceph_dir / 'Dammy' / 'figures' / 'norm_dev_human_no_filt'
    if not norm_dev_figdir.is_dir():
        norm_dev_figdir.mkdir(parents=True)

    # Plot time series
    normdev_line_kwargs = {
        'normal': {'c': 'k', 'ls': '-'},
        'deviant_C': {'c': '#cd2727ff', 'ls': '-'}
    }
    norm_dev_plot = plot_pupil_ts_by_cond(
        A_by_cond, ['normal', 'deviant_C'], sess_list=norm_dev_sessions,
        cond_line_kwargs=normdev_line_kwargs
    )
    norm_dev_plot[1].set_title('')
    norm_dev_plot[1].axvline(0, c='k', ls='--')
    norm_dev_plot[1].locator_params(axis='both', nbins=4)
    norm_dev_plot[0].set_layout_engine('tight')
    norm_dev_plot[0].set_size_inches(3.6, 2.7)
    format_axis(norm_dev_plot[1],vspan=[[t, t+0.15] for t in np.arange(0, 1, 0.25) if t != 0.5])
    norm_dev_plot[1].axvspan(0.5, 0.65, color='r', alpha=0.1)
    norm_dev_plot[0].show()
    norm_dev_plot[0].savefig(norm_dev_figdir / 'norm_dev_plot.pdf')

    # Plot difference time series
    norm_dev_diff_plot = plot_pupil_diff_ts_by_cond(
        A_by_cond, ['deviant_C', 'normal'], sess_list=norm_dev_sessions
    )
    norm_dev_diff_plot[1].set_title('')
    norm_dev_diff_plot[1].set_ylabel('')
    norm_dev_diff_plot[1].set_xlabel('')
    norm_dev_diff_plot[1].axvline(0, c='k', ls='--')
    norm_dev_diff_plot[1].axhline(0, c='k', ls='--')
    norm_dev_diff_plot[1].locator_params(axis='both', nbins=4)
    norm_dev_diff_plot[0].set_layout_engine('tight')
    norm_dev_diff_plot[0].set_size_inches(3.6, 2.7)
    format_axis(norm_dev_diff_plot[1],vspan=[[t, t+0.15] for t in np.arange(0, 1, 0.25) if t != 0.5])
    norm_dev_diff_plot[1].axvspan(0.5, 0.65, color='r', alpha=0.1)
    norm_dev_diff_plot[0].show()
    norm_dev_diff_plot[0].savefig(norm_dev_figdir / 'norm_dev_diff_plot.pdf')

    # Plot max diff over window
    norm_dev_max_diff_plot, normdev_diff_data = plot_pupil_diff_max_by_cond(
        A_by_cond, ['normal', 'deviant_C'], sess_list=norm_dev_sessions,
        window_by_stim=(1.5, 2.5), mean=np.max,
        plot_kwargs={'showfliers': False, 'labels': ['dev'], 'widths': 0.3},
        group_name='name', permutation_test=True, n_permutations=10,
    )

    norm_dev_max_diff_plot[1].set_ylabel('')
    norm_dev_max_diff_plot[1].set_title('')
    norm_dev_max_diff_plot[1].axhline(0, c='k', ls='--')
    norm_dev_max_diff_plot[1].locator_params(axis='y', nbins=4)
    norm_dev_max_diff_plot[0].set_layout_engine('tight')
    norm_dev_max_diff_plot[0].set_size_inches(3.7, 2.7)
    norm_dev_max_diff_plot[0].show()
    norm_dev_max_diff_plot[0].savefig(norm_dev_figdir / 'norm_dev_max_diff_plot.pdf')

    # # ttest
    # ttest = ttest_ind(normdev_diff_data[0][0].values, normdev_diff_data[1][0].values, alternative='greater')
    # print("Max diff t-test (normal > deviant_C):", ttest)
    # save_stats_to_tex(ttest,stats_dir/'human_norm_dev_max_diff_ttest.tex')

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
        sess_list=norm_dev_sessions,
        conditions=['normal', 'deviant_C'][::-1],
        figdir=norm_dev_figdir,
        line_kwargs=normdev_line_kwargs,
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
        # ylim_ts=(-0.05, 0.35),
        # ylim_maxdiff=(-0.25, 0.25)
        event_name='pattern',)
