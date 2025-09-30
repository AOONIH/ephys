import logging
import platform
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml
from cycler import cycler
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from aggregate_psth_analysis import AggregateSession
from aggregate_ephys_funcs import parse_args
from io_utils import posix_from_win
from save_utils import save_stats_to_tex

if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
    logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

    # load config
    config_path = Path(args.config_file)
    if config_path.is_file():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded config from {config_path}")
    else:
        logging.warning(f"Config {config_path} not found. Continuing without.")

    ceph_dir = config['ceph_dir_' + platform.system().lower()]

    # Load plot config if provided
    plot_config = {}
    if args.plot_config_path:
        plot_path = Path(args.plot_config_path)
        if plot_path.is_file():
            with open(plot_path, 'r') as f:
                plot_config = yaml.safe_load(f)
            logging.info(f"Loaded plot config from {plot_path}")
        else:
            logging.warning(f"Plot config {plot_path} not found. Continuing without.")

    pkl_dir = Path(args.event_responses_pkl).parent

    # Set figure directories
    psth_figdir = ceph_dir / posix_from_win(plot_config.get('psth_figdir', r'X:\Dammy\figures\psth_analysis'))

    decoding_figdir = ceph_dir / posix_from_win(plot_config.get('decoding_figdir',
                                                                r'X:\Dammy\figures\rare_freq_decoding'))
    pca_fidir = ceph_dir / posix_from_win(plot_config.get('pca_figdir',
                                                          r'X:\Dammy\figures\pca_plots'))
    aggr_savedir = ceph_dir / posix_from_win(plot_config.get('aggr_savedir',
                                                             r'X:\Dammy\aggr_savedir'))
    for fig_dir in [decoding_figdir, pca_fidir, psth_figdir, aggr_savedir]:
        if not fig_dir.is_dir():
            fig_dir.mkdir(parents=False)

    plt.style.use('figure_stylesheet.mplstyle')
    # plt.rcParams['axes.prop_cycle'] = cycler(color=plot_config['colors'])
    cmap = plt.get_cmap('Blues_r')  # or any other colormap
    n_colors = 5
    # colors = ['k']+[cmap(i / (n_colors - 1)) for i in range(n_colors)]
    colors = ['k']+['#43007fff','#6127a5ff','#2786b9ff','#61aad1ff','#b3e1ffff']
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

    rare_freq_resps = AggregateSession(pkl_dir,args, plot_config,['A-0','A-0','A-0'])
    # rare_freq_resps.aggregate_mean_sess_responses(conds=['rare','frequent'],filt_by_prc=False, prc_thr=0.5,
    #                                               prc_pips=['A-0_rare','A-0_frequent'],prc_mutual=True,
    #                                               concat_savename=aggr_savedir/ 'A-0_rare_A-0_frequent.joblib',
    #                                               reload_save=True)
    # rare_freq_resps.plot_mean_ts(psth_figdir,**plot_config['rare_freq_mean_ts'])
    # rare_freq_resps.scatter_unit_means(['A-0_rare','A-0_frequent'],[0,1],psth_figdir)
    # #
    # rare_freq_resps.aggregate_sess_decoding('rare_freq_decoding',['rare','frequent'],
    #                                         df_save_path=decoding_figdir/'rare_freq_decoding_df.h5')
    # #
    # rare_freq_resps.plot_decoder_boxplot(decoding_figdir)
    # for pips2decode in plot_config['rare_freq_decoding']['pips2decode']:
    #     rare_freq_resps.decoding_ttest('_vs_'.join(pips2decode), 'data', 'shuffled')
    # for ttest_name, ttest_res in rare_freq_resps.ttest_res.items():
    #     save_stats_to_tex(ttest_res,decoding_figdir / f'{ttest_name}.tex')

    # rare freq pca
    # assert False
    conds = ['rare']+ [f'frequent_prate_{ti}' for ti in np.arange(0,30,5)][:5]
    # conds = ['rare_prate','frequent_prate_late','frequent_prate_early'],
    rare_freq_resps = AggregateSession(pkl_dir,args, plot_config,['A-0']*len(conds))
    rare_freq_resps.aggregate_mean_sess_responses(conds=conds,
                                                  concat_savename=aggr_savedir/ 'A-0_rare_A-0_frequent_sliding_window_no_zscore.joblib',
                                                  reload_save=False,
                                                  zscore_flag=True)
    rare_freq_resps.pca_pseudo_pop(pca_name='rare_freq_early_late_pca',standardise=False)

    for pca_combs in plot_config['rare_freq_early_late_pca']['pcas2plot']:
        rare_freq_resps.plot_3d_pca('rare_freq_early_late_pca',pca_combs, pca_fidir,
                                    plot_config['rare_freq_early_late_pca'])

    # for pca_combs in combinations(list(range(4)),2):
    for pca_combs in [[0,3],[1,3]]:
        rare_freq_resps.plot_2d_pca('rare_freq_early_late_pca', pca_combs[::-1], pca_fidir,
                                    plot_config['rare_freq_early_late_pca'])

    # for pca_combs in plot_config['rare_freq_early_late_pca']['pcas2scatter']:
    #     rare_freq_resps.scatter_pca('rare_freq_early_late_pca',[0,1],pca_combs, pca_fidir,
    #                                 plot_config['rare_freq_early_late_pca'])
