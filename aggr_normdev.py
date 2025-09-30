import logging
import platform
from pathlib import Path

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
    for fig_dir in [decoding_figdir, pca_fidir, psth_figdir,aggr_savedir]:
        if not fig_dir.is_dir():
            fig_dir.mkdir(parents=False)

    plt.style.use('figure_stylesheet.mplstyle')
    plt.rcParams['axes.prop_cycle'] = cycler(color=plot_config['colors'])

    normdev_resps = AggregateSession(pkl_dir,args, plot_config,['A-0','A-1'],)
    normdev_resps.aggregate_mean_sess_responses(  # conds=['normal', 'deviant_C'],
        filt_by_trial_num=True,
        pip_4_filt=['A-1'], pip_2_filt=['A-0'],
        filt_func=lambda tn, ft: [max(tn[tn < e] - e) + e for e in ft],
        # filt_func= lambda tn, ft: [e-1 for e in ft],
        concat_savename=aggr_savedir / 'A-0_A-1_1_trial_back.joblib',
        reload_save=True)
    # normdev_resps.aggregate_mean_sess_responses(conds=['normal','deviant_C'],
    #                                             concat_savename=aggr_savedir/'A-0_A-1.joblib',reload_save=False)
    # normdev_resps.plot_mean_ts(psth_figdir,**plot_config['normdev_mean_ts'])
    normdev_resps.aggregate_sess_decoding('normdev_decoding',filt_by_prc=False,
                                          df_save_path=decoding_figdir/'normdev_decoding_0.5_1.5_prev_normal_df.h5',
                                          reload_save=True)
    normdev_resps.plot_decoder_boxplot(decoding_figdir,)
    for pips2decode in plot_config['normdev_decoding']['pips2decode']:
        normdev_resps.decoding_ttest('_vs_'.join(pips2decode), 'data', 'shuffled')
    for ttest_name, ttest_res in normdev_resps.ttest_res.items():
        save_stats_to_tex(ttest_res,decoding_figdir / f'{ttest_name}.tex')
    #
    # normdev_resps.aggregate_mean_sess_responses(#conds=['normal', 'deviant_C'],
    #                                             filt_by_trial_num=True,
    #                                             pip_4_filt=['A-1'],pip_2_filt=['A-0'],
    #                                             filt_func= lambda tn, ft: [max(tn[tn < e] - e)+e for e in ft],
    #                                             # filt_func= lambda tn, ft: [e-1 for e in ft],
    #                                             concat_savename=aggr_savedir / 'A-0_A-1_1_trial_back.joblib',
    #                                             reload_save=False)
    normdev_resps.pca_pseudo_pop(pca_name='normdev_pca')
    for pca_combs in plot_config['normdev_pca']['pcas2plot']:
        normdev_resps.plot_3d_pca('normdev_pca', pca_combs, pca_fidir,
                                    plot_config['normdev_pca'])