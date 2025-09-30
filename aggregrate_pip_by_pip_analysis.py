from copy import copy

import matplotlib
import numpy as np
import pandas as pd
import yaml
import logging
import platform
import pickle
from pathlib import Path

from matplotlib import pyplot as plt
from scipy.stats import sem, ttest_ind, ttest_1samp
from tqdm import tqdm
import joblib
from aggregate_ephys_funcs import get_responses_by_pip_and_condition, run_decoding, parse_args
from population_analysis_funcs import PopPCA
from aggregate_psth_analysis import AggregateSession
from behviour_analysis_funcs import get_all_cond_filts
from io_utils import posix_from_win
from plot_funcs import plot_sorted_psth, format_axis, add_x_scale_bar, get_sorted_psth_matrix, plot_sorted_psth_matrix
from save_utils import save_stats_to_tex
from unit_analysis import UnitAnalysis


def main():
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
    for fig_dir in [decoding_figdir, pca_fidir, psth_figdir]:
        if not fig_dir.is_dir():
            fig_dir.mkdir(parents=False)
    plt.subplots()

    plt.style.use('figure_stylesheet.mplstyle')

    pip_resps = AggregateSession(pkl_dir,args, plot_config,plot_config['pips_2_plot'])
    pip_resps.aggregate_sess_decoding('all_v_all_decoding',
                                      df_save_path=decoding_figdir/'all_v_all_decoding_df.h5')
    pip_resps.plot_confusion_matrix(decoding_figdir,plot_config['all_v_all_cm_kwargs'],)


if __name__ == '__main__':
    main()