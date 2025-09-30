import argparse
import platform
from pathlib import Path

import matplotlib
import numpy as np
import yaml
from matplotlib import pyplot as plt

# from aggregate_ephys_funcs import *
from aggregate_ephys_funcs import load_or_generate_event_responses, run_decoding
from aggregrate_ephys_fam import get_responses_by_pip_and_condition
from behviour_analysis_funcs import get_all_cond_filts
from io_utils import posix_from_win
from plot_funcs import format_axis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('pkldir')
    parser.add_argument('event_responses_pkl')
    parser.add_argument('--plot_config_path', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    # Load config
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    sys_os = platform.system().lower()
    home_dir = Path(config[f'home_dir_{sys_os}'])
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])

    # Load plot config if provided
    plot_config = {}
    if args.plot_config_path is not None:
        plot_config_path = Path(args.plot_config_path)
        if plot_config_path.is_file():
            with open(plot_config_path, 'r') as f:
                plot_config = yaml.safe_load(f)

    event_responses, event_features = load_or_generate_event_responses(args, plot_config)

    cond_filts = get_all_cond_filts()

    event_responses_rare_freq = get_responses_by_pip_and_condition(['A-0'],event_responses,event_features,
                                                                   ['rare','frequent'],cond_filts)

    # Concatenate event responses for plotting
    mutual_sessions = [sess for sess in list(event_responses_rare_freq.values())[0].keys()
                       if sess in list(event_responses_rare_freq.values())[1].keys()]

    concatenated_event_responses = {
        e: np.concatenate([event_responses_rare_freq[e][sessname].mean(axis=0) for sessname in mutual_sessions])
        for e in event_responses_rare_freq
    }

    # --- Plot psth ts --
    decoding_figdir = ceph_dir / posix_from_win(
        plot_config.get('psth_figdir', r'X:\Dammy\figures\rare_freq_decoding'))
    if not decoding_figdir.is_dir():
        decoding_figdir.mkdir(parents=True, exist_ok=True)

    animals = plot_config.get('animals', list({sess.split('_')[0] for sess in event_responses.keys()}))
    sess2use = [sess for sess in event_responses.keys() if any(a in sess for a in animals)]
    window = plot_config.get('window', [-0.25, 1])
    x_ser = np.round(np.arange(window[0], window[1] + 0.01, 0.01), 2)

    psth_ts_plot = plt.subplots()
    [psth_ts_plot[1].plot(x_ser, np.nanmean(concatenated_event_responses[e], axis=0))
     for e in event_responses_rare_freq]
    format_axis(psth_ts_plot[1])
    psth_ts_plot[0].show()

    # --- Run decoding and plot ---
    decoding_figdir = ceph_dir / posix_from_win(
        plot_config.get('psth_figdir', r'X:\Dammy\figures\rare_freq_decoding'))
    if not decoding_figdir.is_dir():
        decoding_figdir.mkdir(parents=True, exist_ok=True)

    animals = plot_config.get('animals', list({sess.split('_')[0] for sess in event_responses.keys()}))
    sess2use = [sess for sess in event_responses.keys() if any(a in sess for a in animals)]
    window = plot_config.get('window', [-0.25, 1])
    x_ser = np.round(np.arange(window[0], window[1] + 0.01, 0.01), 2)
    decoding_window = plot_config.get('decoding_window', [0.1, 0.25])
    pips = plot_config.get('pips2decode', ['A-0', 'B-0', 'C-0', 'D-0'])

    decode_cache_path = decoding_figdir / 'rare_freq_decode_results_cache.pkl'
    decode_dfs,_ = run_decoding(
        event_responses, animals, x_ser, decoding_window, pips, cache_path=decode_cache_path, overwrite=1
    )

    # --- Boxplot plotting kwargs ---
    boxplot_kwargs = dict(
        widths=0.6,
        patch_artist=True,
        showmeans=False,
        medianprops=dict(mfc='k')
    )

    # Plot all decoding results on one figure
    stim_decoding_plot = plt.subplots()
    labels = []
    scatter_points = False  # Set to True to scatter individual points

    dec_sffx = '_vs_'.join(pips)
    data_acc = decode_dfs[f'{dec_sffx}_data_accuracy'].values
    shuff_acc = decode_dfs[f'{dec_sffx}_shuff_accuracy'].values
    box = stim_decoding_plot[1].boxplot(
        [data_acc, shuff_acc],
        labels=[f'{dec_sffx}\ndata', f'{dec_sffx}\nshuffle'],
        **boxplot_kwargs
    )
    # change box colors
    for patch in box['boxes']:
        patch.set_facecolor('white')

    labels.extend([f'{dec_sffx}\ndata', f'{dec_sffx}\nshuffle'])

    stim_decoding_plot[1].set_ylabel('Accuracy')
    stim_decoding_plot[0].tight_layout()
    stim_decoding_plot[0].show()
    stim_decoding_plot[0].savefig(decoding_figdir / f'{dec_sffx}_decoding_accuracy.pdf')
