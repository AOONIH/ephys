import argparse
import pickle
import platform
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import ttest_ind, ttest_1samp, sem, tukey_hsd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics.pairwise import cosine_similarity


from aggregate_ephys_funcs import *
from behviour_analysis_funcs import get_all_cond_filts, get_n_since_last, get_prate_block_num, get_cumsum_columns, \
    add_datetimecol, get_lick_in_patt_trials, get_earlyX_trials, get_last_pattern
from io_utils import posix_from_win
from plot_funcs import plot_2d_array_with_subplots, plot_psth, plot_sorted_psth, format_axis
from neural_similarity_funcs import plot_similarity_mat
from regression_funcs import run_glm

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('analysis_config_file')
    parser.add_argument('--figdir_sffx', default='',
                        help='suffix to add to figure directory name')
    args = parser.parse_args()

    sys_os = platform.system().lower()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    with open(args.analysis_config_file, 'r') as file:
        analysis_config = yaml.safe_load(file)

    home_dir = Path(config[f'home_dir_{sys_os}'])
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])

    session_topology_path = ceph_dir / posix_from_win(analysis_config['session_topology_path'])
    if not session_topology_path.is_file():
        raise FileNotFoundError(f"Session topology file not found at {session_topology_path}")
    session_topology = pd.read_csv(session_topology_path)
    session_topology = session_topology.query('sess_order=="main"')

    sessions2use = analysis_config['sessnames']
    if not isinstance(sessions2use, list):
        raise ValueError("sessions2use should be a list of session names.")
    if not all(isinstance(sess, str) for sess in sessions2use):
        raise ValueError("All session names in sessions2use should be strings.")
    
    sess_name_date_split = [sess.split('_') for sess in sessions2use]
    for sess in sess_name_date_split:
        sess[1] = int(sess[1])
    all_sess_info = pd.concat([session_topology.query(f'name=="{sess[0]}" & date=={sess[1]}') for sess in sess_name_date_split])

    # use the sound_bin value to get the suffix for each session
    sound_bin_suffix = [Path(e).stem[-1] for e in all_sess_info['sound_bin']]
    sessnames = [f'{sess}{sffx}' for sess, sffx in zip(sessions2use, sound_bin_suffix)]
    animals = np.unique([sess.split('_')[0] for sess in sessnames]).tolist()

    all_sess_td_df = load_aggregate_td_df(all_sess_info,home_dir,'Stage==3')

    pkldir = ceph_dir / posix_from_win(analysis_config['pkldir'])
    if not pkldir.is_dir():
        raise FileNotFoundError(f"PKL directory not found at {pkldir}")
    pkls = [Path(pkldir) / f'{sess}.pkl' for sess in sessnames]
    assert all([pkl.is_file() for pkl in pkls]), f"Not all PKL files found in {pkldir}"

    sess_dict_pkl_path = Path(analysis_config['sess_dict_pkl_path'])
    ow_flag = False
    if sess_dict_pkl_path.is_file() and not ow_flag:
        with open(sess_dict_pkl_path, 'rb') as f:
            sessions = pickle.load(f)
    else:
        print('loading sessions from previous pkl...')
        sessions = load_aggregate_sessions(pkls,)

        with open(sess_dict_pkl_path, 'wb') as f:
            pickle.dump(sessions, f)
            

    # [sess for sess in sessions2use if not any([sess in Path(pkl).stem for pkl in pkls2load])]

    [sessions.pop(sess) for sess in list(sessions.keys()) if 'A-0' not in list(sessions[sess].sound_event_dict.keys())]
    # [sessions.pop(sess) for sess in list(sessions.keys()) if 'D-1' not in list(sessions[sess].sound_event_dict.keys())]
    # [sessions.pop(sess) for sess in list(sessions.keys()) if 'A-2' in list(sessions[sess].sound_event_dict.keys())]
    [print(sess,sessions[sess].sound_event_dict.keys()) for sess in list(sessions.keys()) ]

    # update styles
    plt.style.use('figure_stylesheet.mplstyle')
    matplotlib.rcParams['figure.figsize'] = (4,3)

    aggr_figdir = ceph_dir / 'Dammy' / 'figures' / 'decoding_acc_across_sessions'
    if not aggr_figdir.is_dir():
        aggr_figdir.mkdir()

    window = (-0.1, 0.25)
    hipp_animals = np.unique([sess.split('_')[0] for sess in sessions.keys() if 'Hipp' in sess])
    event_responses = aggregate_event_responses(sessions, events=None,  # [f'{pip}-0' for pip in 'ABCD']
                                                events2exclude=['trial_start',], window=window,
                                                pred_from_psth_kwargs={'use_unit_zscore': True, 'use_iti_zscore': False,
                                                                      'baseline': 0, 'mean': None, 'mean_axis': 0})
    concatenated_event_responses = {
        e: np.concatenate([event_responses[sessname][e].mean(axis=0) for sessname in event_responses])
        for e in list(event_responses.values())[0].keys()}

    cond_filters = get_all_cond_filts()

    # decode across each session
    decoder_dict = {}

    # Define mapping of pips to integers for decoding
    pips_as_ints = {pip: pip_i for pip_i, pip in enumerate([f'{pip}-0' for pip in 'ABCD'])}

    # Initialize a set to track sessions that fail decoding
    bad_dec_sess = set()

    # Iterate over each session in event_responses
    for sessname in tqdm(event_responses.keys(), desc='Running all-vs-all decoder for each session'):
        try:
            # Prepare the data (X) and labels (Y) for decoding
            xys = [(event_responses[sessname][pip], 
                    np.full_like(event_responses[sessname][pip][:, 0, 0], pips_as_ints[pip]))
                   for pip in [f'{p}-0' for p in 'ABCD']]

            # Stack the data and labels
            xs = np.vstack([xy[0][:, :, -15:-5].mean(axis=-1) for xy in xys])
            ys = np.hstack([xy[1] for xy in xys])

            # Run the decoder and store the result in decoder_dict
            decoder_dict[f'{sessname}-allvall'] = decode_responses(xs, ys, dec_kwargs=dict(cv_folds=5))
        except ValueError:
            # Handle decoding failure
            print(f'{sessname}-allvall decoding failed')
            bad_dec_sess.add(sessname)
            continue

    [decoder_dict[dec_name]['data'].plot_confusion_matrix([f'{p}-{0}' for p in 'ABCD'], )
     for dec_name in decoder_dict.keys()]
    [decoder_dict[dec_name]['shuffled'].plot_confusion_matrix([f'{p}-{0}' for p in 'ABCD'], )
     for dec_name in decoder_dict.keys()]

    # Create a figure with subplots for confusion matrices
    all_sess_dec_cms = plt.subplots(ncols=4, nrows=2, figsize=(12,6))

    # Ensure axes is iterable even if there's only one subplot
    if len(decoder_dict) == 1:
        axes = [all_sess_dec_cms[1]]
    else:
        axes = all_sess_dec_cms[1].flatten()

    # Iterate over sessions and plot confusion matrices
    cmap_norm = TwoSlopeNorm(vmin=0, vmax=0.6,
                             vcenter=1 / len(pips_as_ints))
    for ax, (sessname, decoder) in zip(axes, decoder_dict.items()):
        if 'data' in decoder and hasattr(decoder['data'], 'cm'):
            ConfusionMatrixDisplay(decoder['data'].cm, display_labels=[f'{p}-0' for p in 'ABCD']).plot(ax=ax,
                                                                                                       cmap='bwr',
                                                                                                       im_kw=dict(
                                                                                                           norm=cmap_norm),
                                                                                                       colorbar=False)
            ax.set_title(sessname)
    [cm_plot.invert_yaxis() for cm_plot in axes]
    [cm_plot.set_xlabel('') for cm_plot in axes]
    [cm_plot.set_ylabel('') for cm_plot in axes]

    # Adjust layout and show the plot
    all_sess_dec_cms[0].show()
    # Save the plot
    all_sess_dec_cms[0].savefig(aggr_figdir / 'decoding_cms_all_v_all_no_cbar.pdf')

    # format accuracy across sessions df
    all_cms_by_pip = np.array([dec['data'].cm for dec_name, dec in decoder_dict.items() if dec['data'].cm is not None])
    accuracy_across_sessions = np.diagonal(all_cms_by_pip,axis1=1, axis2=2)
    accuracy_across_sessions_df = pd.DataFrame(accuracy_across_sessions,
                                                index=list([sess for sess in sessions.keys()
                                                                      if sess not in bad_dec_sess]),)
    accuracy_across_sessions_df.columns = list('ABCD')
    # add name to multiindex
    accuracy_across_sessions_df['name'] = accuracy_across_sessions_df.index.map(lambda x: x.split('_')[0])
    accuracy_across_sessions_df = accuracy_across_sessions_df.set_index('name',append=True)

    # plot acc for each pip over sessions
    pip_acc_over_sess_plot = plt.subplots(len(animals),sharey=True)
    for ai, animal in enumerate(animals):
        animal_pip_accs = accuracy_across_sessions_df.xs(animal,level='name')
        for pi, pip in enumerate('ABCD'):
            pip_acc_over_sess_plot[1][ai].plot(animal_pip_accs[pip].values,label=f'{pip}',marker='o')
    # format
    [format_axis(ax,ylim=(0.1,0.7),hlines=[0.25],ylabel='decoding accuracy',xlabel='session #')
     for ax in pip_acc_over_sess_plot[1]]
    pip_acc_over_sess_plot[1][0].legend(ncols=accuracy_across_sessions_df.shape[1],loc='upper left')
    pip_acc_over_sess_plot[0].show()
    # save
    pip_acc_over_sess_plot[0].savefig(aggr_figdir / f'pip_acc_over_sessions.pdf')




