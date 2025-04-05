import pickle
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import ttest_ind, ttest_1samp, f_oneway, tukey_hsd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

from aggregate_ephys_funcs import *
from ephys_analysis_funcs import get_pip_desc, format_axis, plot_sorted_psth
from neural_similarity_funcs import get_list_pips_by_property, plot_similarity_mat, compute_self_similarity, \
    compare_pip_sims_2way, plot_sim_by_grouping
from population_analysis_funcs import compute_mi
# from npeet import entropy_estimators as ee
# from pyentrp import entropy as ent


def initialize_directories(base_dir, sub_dirs):
    """
    Initialize directories for saving figures and data.

    Parameters:
        base_dir (Path): Base directory path.
        sub_dirs (list): List of sub-directory names.

    Returns:
        dict: Dictionary of initialized directories.
    """
    directories = {}
    for sub_dir in sub_dirs:
        dir_path = base_dir / sub_dir
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
        directories[sub_dir] = dir_path
    return directories


def plot_event_responses(event_responses, title='', xlabel='', ylabel='', save_path=None):
    """
    Plot event responses with optional saving.

    Parameters:
        event_responses (dict): Dictionary of event responses.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        save_path (Path, optional): Path to save the plot.
    """
    plt.figure()
    for event, response in event_responses.items():
        plt.plot(response, label=event)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('pkldir')
    args = parser.parse_args()
    sys_os = platform.system().lower()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    home_dir = Path(config[f'home_dir_{sys_os}'])
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])

    session_topology_path = ceph_dir / posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology.csv')
    session_topology = pd.read_csv(session_topology_path)

    # Directly filter sessions without using a separate function
    all_sess_info = session_topology.query('sess_order=="main" ')
    all_sess_td_df = load_aggregate_td_df(all_sess_info, home_dir, 'Stage==5')
    sessions2use = all_sess_td_df.index.get_level_values('sess').unique()

    hipp_animals = ['DO79', 'DO81']
    animals = session_topology['name'].unique().tolist()

    # Initialize directories
    directories = initialize_directories(ceph_dir / 'Dammy', ['figures/pca_aggregate_sessions_all_new_abstraction_new',
                                                              'figures/aggregate_abstraction_sessions'])

    pkldir = ceph_dir / posix_from_win(args.pkldir)
    pkls = list(pkldir.glob('*pkl'))
    pkls2load = [pkl for pkl in pkls if Path(pkl).stem in sessions2use]
    sessions = load_aggregate_sessions(pkls2load, 'Stage==5')

    [sessions.pop(sess) for sess in list(sessions.keys()) if 'A-0' not in list(sessions[sess].sound_event_dict.keys())]
    [sessions.pop(sess) for sess in list(sessions.keys()) if 'A-3' not in list(sessions[sess].sound_event_dict.keys())]
    [sessions.pop(sess) for sess in list(sessions.keys()) if 'D-3' not in list(sessions[sess].sound_event_dict.keys())]
    [sessions.pop(sess) for sess in list(sessions.keys()) if 'A-4' in list(sessions[sess].sound_event_dict.keys())]
    [print(sess, sessions[sess].sound_event_dict.keys()) for sess in list(sessions.keys())]

    # Update styles
    plt.style.use('figure_stylesheet.mplstyle')

    full_pattern_responses = aggregate_event_reponses(sessions, events=None,
                                                      events2exclude=['trial_start'], window=[-0.25, 1],
                                                      pred_from_psth_kwargs={'use_unit_zscore': True,
                                                                             'use_iti_zscore': False,
                                                                             'baseline': 0, 'mean': None,
                                                                             'mean_axis': 0})

    # Dump full pattern responses
    full_patt_dict_path = ceph_dir / Path(posix_from_win(r'X:\for_will')) / 'full_patt_dict_ABCD_vs_ABBA.pkl'
    if not full_patt_dict_path.is_file():
        with open(full_patt_dict_path, 'wb') as f:
            pickle.dump(full_pattern_responses, f)

    window = (-0.1, 0.25)
    event_responses = aggregate_event_reponses(sessions,
                                               events2exclude=['trial_start'], window=window,
                                               pred_from_psth_kwargs={'use_unit_zscore': True, 'use_iti_zscore': False,
                                                                      'baseline': 0, 'mean': None, 'mean_axis': 0})

    event_features = aggregate_event_features(sessions,
                                              events2exclude=['trial_start'])

    concatenated_event_responses = {
        e: np.concatenate([event_responses[sessname][e].mean(axis=0) for sessname in event_responses])
        for e in list(event_responses.values())[0].keys()}
    concatenated_event_times = {
        e: np.concatenate([(event_features[sessname][e]['times']) for sessname in event_features])
        for e in list(event_features.values())[0].keys()}

    n_units = concatenated_event_responses[list(concatenated_event_responses.keys())[0]].shape[0]
    cond_filts = get_all_cond_filts()
    common_events = sorted(set.intersection(*[set(list(e.keys())) for e in event_responses.values()]))
    n_per_rule = 2
    events_by_property = {
        'position': {pip: ord(pip.split('-')[0]) - ord('A') + 1 for pip in common_events
                                       if pip.split('-')[0] in 'ABCD'},
        'group': {pip: int(int(pip.split('-')[1]) / n_per_rule) for pip in common_events
                  if pip.split('-')[0] in 'ABCD'},
        'ptype_i': {pip: 0 if int(pip.split('-')[1]) % n_per_rule == 0 else 1 for pip in common_events
                    if pip.split('-')[0] in 'ABCD'},
        'name': {pip: pip for pip in common_events
                 if pip.split('-')[0] in 'ABCD'},
        'pattern_i': {pip: pip.split('-')[1] for pip in common_events
                      if pip.split('-')[0] in 'ABCD'},
    }
    pip_desc = sessions[list(sessions.keys())[1]].pip_desc
    events_by_property['id'] = {pip: pip_desc[pip]['idx']
                                for pip in common_events
                                if pip.split('-')[0] in 'ABCD'}

    concatenated_events_by_pip_prop = group_responses_by_pip_prop(concatenated_event_responses, events_by_property)

    # Plot mean responses to rare and frequent pips
    for prop, prop_val in concatenated_events_by_pip_prop.items():
        property_response_plot = plt.subplots()
        for cond_i, (cond, ls) in enumerate(zip(['ptype_i'], ['-', '--'])):
            for pip_i, pip in enumerate(sorted(concatenated_events_by_pip_prop[cond])):
                x_ser = np.linspace(*window, len(concatenated_events_by_pip_prop[cond][pip].mean(axis=0)))
                property_response_plot[1].plot(x_ser, concatenated_events_by_pip_prop[cond][pip].mean(axis=0),
                                               c=f'C{pip_i}', label=f'{cond} {pip}', ls=ls)
        property_response_plot[1].legend()
        property_response_plot[1].set_title(f'Population responses to rare and frequent pips')
        property_response_plot[1].set_xlabel('Time from sound onset (s)')
        property_response_plot[1].axvline(0, color='k', linestyle='--')
        property_response_plot[1].set_ylabel('Firing rate')
        property_response_plot[0].set_layout_engine('tight')
        property_response_plot[0].show()

    # Example usage of the new plotting function
    example_event_responses = {
        'event1': np.random.rand(100),
        'event2': np.random.rand(100)
    }
    plot_event_responses(example_event_responses, title='Example Event Responses', xlabel='Time', ylabel='Response',
                         save_path=directories['figures/pca_aggregate_sessions_all_new_abstraction_new'] / 'example_event_responses.png')