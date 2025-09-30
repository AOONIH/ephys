import numpy as np
import pandas as pd
from pathlib import Path

from tqdm import tqdm
import json

# import scicomap as sc
from io_utils import posix_from_win


# import seaborn as sns
# from os import register_at_fork

# try:matplotlib.use('TkAgg')
# except ImportError:pass


def gen_metadata(data_topology_filepath: [str, Path], ceph_dir, col_name='sound_bin_stem',
                 harp_bin_dir=r'X:\Dammy\harpbins'):
    if not isinstance(data_topology_filepath, Path):
        data_topology_filepath = Path(data_topology_filepath)
    data_topology = pd.read_csv(data_topology_filepath)
    for idx, sess in data_topology.iterrows():
        try:
            e_dir = next((ceph_dir / posix_from_win(sess['ephys_dir'])).rglob('continuous')).parent
            bin_stem = sess[col_name]
            dir_files = list(e_dir.iterdir())
            if 'metadata.json' not in dir_files or True:
                if harp_bin_dir and isinstance(harp_bin_dir, Path):
                    harp_bin = harp_bin_dir / f'{bin_stem}_write_data.csv'
                else:
                    bin_path = Path(bin_stem)
                    harp_bin = bin_path.with_stem(f'{bin_path.stem}_write_data').with_suffix('.csv')
                # harp_bin_dir = Path(harp_bin_dir)
                harp_writes = pd.read_csv(harp_bin)
                trigger_time = harp_writes[harp_writes['DO3'] == True]['Times'].values[0]
                metadata = {'trigger_time': trigger_time}
                with open(e_dir / 'metadata.json', 'w') as jsonfile:
                    json.dump(metadata, jsonfile)
        except:
            continue


def gen_spike_matrix_old(spike_time_dict: dict, window, fs):
    precision = np.ceil(np.log10(fs)).astype(int)
    time_cols = np.round(np.arange(window[0], window[1] + 1 / fs, 1 / fs), precision)
    spike_matrix = pd.DataFrame(np.zeros((len(spike_time_dict), int((window[1] - window[0]) * fs) + 1)),
                                index=list(spike_time_dict.keys()), columns=time_cols)
    rounded_spike_dict = {}
    for cluster_id in tqdm(spike_time_dict, desc='rounding spike times for event', total=len(spike_time_dict),
                           disable=True):
        rounded_spike_dict[cluster_id] = np.round(spike_time_dict[cluster_id], precision)
    for cluster_id in tqdm(rounded_spike_dict, desc='assigning spikes to matrix', total=len(spike_time_dict),
                           disable=True):
        spike_matrix.loc[cluster_id][rounded_spike_dict[cluster_id]] = 1
    return spike_matrix


def get_patts_by_rule(patts):
    rules = [np.diff(patt,) for patt in patts]
    return {';'.join([str(e) for e in rule]): [patt for patt in patts if np.array_equal(np.diff(patt),rule)]
            for rule in rules}


def get_pip_desc(pip: str, pip_idx_dict: dict, pip_lbl_dict: dict, n_per_rule: int) -> [int, int, str, str, int,str]:
    position = ord(pip.split('-')[0]) - ord('A') + 1
    idx = pip_idx_dict[pip] if pip_idx_dict else None
    name = pip_lbl_dict[idx] if pip_lbl_dict else None
    ptype = 'ABCD' if int(pip.split('-')[1]) % n_per_rule==0 else 'ABBA'
    ptype_i = 0 if int(pip.split('-')[1]) % n_per_rule==0 else 1
    group = int(int(pip.split('-')[1]) / n_per_rule)
    full_desc = f'pip {position}\n {ptype}({group})'
    return {'position': position, 'idx': idx, 'name': name, 'ptype': ptype,'ptype_i': ptype_i,
            'group': group, 'desc': full_desc}


def get_pip_info(sound_event_dict,normal_patterns,n_patts_per_rule):
    pip_idxs = {event_lbl: sound_event_dict[event_lbl].idx
                for event_lbl in sound_event_dict
                if any(char in event_lbl for char in 'ABCD')}
    pip_lbls = list(pip_idxs.keys())
    pip_desc = {pip: {} for pip in pip_idxs}  # position in pattern, label, index
    pip_names = {idx: lbl for idx, lbl in zip(sum(normal_patterns, []), 'ABCD' * len(normal_patterns))}
    [pip_desc.update({pip: get_pip_desc(pip, pip_idxs, pip_names, n_patts_per_rule)}) for pip in pip_idxs]
    return pip_desc, pip_lbls,pip_names


def group_responses(event_dict:dict,pip_desc:dict,property:dict):

    events2use = [ee for ee in pip_desc if pip_desc[ee][property.key()] == property.value()]
    responses = np.array([event_dict[e] for e in events2use])

    return responses


def gen_response_df(event_psth_dict:dict, pip_desc:dict, units:list):
    dfs = []
    for event in event_psth_dict:
        event_responses = pd.DataFrame(np.vstack(event_psth_dict[event].mean(axis=0)))
        units = list(range(event_responses.shape[0]))
        properties = [[pip_desc[event][v]]*len(units) for v in ['idx','position','ptype_i','group']]
        multi_idx = pd.MultiIndex.from_arrays([units,[event]*len(units),*properties],
                                              names = ['units','name','idx','position','ptype_i','group'])
        event_responses.index = multi_idx
        dfs.append(event_responses)
    return pd.concat(dfs,axis=0)


def zscore_w_iti(rate_arr, iti_mean, iti_std):
    iti_mean[iti_mean == 0] = np.nan
    iti_std[iti_std == 0] = np.nan
    # ratemat_arr3d = np.transpose((ratemat_arr3d.T - iti_zscore[0].T)/iti_zscore[1].T)
    # for ti in range(iti_zscore[0].shape[0]):
    for ui in range(iti_mean.shape[1]):
        rate_arr[:, ui] = (rate_arr[:, ui] - np.nanmean(iti_mean, axis=0)[ui]) / \
                          np.nanmean(iti_std, axis=0)[ui]
    assert np.isnan(rate_arr).sum() == 0
    return rate_arr


