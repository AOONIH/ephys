from pathlib import Path
import numpy as np
import spikeinterface.full as si
import argparse
import yaml
import platform
import os
from ephys_analysis_funcs import posix_from_win,plot_2d_array_with_subplots
from postprocessing_utils import get_sorting_dirs, get_sorting_objs, plot_periodogram,get_probe_power, postprocess
from datetime import datetime
import pandas as pd
from scipy import signal
from matplotlib import pyplot as plt
import shutil
from tqdm import tqdm
import multiprocessing
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('sess_dates')
    parser.add_argument('--ow_flag',default=0,type=int)
    parser.add_argument('--sess_top_filts', default='')
    args = parser.parse_args()

    with open(args.config_file,'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])
    print(args)
    assert args.sess_dates and '_' in args.sess_dates
    sess_names = args.sess_dates.split('-')
    ephys_dir = ceph_dir / 'Dammy' / 'ephys'
    # ephys_dir = ceph_dir / posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\rawdata')
    assert ephys_dir.is_dir()

    sess_topology_path = ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology.csv')
    session_topology = pd.read_csv(sess_topology_path)

    for sess_name in sess_names:
        name, date = args.sess_dates.split('_')
        date = int(date)
        all_sess_info = session_topology.query('name==@name & date==@date').reset_index(drop=True)
        if args.sess_top_filts:
            all_sess_info = all_sess_info.query(args.sess_top_filts)
        dir1_name, dir2_name = 'sorting_no_si_drift', 'kilosort2_5_ks_drift'
        date_str = datetime.strptime(sess_name.split('_')[-1],'%y%m%d').strftime('%Y-%m-%d')
        sorting_dirs = get_sorting_dirs(ephys_dir, f'{sess_name.split("_")[0]}_{date_str}', dir1_name, dir2_name)
        sorting_dirs = [e for ei,e in enumerate(sorting_dirs) if ei in all_sess_info.index]

        if len(sorting_dirs) < 2:
            single_sorting,single_rec = get_sorting_objs(sorting_dirs)
            postprocess(single_rec[0],single_sorting[0],sorting_dirs[0])
            spike_vectors = [sorting.to_spike_vector() for sorting in single_sorting]
            [np.save(sort_dir.parent / f'si_output' / 'spike_times.npy', [e['sample_index'] for e in spikes])
             for sort_dir, spikes in zip(sorting_dirs, spike_vectors)]
            [np.save(sort_dir.parent / f'si_output' / 'spike_clusters.npy', [e['unit_index'] for e in spikes])
             for sort_dir, spikes in zip(sorting_dirs, spike_vectors)]
            continue
        concat_dirs = get_sorting_dirs(ephys_dir, sess_name, dir1_name, dir2_name)
        concat_sorting, concat_rec = get_sorting_objs(concat_dirs)
        if not (concat_dirs[0].parent/'good_units.csv').is_file() or args.ow_flag:
            postprocess(concat_rec[0], concat_sorting[0], concat_dirs[0])
        concat_dir = concat_dirs[0]
        concat_preprocessed_dir = concat_dir
        while not (concat_preprocessed_dir/'preprocessed').is_dir():
            concat_preprocessed_dir = concat_preprocessed_dir.parent
        concat_preprocessed_dir = concat_preprocessed_dir/'preprocessed'

        try: segment_info = pd.read_csv(concat_preprocessed_dir / 'segment_info.csv')
        except FileNotFoundError: segment_info = pd.DataFrame()

        segments = []

        if segment_info.empty:
            for s_dir in sorting_dirs:
                while not (s_dir/'preprocessed').is_dir():
                    s_dir = s_dir.parent
                s_dir_preprocessed = si.load_extractor(s_dir/'preprocessed')
                segments.append(s_dir_preprocessed.get_num_frames())
            segment_info['n_frames'] = segments

        print(f'splitting recordings for {concat_dirs[0]}')
        split_recordings,split_sortings = [[obj.frame_slice(start, start+n_frames)
                                           for start, n_frames in zip(np.cumsum(np.pad(segment_info['n_frames'], [1, 0])),
                                                                      segment_info['n_frames'])] for obj in (concat_rec[0],
                                                                                                             concat_sorting[0])]
        # assert False
        [(sort_dir.parent/'from_concat').mkdir(exist_ok=True) for sort_dir in sorting_dirs if
         not (sort_dir.parent/'from_concat').is_dir() or args.ow_flag]
        [(sorting.save(folder=sort_dir.parent/f'from_concat'/'si_output',overwrite=args.ow_flag),print(f'saved {sort_dir.parent/f"from_concat"}/si_output'))
         for sorting, sort_dir in zip(split_sortings, sorting_dirs) if not (sort_dir.parent/f'from_concat'/'si_output').is_dir() or args.ow_flag]
        spike_vectors = [sorting.to_spike_vector() for sorting in split_sortings]
        [np.save(sort_dir.parent/f'from_concat'/'spike_times.npy', [e['sample_index'] for e in spikes])
         for sort_dir,spikes in zip(sorting_dirs,spike_vectors)]
        [np.save(sort_dir.parent / f'from_concat'/'spike_clusters.npy', [e['unit_index'] for e in spikes])
         for sort_dir, spikes in zip(sorting_dirs, spike_vectors)]
        [shutil.copy(concat_dir.parent/'good_units.csv', sort_dir.parent/'from_concat'/'good_units.csv',)
         for sort_dir in sorting_dirs]

        print(f'finished splitting recordings for {concat_dirs[0]}')
    print('done')
    # cnt=0

    # for folder in ephys_dir.iterdir():
    #     try:
    #         shutil.rmtree(folder/'sorting_no_si_drift'/'sorting_no_si_drift')
    #     except:
    #         pass
    #     try:
    #         shutil.rmtree(folder/'sorting_no_si_drift'/'kilosort2_5_ks_drift'/'sorting_no_si_drift')
    #     except:
    #         pass
    #     try:
    #         shutil.rmtree(folder/'sorting_no_si_drift'/'in_container_python_base')
    #     except:
    #         pass
    #     if (folder/'sorting_no_si_drift').is_dir():
    #         for ff in (folder/'sorting_no_si_drift').iterdir():
    #             try:
    #                 shutil.rmtree(ff/'sorting_no_si_drift')
    #             except:

    #                 pass

    # while next(ephys_dir.rglob('sorting_no_si_drift/sorting_no_si_drift')):
    #     try:
    #         dir2del = next(ephys_dir.rglob('sorting_no_si_drift/sorting_no_si_drift/sorting_no_si_drift'))
    #         if dir2del.is_dir():
    #             subprocess.run(['rm', '-rf', dir2del], check=True)
    #             cnt+=1
    #             assert not dir2del.is_dir()
    #             print(f'{cnt,dir2del}')
    #     except FileNotFoundError:
    #         pass
