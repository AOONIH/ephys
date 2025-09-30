import shutil

import spikeinterface.full as si
from spikeinterface.preprocessing.motion import motion_options_preset
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from io_utils import posix_from_win
import probeinterface as pi
import warnings
import yaml
import argparse
import sys
import platform
import numpy as np
from pathlib import Path, PurePosixPath
from loguru import logger
from rich.logging import RichHandler
from pyinspect import install_traceback
from datetime import datetime
from tqdm import tqdm
import pathlib
from copy import deepcopy as copy
import pandas as pd
import os
from matplotlib import pyplot as plt
import functools
import subprocess
warnings.simplefilter("ignore")


def sort_recording(base_dir,sorter, probe_name,index=0, ow_flag=False,container_flag=True,sorter_dir_suffix='',
                   extra_folders=None, recording_dir_suffix='', **kwargs):

    dirs2sort = [base_dir]
    preprocessed_recs = []
    job_kwargs = dict(n_jobs=32, chunk_duration="1s", progress_bar=True)

    if extra_folders:
        dirs2sort = dirs2sort+extra_folders
    for base_dir in dirs2sort:
        rec_dir = base_dir / f'sorting{recording_dir_suffix}'
        preprocessed_dir = rec_dir/'preprocessed'
        if not isinstance(base_dir, Path):
            base_dir = Path(base_dir)
        logger.debug(f'rec_dir = {rec_dir}: {rec_dir.is_dir()}')

        if not rec_dir.is_dir():
            rec_dir.mkdir()

        if not preprocessed_dir.is_dir():
            preprocessed_dir.mkdir()

        if preprocessed_dir.is_dir():
            raw_files = list(preprocessed_dir.glob('*.raw'))
        else:
            raw_files = []

        preprocessed_rec = None
        if raw_files and not kwargs.get('ow_flag_preprocessing',ow_flag):
        # if raw_files:
            try:
                logger.debug('reading preprocessed dir')
                preprocessed_rec = si.load_extractor(preprocessed_dir)
                logger.debug('read  preprocessed folder')
            except FileNotFoundError: pass
            except ValueError: pass

        if not preprocessed_rec:
            logger.debug('reading openephys dir')
            full_raw_rec = si.read_openephys(base_dir, block_index=index,stream_id='0')
            logger.debug('read openephys dir')
            probes = gen_probe_group(probe_name)
            full_raw_rec = full_raw_rec.set_probegroup(probes)
            shank_ids = full_raw_rec.get_probegroup().to_dataframe()['shank_ids'].values
            full_raw_rec.set_property("shank_id", shank_ids)
            split_recording_dict = full_raw_rec.split_by("shank_id")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preprocessed_rec_by_group = [preprocess(rec,rec_dir=rec_dir) for rec in split_recording_dict.values()]
            ids = np.hstack([rec._main_ids for rec in preprocessed_rec_by_group])
            preprocessed_rec = si.aggregate_channels(preprocessed_rec_by_group,ids)
            logger.debug('read and processed openephys folder')
            preprocessed_rec.save(folder=rec_dir/'preprocessed', **job_kwargs, overwrite=True,verbose=False,
                                  format='binary',export_probe=True, probe_name='probe.prb',dtype=np.int16)

        preprocessed_recs.append(preprocessed_rec)
        # plot_rec_overview(preprocessed_rec,rec_dir)
    pi.write_prb(preprocessed_dir / 'probe.prb', preprocessed_rec.get_probegroup())

    if len(preprocessed_recs) > 1:
        # raise Warning('multisession not supported yet')
        name = dirs2sort[0].parts[-1].split('_')[0]
        sess_date = datetime.strptime(dirs2sort[0].parts[-1].split('_')[1][:10],'%Y-%m-%d').strftime('%y%m%d')
        concat_dir = dirs2sort[0].parent/f'{name}_{sess_date}_concat'
        if not concat_dir.is_dir():
            concat_dir.mkdir()
        rec_dir = concat_dir / f'sorting{recording_dir_suffix}'
        if not rec_dir.is_dir():
            rec_dir.mkdir()
        pre_rec_dir = rec_dir / 'preprocessed'
        if not pre_rec_dir.is_dir():
            pre_rec_dir.mkdir()
        logger.debug(f'rec_dir = {rec_dir}: {rec_dir.is_dir()}')
        common_channels = functools.reduce(np.intersect1d, [recording.channel_ids for recording in preprocessed_recs])
        segment_info = [[recording.get_num_frames(),recording.get_duration()]
                        for ri,recording in enumerate(preprocessed_recs)]
        preprocessed_recs = [recording.channel_slice(common_channels) for recording in preprocessed_recs]
        all_recordings = si.concatenate_recordings(preprocessed_recs)
        if not list(pre_rec_dir.glob('si_folder.json')):
            all_recordings.save(folder=pre_rec_dir, **job_kwargs, overwrite=True, verbose=False,
                                export_probe=True, probe_name='probe.prb')
        # all_recordings.save(folder=rec_dir/, **job_kwargs, overwrite=True)
        all_recordings = si.load_extractor(pre_rec_dir)
        pd.DataFrame(segment_info,columns=['n_frames','duration']).to_csv(rec_dir/'preprocessed'/'segment_info.csv',
                                                                          index=False)
    else:
        # all_recordings=preprocessed_recs[0]
        all_recordings = si.load_extractor(dirs2sort[0] / f'sorting{recording_dir_suffix}'/'preprocessed')
        rec_dir = dirs2sort[0] / f'sorting{recording_dir_suffix}'



    drift_corr_flag = False if 'no_ks_drift' in sorter_dir_suffix else True

    if (rec_dir / f"{sorter}{sorter_dir_suffix}").is_dir() and not ow_flag:
        try:
            si.load_extractor((rec_dir / f"{sorter}{sorter_dir_suffix}"))
            logger.debug(f'{rec_dir / f"{sorter}{sorter_dir_suffix}"} already exists, checking if it is empty')
            return
        except:
            pass

    if sorter in ['mountainsort5',]:
        sorting_kwargs = {'whiten': True,
                          'filter': False}
    elif sorter == 'tridesclous2':
        sorting_kwargs = {'apply_preprocessing':True}
    else:
        sorting_kwargs = {}
    if 'kilosort' in sorter:
        sorting_kwargs = dict(do_correction=drift_corr_flag,nblocks=0)  # dmin=12.5, dminx=22.5
        container_flag = container_flag
        detect_threshold = 3  # Whitened Threshold (STD from Absolute traces)

        sorting_kwargs = {
            # Use default thresholds first as recommended, change only if necessary
            # 'Th_universal': 9,
            # 'Th_learned': 8,
            # 'Th_single_ch': detect_threshold,

            # Basic recording parameters
            # 'nt': nt,  # Keep your existing time window
            # 'nt0min': snippet_T1,  # Keep your existing alignment point

            # For 16 channels, use 2-4x channels (32-64) rounded to multiple of 32, lower runs faster
            # 'nearest_templates': 32,  #

            # Don't modify spatial parameters unnecessarily
            # 'dminx': 22.5,  #
            # 'dmin': 12.5,  # Let it use defaults
            # 'max_channel_distance': None,  # Remove - no maximum in space needed

            # Keep default template parameters
            # 'min_template_size': 10,  # Use default
            # 'nearest_chans': 1,  # Use default
            # 'x_centers': 1,  # Use default

            # Processing settings
            # 'do_CAR': False,  # Common average reference
            'do_correction': drift_corr_flag,  # Keep drift correction off since MEA plate (diff for neuropixel)
            'skip_kilosort_preprocessing': False,  # Since you do your own
            'nblocks': 0,  # Keep default

            # GPU settings
            # 'save_extra_vars': True  # Save debug info
        }
    logger.debug('Launching sorter')
    if (rec_dir / f'{sorter}{sorter_dir_suffix}').is_dir() and ow_flag:
        shutil.rmtree(rec_dir / f'{sorter}{sorter_dir_suffix}')
    if sorter == 'spykingcircus2' and False:
        aggregate_sorting = si.run_sorter_by_property(sorter_name=sorter, recording=all_recordings,
                                                      grouping_property='by_shank',
                                                      working_folder=rec_dir / f'{sorter}{sorter_dir_suffix}',
                                                      verbose=True, **sorting_kwargs
                                                      )
    elif sorter == 'mountainsort5':
        aggregate_sorting = si.run_sorter(sorter_name=sorter, recording=all_recordings,
                                           folder=rec_dir / f'{sorter}{sorter_dir_suffix}',
                                           remove_existing_folder=True,
                                           verbose=True, **sorting_kwargs
                                           )
    else:
        logger.debug(f'{sorter} outdir = {rec_dir / f"{sorter}{sorter_dir_suffix}"}')
        aggregate_sorting = si.run_sorter_by_property(sorter_name=sorter, recording=all_recordings,
                                                      folder=rec_dir / f'{sorter}{sorter_dir_suffix}',
                                                      grouping_property='shank_id',
                                                      singularity_image=container_flag,
                                                      delete_container_files=False,
                                                      delete_output_folder=False,
                                                      remove_existing_folder=True,verbose=True,
                                                      # extra_requirements=['kilosort==4.0.18'] if (container_flag and 'kilosort' in sorter) else None,
                                                      **sorting_kwargs
                                                      )
        # aggregate_sorting = si.run_sorter_by_property(sorter_name=sorter, recording=all_recordings,
        #                                               grouping_property='group',
        #                                               working_folder=rec_dir / f'{sorter}{sorter_dir_suffix}',
        #                                               singularity_image=container_flag, delete_container_files=False,
        #                                               remove_existing_folder=True,verbose=True, **sorting_kwargs
        #                                               )
    aggregate_sorting.save(folder=rec_dir / f'{sorter}{sorter_dir_suffix}'/'si_output', overwrite=True,verbose=True)
    # export to phy

    # if len(preprocessed_recs) > 1:
    #     # get date_str
    #     sess_date = datetime.strptime(dirs2sort[0].parts[-1].split('_')[1][:10], '%Y-%m-%d').strftime('%y%m%d')
    #     subprocess.run(f'python split_concat.py config.yaml {sess_date} --ow_flag 1'.split(' '),shell=True)

    return aggregate_sorting


def get_probe(probe_name):
    manufacturer = 'cambridgeneurotech'

    probe1 = pi.get_probe(manufacturer, probe_name)
    probe1.wiring_to_device('cambridgeneurotech_mini-amp-64')

    return probe1


def preprocess(recording,filter_range=(300,6000),rec_dir=''):
    bad_channel_ids, channel_labels = si.detect_bad_channels(recording, seed=1)
    recording = si.interpolate_bad_channels(recording=recording, bad_channel_ids=bad_channel_ids)
    recording = cmr_by_shank(recording,filter_range)
    job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)
    logger.debug(f'drift kwargs = {job_kwargs}')
    # recording, motion_info = correct_drift(recording,'nonrigid_fast_and_accurate',rec_dir,job_kwargs)
    # recording, motion_info = correct_drift(recording,'nonrigid_accurate',rec_dir,job_kwargs)
    # set shank id property
    shank_ids = recording.get_probegroup().to_dataframe()['shank_ids'].values
    recording.set_property("shank_id", shank_ids)
    # recording = si.whiten(recording, mode='local', radius_um=12.5 * 3, dtype="float32")
    return recording


def cmr_by_shank(recording,filter_range=(300,9000)):
    return recording
    # bad_channels_ids, _ = si.detect_bad_channels(recording)
    # logger.info(f'bad channels = {bad_channels_ids}, total={len(bad_channels_ids)}')
    # recording = recording.remove_channels(bad_channels_ids)
    recording = si.bandpass_filter(recording, freq_min=filter_range[0], freq_max=filter_range[1])

    probe_df = recording.get_probe().to_dataframe(complete=True)
    main_ids = copy(recording._main_ids)
    # recording._main_ids = recording.ids_to_indices(ids=None)
    # chan_letters = np.unique(
    #     [chan_id.split('_')[0] if len(chan_id.split('_')) > 1 else [''] for chan_id in recording._main_ids])

    # cmr_group_param = 'shank_ids'
    # cmr_groups_idx = [probe_df[probe_df[cmr_group_param] == i]['device_channel_indices'].astype(int).to_list()
    #                   for i in probe_df[cmr_group_param].unique()]

    # recording_cmr = si.common_reference(recording, reference='global', operator='median', groups=cmr_groups)
    recording_cmr = si.common_reference(recording, reference='global', operator='median')
    # recording_cmr._main_ids = main_ids
    return recording_cmr


def correct_drift(recording,preset,rec_dir='',job_kwargs={}):
    logger.info(rec_dir)
    output_dir = unique_file_path(Path(rec_dir) / 'drift_corr','_a')
    recording_corrected, motion_info = si.correct_motion(recording,
                                                         folder=output_dir,
                                                         preset=preset,output_motion_info=True,
                                                         interpolate_motion_kwargs={'border_mode':'force_extrapolate'},
                                                         **job_kwargs)
    motion_plot = si.plot_motion(motion_info,
                                 color_amplitude=True, amplitude_cmap='inferno', scatter_decimate=10)
    motion_plot.figure.set_size_inches(21, 14)
    motion_plot.figure.savefig(output_dir/'motion_info.svg')
    return recording_corrected,motion_info


def gen_probe_group(probe_name='ASSY-236-P-1'):

    manufacturer = 'cambridgeneurotech'

    probes = pi.ProbeGroup()

    if probe_name.endswith('.json'):
        probes = pi.read_probeinterface(probe_name)
        return probes

    probe1 = pi.get_probe(manufacturer, probe_name)
    probe1.wiring_to_device('cambridgeneurotech_mini-amp-64')
    # probe1 = probe.copy()
    probe2 = probe1.copy()
    probe2.wiring_to_device('cambridgeneurotech_mini-amp-64')
    probe2.set_device_channel_indices(probe1.device_channel_indices+64)
    if np.unique(probe1.shank_ids).shape[0]>1:
        probe2.set_shank_ids((probe1.shank_ids.astype(int)+4).astype(str))
    # logger.debug(probe1.device_channel_indices,probe2.device_channel_indices)
    probe2.move([5000,0])
    probes.add_probe(probe1)
    probes.add_probe(probe2)
    probes.set_global_device_channel_indices(np.concatenate([probe1.device_channel_indices,
                                                             probe2.device_channel_indices]))

    return probes


def plot_rec_overview(recording,rec_dir):
    figdir_path = unique_file_path(Path(rec_dir) / 'overview_plots','_a')

    # plot peaks
    job_kwargs = dict(n_jobs=os.cpu_count(), chunk_duration='1s', progress_bar=True)
    noise_levels_int16 = si.get_noise_levels(recording, return_scaled=False)
    peaks = detect_peaks(recording, method='locally_exclusive', noise_levels=noise_levels_int16,
                         detect_threshold=5, radius_um=50., **job_kwargs)
    peak_locations = localize_peaks(recording, peaks, method='center_of_mass', radius_um=50., **job_kwargs)

    fig, ax = plt.subplots(figsize=(15, 10))
    si.plot_probe_map(recording, ax=ax, with_channel_ids=True)
    ax.set_ylim(-50, 200)
    ax.scatter(peak_locations['x'], peak_locations['y'], color='purple', alpha=0.002)
    fig.savefig(figdir_path/'peaks.svg')


def unique_file_path(path, suffix='_a'):
    if not isinstance(path, (pathlib.WindowsPath, pathlib.PosixPath)):
        path = Path(path)
    if suffix:
        path = path.with_stem(f'{path.stem}{suffix}')
    while path.exists():
        new_stem = f'{path.stem[:-1]}{chr(ord(path.stem[-1])+1)}'
        path = path.with_stem(new_stem)
    return path


if __name__ == "__main__":
    install_traceback()

    # logger.configure(
    #     handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}]
    # )
    logger.info(f'started and loading config ')
    today_str = datetime.today().date().strftime('%y%m%d')
    logger_path = Path.cwd() / f'log' / f'log_{today_str}.txt'
    logger.add(logger_path, level='INFO')

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('--datadir',default=None)
    parser.add_argument('--extra_datadirs',default='')
    parser.add_argument('--ow_flag_preprocessing',default=0,type=int)
    parser.add_argument('--ow_flag',default=0,type=int)
    parser.add_argument('--use_ceph',default=1,type=int)

    args = parser.parse_args()
    with open(args.config_file,'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])

    rec_dir_suffix = config.get('rec_dir_suffix','')
    sorter_name = config['sorter']
    if args.datadir:
        folder = args.datadir
        if args.extra_datadirs and not args.extra_datadirs == 'na':
            extra_folders = [ceph_dir / posix_from_win(e) for e in args.extra_datadirs.split(';')]
        else:
            extra_folders = []
    else:
        folder = config['recording_dir']
        extra_folders = config.get('extra_dirs', [])
    if extra_folders:
        # extra_folders = [ceph_dir/posix_from_win(recdir) for recdir in extra_folders]
        if not rec_dir_suffix:
            rec_dir_suffix = '_concat'

    logger.info(f'loaded config for {folder[-1]}')
    ow_flag = args.ow_flag if args.ow_flag else config.get('ow_flag',False)
    container_flag = config.get('container_flag', False)
    block_idx = config.get('block_idx', 0)

    print(f'{args = }')

    recording_dir = ceph_dir/posix_from_win(folder) if args.use_ceph else Path(folder)
    sorter_output = sort_recording(recording_dir,sorter_name,probe_name=config['probe_name'],
                                   ow_flag_preprocessing=args.ow_flag_preprocessing,
                                   ow_flag=ow_flag,container_flag=container_flag,
                                   sorter_dir_suffix=config.get('sorter_dir_suffix', ''),index=block_idx,
                                   extra_folders=extra_folders,recording_dir_suffix=rec_dir_suffix,sys_os=sys_os)
