import spikeinterface.full as si
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
from copy import deepcopy as copy
import pandas as pd
import os
warnings.simplefilter("ignore")


def sort_recording(base_dir,sorter, probe_name,index=0, ow_flag=False,container_flag=True,sorter_dir_suffix='',):
    if not isinstance(base_dir, Path):
        base_dir = Path(base_dir)
    rec_dir = base_dir / 'sorting'
    logger.debug(f'rec_dir = {rec_dir}: {rec_dir.is_dir()}')
    if not rec_dir.is_dir():
        rec_dir.mkdir()

    job_kwargs = dict(n_jobs=os.cpu_count(), chunk_duration="1s", progress_bar=True)
    if rec_dir.is_dir():
        raw_files = list(rec_dir.glob('*.raw'))
    else:
        raw_files = []

    recording_cmr = None
    if raw_files and not ow_flag:
        try:
            recording_cmr = si.load_extractor(rec_dir)
            logger.debug('read processed openephys folder')
        except FileNotFoundError: pass

    if not recording_cmr:
        full_raw_rec = si.read_openephys(base_dir, block_index=index)
        logger.debug('read openephys dir')
        # full_raw_rec = full_raw_rec.save(folder=rec_dir, **job_kwargs, overwrite=True)

        probes = gen_probe_group(probe_name)
        probe_df = probes.to_dataframe(complete=True)

        full_raw_rec = full_raw_rec.set_probegroup(probes)
        recording_f = si.bandpass_filter(full_raw_rec, freq_min=300, freq_max=9000)

        chan_letters = np.unique([chan_id.split('_')[0] if len(chan_id.split('_'))>1 else [''] for chan_id in recording_f._main_ids] )

        cmr_group_param = 'shank_ids'
        cmr_groups_idx = [probe_df[probe_df[cmr_group_param]==i]['device_channel_indices'].astype(int).to_list()
                      for i in probe_df[cmr_group_param].unique()]
        if chan_letters[0]:
            cmr_groups = [[f'{chan_letters[0]}_CH{int(cid)+1}' if int(cid)<64 else f'{chan_letters[1]}_CH{int((int(cid)+1)/2)}' for cid in group] for group in cmr_groups_idx]
        else:
            cmr_groups = [[f'CH{int(cid)+1}' for cid in group] for group in cmr_groups_idx]
        recording_cmr = si.common_reference(recording_f, reference='global', operator='median',groups=cmr_groups)
        # recording_cmr = si.common_reference(recording_f, reference='global', operator='median')

        logger.debug('read and processed openephys folder')
        recording_cmr = recording_cmr.save(folder=rec_dir, **job_kwargs, overwrite=True)



    # aggregate_sorting = si.run_sorter_by_property(sorter_name=sorter, recording=recording_cmr,
    #                                               grouping_property='group',
    #                                               working_folder=rec_dir / sorter,
    #                                               singularity_image=container_flag,
    #                                               remove_existing_folder=True,verbose=True)
    if sorter in ['spykingcircus2']:
        container_flag=False
    logger.debug('Launching sorter')
    aggregate_sorting = si.run_sorter(sorter_name=sorter, recording=recording_cmr,
                                      output_folder=rec_dir / f'{sorter}{sorter_dir_suffix}',
                                      singularity_image=container_flag, delete_container_files=False,
                                      remove_existing_folder=ow_flag,verbose=True)
    aggregate_sorting.to_dict()
    # si.KiloSortSortingExtractor.to_dict()
    waveforms = si.extract_waveforms(recording_cmr,aggregate_sorting,rec_dir / f'{sorter}{sorter_dir_suffix}'/'waveforms',
                                     sparse=True,**job_kwargs,overwrite=True)
    si.export_to_phy(waveforms,output_folder=rec_dir / f'{sorter}{sorter_dir_suffix}'/'phy',
                     compute_amplitudes=False, compute_pc_features=False, copy_binary=False,remove_if_exists=True)

    # recordings = recording_cmr.split_by(property='group', outputs='list')
    # multirecording = si.concatenate_recordings(recordings)



    # multirecording = multirecording.set_probe(probe)
    # multisorting = si.run_sorter(sorter_name=sorter, recording=multirecording,
    #                              output_folder=rec_dir / sorter,
    #                              docker_image=True,remove_existing_folder=True)
    return aggregate_sorting

def get_probe(probe_name):
    manufacturer = 'cambridgeneurotech'

    probe1 = pi.get_probe(manufacturer, probe_name)
    probe1.wiring_to_device('cambridgeneurotech_mini-amp-64')

    return probe1


def gen_probe_group(probe_name='ASSY-236-P-1'):
    manufacturer = 'cambridgeneurotech'

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
    probes = pi.ProbeGroup()
    probes.add_probe(probe1)
    probes.add_probe(probe2)
    probes.set_global_device_channel_indices(np.concatenate([probe1.device_channel_indices,
                                                             probe2.device_channel_indices]))

    return probes


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
    args = parser.parse_args()
    with open(args.config_file,'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])
    folder, sorter_name = config['recording_dir'],config['sorter']
    if args.datadir:
        folder[1] = args.datadir
    logger.info(f'loaded config for {folder[-1]}')
    ow_flag = config.get('ow_flag',False)
    container_flag = config.get('container_flag', False)
    block_idx = config.get('block_idx', 0)
    sorter_output = sort_recording(ceph_dir/Path(*folder),sorter_name,probe_name=config['probe_name'],
                                   ow_flag=ow_flag,container_flag=container_flag,
                                   sorter_dir_suffix=config.get('sorter_dir_suffix', ''),index=block_idx)
