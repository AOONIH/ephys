import logging

import spikeinterface.full as si
from pathlib import Path
from spikeinterface.extractors import NpzSortingExtractor
import  probeinterface as pi
from loguru import logger
from rich.logging import RichHandler
from pyinspect import install_traceback
from datetime import datetime
import yaml
import argparse
import platform
from sorting_functions import gen_probe_group
import os
# import multiprocessing


def load_saved_sorting_output(
    sorter_output_path: Path, sorter: str
) -> si.BaseSorting:
    """ """
    if "kilosort" in sorter:
        sorting = si.read_kilosort(
            folder_path=sorter_output_path,
            keep_good_only=False,
        )
    elif sorter == "mountainsort5":
        sorting = NpzSortingExtractor((sorter_output_path / "firings.npz").as_posix())

    elif sorter == "tridesclous":
        sorting = si.read_tridesclous(
            folder_path=sorter_output_path
        )

    elif sorter == "spykingcircus":
        sorting = si.read_spykingcircus(
            folder_path=sorter_output_path
        )
    else:
        raise Warning('Incorrect param given')
    return sorting


if __name__ == "__main__":
    install_traceback()

    logger.configure(
        handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}]
    )
    logger.info('started and loading config')
    today_str = datetime.today().date().strftime('%y%m%d')
    logger_path = Path.cwd() / f'log' / f'log_{today_str}.txt'
    logger.add(logger_path, level='INFO')

    install_traceback()
    logger.configure(
        handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}]
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    args = parser.parse_args()
    with open(args.config_file,'r') as file:
        config = yaml.safe_load(file)
    os_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{os_os}'])
    folder, sorter_name = config['recording_dir'],config['sorter']
    ow_flag = config.get('ow_flag',False)
    container_flag = config.get('container_flag', False)

    recording = si.load_extractor(ceph_dir/Path(*folder)/'sorting')
    probes = gen_probe_group()
    recording = recording.set_probegroup(probes)
    ks_out_path = ceph_dir / Path(*folder) / 'sorting' / 'kilosort2_5'
    # outputs = {}
    # waveforms = {}
    # logger.info(multiprocessing.cpu_count())
    job_kwargs = dict(n_jobs=os.cpu_count(), chunk_duration="1s", progress_bar=True)
    # for i, probe in enumerate(list(ks_out_path.iterdir())):
    #     outputs[i] = load_saved_sorting_output(ks_out_path / str(i) / 'sorter_output', 'kilosort2_5')
    # for i in list(outputs):
    #     output = outputs[i]
    #     waveforms[i] = si.extract_waveforms(recording,output,folder=ks_out_path / str(i) / 'waveforms',sparse=True,
    #                                         **job_kwargs,overwrite=True)
    #     si.export_to_phy(waveforms[i],ks_out_path / str(i) / 'phy',remove_if_exists=True,
    #                      compute_amplitudes=False, compute_pc_features=False, copy_binary=False,
    #                      **job_kwargs)
    output = load_saved_sorting_output(ks_out_path /'sorter_output', 'kilosort2_5')
    waveforms = si.extract_waveforms(recording, output, folder=ks_out_path / 'waveforms', sparse=True,)
    si.export_to_phy(waveforms, ks_out_path / 'phy', remove_if_exists=True,
                     compute_amplitudes=True, compute_pc_features=False, copy_binary=False,
                     **job_kwargs)
