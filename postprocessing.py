
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.full as si
import argparse
import yaml
import platform
import os
from ephys_analysis_funcs import posix_from_win
from spikeinterface import comparison as sc, widgets as sw
from datetime import datetime
import functools
import pandas as pd
from postprocessing_utils import get_sorting_dirs, get_sorting_objs, get_waveforms,postprocess
from spikeinterface.postprocessing import (compute_spike_amplitudes, compute_correlograms,compute_template_similarity,
                                           compute_unit_locations)
from spikeinterface.qualitymetrics import compute_quality_metrics
from spikeinterface.exporters import export_report
from spikeinterface.curation import MergeUnitsSorting, get_potential_auto_merge, CurationSorting


if __name__ == '__main__':
    # assert False
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('sess_date')
    args = parser.parse_args()
    with open(args.config_file,'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])
    date_str = datetime.strptime(args.sess_date,'%y%m%d').strftime('%Y-%m-%d')

    ephys_dir = ceph_dir / 'Dammy' / 'ephys'
    assert ephys_dir.is_dir()

    sorting_dirs = get_sorting_dirs(ephys_dir, date_str, 'sorting_no_si_drift', 'kilosort2_5_ks_drift')
    sorter_outputs, recordings = get_sorting_objs(sorting_dirs)

    print(f'generating waveform extractors')

    [postprocess(recording, sorting, sort_dir)
     for sorting, recording, sort_dir in zip(sorter_outputs, recordings, sorting_dirs)]