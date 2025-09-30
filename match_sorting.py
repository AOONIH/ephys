from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.full as si
import argparse
import yaml
import platform
from io_utils import posix_from_win
from spikeinterface import comparison as sc, widgets as sw
from datetime import datetime
import functools
import pandas as pd

from postprocessing_utils import get_sorting_dirs, get_sorting_objs


parser = argparse.ArgumentParser()
parser.add_argument('config_file')
parser.add_argument('sess_date')
args = parser.parse_args()
with open(args.config_file,'r') as file:
    config = yaml.safe_load(file)
sys_os = platform.system().lower()
ceph_dir = Path(config[f'ceph_dir_{sys_os}'])
date_str = datetime.strptime(args.sess_date,'%y%m%d').strftime('%Y-%m-%d')

ephys_dir = ceph_dir /'Dammy'/ 'ephys'
assert ephys_dir.is_dir()

sorting_dirs = get_sorting_dirs(ephys_dir, date_str, 'sorting_no_si_drift', 'kilosort2_5_ks_drift')

recording_dirs = [Path(*folder.parts[:-2])/'preprocessed' for folder in sorting_dirs]

# sorter_outputs = [si.load_extractor(folder) for folder in sorting_dirs]
# recordings = [si.load_extractor(recording_dir) for recording_dir in recording_dirs]
sorter_outputs, recordings = get_sorting_objs(sorting_dirs)
common_channels = functools.reduce(np.intersect1d, [recording.channel_ids for recording in recordings])
recordings = [recording.channel_slice(common_channels) for recording in recordings]
waveforms = [si.extract_waveforms(recording, sorting,sort_dir.parent/'waveforms_common',allow_unfiltered=True,)
             if not (sort_dir.parent/'waveforms_common').is_dir() else si.load_waveforms(sort_dir.parent/'waveforms_common')
             for sorting, recording,sort_dir in zip(sorter_outputs, recordings, sorting_dirs)]
m_tcmp = sc.compare_multiple_templates(waveform_list=waveforms, name_list=['pre','task','post'])
unit_df = pd.DataFrame(m_tcmp.units.values())
# comp = sw.plot_multicomparison_graph(multi_comparison=mcmp)
# mcmp.get_agreement_sorting()
