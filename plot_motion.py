from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.full as si
import argparse
import yaml
import platform
from ephys_analysis_funcs import posix_from_win
from spikeinterface import widgets


parser = argparse.ArgumentParser()
parser.add_argument('config_file')
args = parser.parse_args()
with open(args.config_file,'r') as file:
    config = yaml.safe_load(file)
sys_os = platform.system().lower()
ceph_dir = Path(config[f'ceph_dir_{sys_os}'])

folder = ceph_dir / posix_from_win(r'X:\Dammy\ephys\DO79_2024-01-23_18-48-18_003\sorting\drift_corr_a')

motion_info = si.load_motion_info(folder)

motion_plot = si.plot_motion(motion_info,depth_lim=(400, 600),
                             color_amplitude=True, amplitude_cmap='inferno', scatter_decimate=10)
motion_plot.figure.show()
