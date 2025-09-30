import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import yaml
from io_utils import posix_from_win


def check_pupil_data(sessname:str, ceph_dir:Path) -> None:
    assert len(sessname.split('_')) == 2

    topviddir = ceph_dir/posix_from_win(r'X:\Dammy\mouse_pupillometry\mouse_hf')
    harpbins_dir = ceph_dir/posix_from_win(r'X:\Dammy\harpbins')
    assert topviddir.is_dir() and harpbins_dir.is_dir()

    name, date = sessname.split('_')

    # loop over sessions for this date
    ttl_paths = sorted(harpbins_dir.glob(f'{name}*{date}*_event_data_92.csv'))
    vid_dirs = sorted(topviddir.glob(f'{name}_{date}*'))
    # print(list(ttl_paths), list(vid_dirs))
    for ttl_path, vid_dir in zip(ttl_paths, vid_dirs):
        print(ttl_path)
        ttl_df= pd.read_csv(ttl_path)
        vid_df = pd.read_csv(vid_dir/f'{name}_{date}_eye0_timestamps.csv')
        if ttl_df.shape[0] != vid_df.shape[0]:
            print(f'MISMATCH {ttl_path, vid_dir}:{ttl_df.shape, vid_df.shape =} ')
            print(f'skipped frames:  {(vid_df["FrameID"].diff()>1).sum()}/ {ttl_df.shape[0] -vid_df.shape[0]} missing')
        # assert ttl_df.shape[0] == vid_df.shape[0]

