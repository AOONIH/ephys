import pandas as pd
from pathlib import Path
import yaml
from tqdm import tqdm

from io_utils import posix_from_win

sess_top_dir = r"X:\Dammy\Xdetection_mouse_hf_test"
sess_top_paths = list(Path(sess_top_dir).glob('*session_topology*.csv'))
config_path  = Path('config.yaml')
config = yaml.safe_load(open(config_path,'r'))
home_dir = Path(config['home_dir_windows'])

for sess_top_path in tqdm(sess_top_paths,desc='Processing session topology files', total=len(sess_top_paths)):
    session_topology = pd.read_csv(sess_top_path)
    if 'tdata_file' not in session_topology.columns:
        continue
    stages = []
    for td_path in session_topology['tdata_file'].values:
        if not isinstance(td_path,str):
            stages.append(-1)
            continue
        abs_td_path = home_dir/posix_from_win(td_path,'/nfs/nhome/live/aonih')
        if not abs_td_path.is_file():
            stages.append(-1)
            continue
        try:
            td_df = pd.read_csv(abs_td_path)
        except pd.errors.EmptyDataError:
            td_df = pd.DataFrame()
        if td_df.empty:
            stages.append(-1)
            continue
        stages.append(td_df['Stage'].values[0])
    session_topology['Stage'] = stages
    session_topology.to_csv(sess_top_path,index=False)
