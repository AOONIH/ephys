import pandas as pd
import yaml
from pathlib import Path, PurePosixPath, PureWindowsPath
import argparse
import subprocess
import platform
from datetime import datetime
# from postprocessing_utils import get_sorting_dirs


def posix_from_win(path: str) -> Path:
    """
    Convert a Windows path to a Posix path.

    Args:
        path (str): The input Windows path.

    Returns:
        Path: The converted Posix path.
    """
    if ':\\' in path:
        path_bits = PureWindowsPath(path).parts
        path_bits = [bit for bit in path_bits if '\\' not in bit]
        return Path(PurePosixPath(*path_bits))
    else:
        return Path(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('sessions')
    parser.add_argument('--partition',default='gpu')
    parser.add_argument('--sess_top_filts', default='')
    parser.add_argument('--analyse_only',default=0,type=int)
    parser.add_argument('--redo_sorting',default=0)
    parser.add_argument('--redo_analysis',default=0)
    parser.add_argument('--run_local',default=False)

    args = parser.parse_args()

    sessions = args.sessions.split('-')
    # print(f'{sessions = }')

    with open(args.config_file,'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])

    ephys_dir = ceph_dir / 'Dammy' / 'ephys'
    assert ephys_dir.is_dir()
    partition = args.partition
    if partition == 'fast':
        run_t = '0-2:0'
    else:
        run_t = '0-12:0'
    if args.analyse_only:
        script_base = rf'sbatch run_ephys_analysis.sh <sessname> <output2use>'
        if args.run_local:
            script_base = script_base.replace('sbatch run_ephys_analysis.sh','run_ephys_analysis.bat')
            print(script_base)
    else:
        script_base = rf'sbatch -p {partition} -t {run_t} run_sorter_slurm.sh config.yaml <rec_dir> <extras> <sessname> <redo_sorting> <redo_analysis> <output2use>'

    sess_topology_path = ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology.csv')

    session_topology = pd.read_csv(sess_topology_path)
    for session in sessions:
        mouseID = session.split('_')[0]
        date_str = session.split('_')[1] if len(session.split('_')) == 2 else None
        if not date_str:
            dates = session_topology.query('name==@mouseID')['date'].unique()
        else:
            dates = [int(date_str)]
        for date in dates:
            sess_info = session_topology.query(f'name==@mouseID & date==@date  '
                                               f'{f"& {args.sess_top_filts}" if args.sess_top_filts else ""}')
            if sess_info.empty:
                continue
            # recording_dirs = sorted(list(ephys_dir.glob(f'{mouseID}_{date_str}*')))
            recording_dirs = sorted(sess_info['ephys_dir'].tolist())
            extras = ';'.join([str(e) for e in recording_dirs[1:]])
            cmd = script_base.replace('<rec_dir>',str(recording_dirs[0])).replace('<extras>',extras if extras else "na").replace('<sessname>',f'{mouseID}_{date}')
            concat_flag = 'from_concat' if sess_info.shape[0] > 1 else 'si_output'
            cmd = cmd.replace('<redo_sorting>',str(args.redo_sorting)).replace('<redo_analysis>',str(args.redo_analysis)).replace('<output2use>',concat_flag)
            print(cmd)
            subprocess.run(cmd.split()+[f'{args.sess_top_filts}'])

        # date_str = datetime.strptime(session.split('_')[1], '%y%m%d').strftime('%Y-%m-%d')

    exit()
