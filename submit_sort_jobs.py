import yaml
from pathlib import Path
import argparse
import subprocess
import platform
from datetime import datetime
# from postprocessing_utils import get_sorting_dirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('sessions')
    parser.add_argument('--fast_queue_flag',default=0)
    args = parser.parse_args()

    sessions = args.sessions.split('-')

    with open(args.config_file,'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])

    ephys_dir = ceph_dir / 'Dammy' / 'ephys'
    assert ephys_dir.is_dir()
    script_base = r'sbatch -p gpu -t 0-12:0 run_sorter_slurm.sh config.yaml <rec_dir> <extras>'
    if args.fast_queue_flag:
        script_base = r'sbatch -p fast -t 0-2:0 run_sorter_slurm.sh config.yaml <rec_dir> <extras>'
    for session in sessions:
        mouseID = session.split('_')[0]
        date_str = datetime.strptime(session.split('_')[1], '%y%m%d').strftime('%Y-%m-%d')
        recording_dirs = sorted(list(ephys_dir.glob(f'{mouseID}_{date_str}*')))
        extras = ';'.join([str(e) for e in recording_dirs[1:]])
        cmd = script_base.replace('<rec_dir>',str(recording_dirs[0])).replace('<extras>',extras)
        print(cmd)
        subprocess.Popen(cmd.split())
    exit()
