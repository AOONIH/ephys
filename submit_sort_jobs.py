import pandas as pd
import yaml
from pathlib import Path, PurePosixPath, PureWindowsPath
import argparse
import subprocess
import platform

from tqdm import tqdm

from aggregate_ephys_funcs import load_aggregate_td_df


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
    parser.add_argument('--partition', default='gpu')
    parser.add_argument('--sess_top_tag', default='ephys_2401')
    parser.add_argument('--sess_top_filts', default='')
    parser.add_argument('--td_df_filts', default='')
    parser.add_argument('--analyse_only', default=0, type=int)
    parser.add_argument('--redo_preprocessing', default=0)
    parser.add_argument('--redo_sorting', default=0)
    parser.add_argument('--redo_postprocessing', default=0)
    parser.add_argument('--run_local', default=False)
    parser.add_argument('--concat', default=False, action='store_true',help='Whether to concatenate the sorting results for multiple sessions.')
    parser.add_argument('--by_sesstype', default=False, action='store_true',)

    args = parser.parse_args()

    sessions = args.sessions.split('-')

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])
    ephys_dir = ceph_dir / 'Dammy' / 'ephys'
    home_dir = Path(config[f'home_dir_{sys_os}'])

    assert ephys_dir.is_dir()

    partition = args.partition
    run_t = '0-2:0' if partition == 'fast' else '0-12:0'

    run_script = 'run_ephys_analysis.sh' if not args.run_local else 'python ephys_analysis_multisess.py]'

    if args.run_local:
        run_script = 'python ephys_analysis_multisess.py'
        print(f'Running locally with script: {run_script}')
        script_base = (
            f"{run_script} {args.config_file} "
            f"<sessname> "
            f"--sorter_dirname <output2use> "
            f'--sess_top_filts "{args.sess_top_filts}" '
            f'--rel_sorting_path <rel_path> '
            f"--sess_top_tag {args.sess_top_tag}"
        )

    elif args.analyse_only:
        script_base = f"{run_script} --session_id <sessname> --sess_path <output2use> --sess_top_tag {args.sess_top_tag}"
    else:
        script_base = (
            f"sbatch -p {partition} -t {run_t} run_sorter_slurm.sh "
            f"--config_file {args.config_file} "
            f"--session_id <sessname> "
            f"--datadir <rec_dir> "
            f"--extra_datadirs <extras> "
            f"--sess_path <sess_path> "
            f"--ow_flag_preprocessing {args.redo_preprocessing} "
            f"--ow_flag_sorting {args.redo_sorting} "
            f"--ow_flag_concat {args.redo_postprocessing} "
            f"--sorter_dirname <output2use> "
            f"--sess_top_filts {args.sess_top_filts} "
            f"--rel_sorting_path <rel_path> "
            f"--sess_top_tag {args.sess_top_tag}"
        )

    sess_topology_path = ceph_dir / posix_from_win(
        rf'X:\Dammy\Xdetection_mouse_hf_test\session_topology_{args.sess_top_tag}.csv'
    )

    session_topology = pd.read_csv(sess_topology_path)
    if args.sess_top_filts:
        session_topology = session_topology.query(f'{args.sess_top_filts}')
    all_td_df = load_aggregate_td_df(session_topology, home_dir, args.td_df_filts) if args.td_df_filts else None

    rel_path_to_sorting = Path(f'sorting{config["rec_dir_suffix"]}') / f'{config["sorter"]}{config["sorter_dir_suffix"]}'
    sess_types = ['pre','main','post']

    for session in tqdm(sessions, total=len(sessions), desc='Submitting jobs'):
        mouseID = session.split('_')[0]
        date_str = session.split('_')[1] if len(session.split('_')) == 2 else None

        if not date_str:
            dates = session_topology.query('name==@mouseID')['date'].unique()
        else:
            dates = [int(date_str)]
    

        if args.td_df_filts:
            dates = [
                date for date in dates
                if any(f'{mouseID}_{date}' in s for s in all_td_df.index.get_level_values('sess').unique())
            ]

        for date in dates:
            sess_filt_str = args.sess_top_filts if args.sess_top_filts else ''
            sess_filt_str = sess_filt_str.replace('== <sess_order>', f'in {sess_types}') if args.by_sesstype else sess_filt_str
            sess_info = session_topology.query(
                f'name==@mouseID & date==@date & {sess_filt_str}'
            )

            # print(session_topology.query(
            #     f'name==@mouseID & date==@date'
                
            # )['ephys_dir'])

            if sess_info.empty:
                continue

            recording_dirs = sorted(sess_info['ephys_dir'].tolist())
            sess_orders = sess_info.sort_values('ephys_dir')['sess_order'].tolist()
            if not recording_dirs:
                continue

            rec_dirs = sorted([str(e) for e in recording_dirs if isinstance(e, (str, Path)) and e])
            if not rec_dirs:
                continue
            if args.concat:
                rec_dirs = [rec_dirs[0]]
            # print(f'{rec_dirs,type(rec_dirs) = }')

            extras = ';'.join(map(str, recording_dirs[1:])) if len(recording_dirs) > 1 and args.concat else 'na'
            sessname = f'{mouseID}_{date}'
            output2use = 'from_concat' if sess_info.shape[0] > 1 and args.concat else 'si_output'
            rel_path = str(rel_path_to_sorting)

            # cmd = script_base.replace('<rec_dir>', rec_dir)\
            #                  .replace('<extras>', extras)\
            #                  .replace('<sessname>', sessname)\
            #                  .replace('<output2use>', output2use)\
            #                  .replace('<rel_path>', rel_path)\
            #                  .replace('<sess_path>', output2use)
            for ri,rec_dir in enumerate(rec_dirs):
                if args.run_local:
                    _sess_top_query = args.sess_top_filts.replace('<sess_order>',f"'{sess_orders[ri]}'" if args.by_sesstype else '')
                    cmd = [
                        'python',
                        'ephys_analysis_multisess.py',
                        args.config_file, sessname,
                        '--sorter_dirname', output2use,
                        '--sess_top_tag', args.sess_top_tag,
                        '--sess_top_filts', f'{_sess_top_query}',
                        '--rel_sorting_path', rel_path
                    ]
                    print(f'Running locally for {sessname} with command:\n{cmd}')
                    print(' '.join(cmd))
                    subprocess.run(cmd, shell=False)
                else:
                    cmd = [
                        "sbatch", "-p", partition, "-t", run_t, "run_sorter_slurm.sh",
                        "--session_id", sessname,
                        "--datadir", rec_dir,
                        "--extra_datadirs", extras if extras else "na",
                        "--sess_path", output2use,
                        "--ow_flag_sorting", str(args.redo_sorting),
                        '--ow_flag_preprocessing', str(args.redo_preprocessing),
                        "--ow_flag_concat", str(args.redo_postprocessing),
                        "--sorter_dirname", output2use,
                        "--sess_top_filts", args.sess_top_filts.replace('<sess_order>',f"'{sess_orders[ri]}'" if args.by_sesstype else ''),
                        "--rel_sorting_path", rel_path,
                        "--sess_top_tag", args.sess_top_tag,
                        "--config_file", args.config_file
                        ]

                    # print(f'Submitting job for {sessname}\n {cmd}')
                    subprocess.run(cmd)

    exit()