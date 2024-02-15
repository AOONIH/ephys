import multiprocessing

from ephys_analysis_funcs import *
import platform
import argparse
import yaml
from scipy.stats import pearsonr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('pkldir')
    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])

    pkldir = ceph_dir/posix_from_win(args.pkldir)
    assert pkldir.is_dir()
    sess_pkls = list(pkldir.glob('*.pkl'))

    sessions = {}
    # for sess_pkl in tqdm(sess_pkls,desc='load session pickle', total=len(sess_pkls)):
    with multiprocessing.Pool() as pool:
        sess_objs = list(tqdm(pool.imap(load_session_pkls,sess_pkls),
                              desc='load session pickle', total=len(sess_pkls)))
        for s in sess_objs:
            sessions[s.sessname] = s

        # with open(sess_pkl, 'rb') as pklfile:
        #     sess_obj = pickle.load(pklfile)
        #     sess_obj.sound_event_dict = {}
        #     sessions[sess_obj.sessname] = sess_obj
