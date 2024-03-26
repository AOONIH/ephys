import multiprocessing

import matplotlib.pyplot as plt
import numpy as np

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
    # with multiprocessing.Pool() as pool:
    #     sess_objs = list(tqdm(pool.imap(load_session_pkls,sess_pkls),
    #                           desc='load session pickle', total=len(sess_pkls)))
    #     for s in sess_objs:
    #         sessions[s.sessname] = s
    #
    #     # with open(sess_pkl, 'rb') as pklfile:
    #     #     sess_obj = pickle.load(pklfile)
    #     #     sess_obj.sound_event_dict = {}
    #     #     sessions[sess_obj.sessname] = sess_obj
    # for sess in sessions:
    #     sess_td = sessions[sess].td_df
    #     rare = sess_td[sess_td['local_rate']]>=0.8
    #     freq = sess_td[sess_td['local_rate']]<=0.2
    #     # fig,ax = plt.subplots()
    #     print(f'{sess}: rare:{rare.index}, freq:{freq.index}')

    npyss = pkldir.glob('*.npy')
    dates2use = ['240219','240221']
    data = [np.load(file) for file in npyss if any(d in str(file) for d in dates2use)]
    # data.pop(0)
    window = [-1,3]
    x_ser = np.linspace(window[0],window[1],data[0].shape[-1])

    all_data = np.vstack(data)
    all_data = savgol_filter(all_data,10,1,axis=1)
    rare_freq_psth_ts = plot_psth_ts(all_data,x_ser,'',
                                     '',c='k')
    rare_freq_psth_ts[0].set_size_inches(3.5,2.5)
    plot_ts_var(x_ser,all_data,'k',rare_freq_psth_ts[1])
    rare_freq_psth_ts[1].axvline(0,c='k',ls='--')
    rare_freq_psth_ts[1].locator_params(axis='both', nbins=4)
    rare_freq_psth_ts[1].tick_params(axis='both', which='major', labelsize=14)
    rare_freq_psth_ts[0].show()
    rare_freq_psth_ts[0].savefig('rare_freq_diff_psth_ts.svg')

    rare_freq_psth_plot = plot_psth((all_data),'Time from A',[-2,3],cmap='bwr',
                                    cbar_label='zscored firing rate\n( rare - frequent)',vmax=2.5)
    rare_freq_psth_plot[0].set_size_inches(6,4)
    [rare_freq_psth_plot[1].axhline(i,c='k',ls='--') for i in np.cumsum([e.shape[0] for e in data[1:]])]
    rare_freq_psth_plot[1].axvline(0,c='k',ls='--')
    rare_freq_psth_plot[1].tick_params(axis='both', which='major', labelsize=14)
    rare_freq_psth_plot[1].locator_params(axis='both', nbins=4)
    rare_freq_psth_plot[0].savefig('rare_freq_diff_psth.svg')


    rare_freq_psth_plot[0].show()
