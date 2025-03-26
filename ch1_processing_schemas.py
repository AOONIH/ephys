import numpy as np
import pandas as pd
from pathlib import Path

import scipy
from scipy.stats import zscore
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def butter_highpass(cutoff, fs, order=5,filtype='high'):
    nyq = 0.5 * fs
    if filtype == 'band':
        if isinstance(cutoff, (list,tuple)):
            normal_cutoff = [e/nyq for e in cutoff]
        else:
            print('List of filter needed for bandpass. Not filtering')
            return None
    else:
        normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype=filtype, analog =False)
    return b, a



def butter_filter(data, cutoff, fs, order=3, filtype='high'):
    b, a = butter_highpass(cutoff, fs, order=order,filtype=filtype)
    y = scipy.signal.filtfilt(b, a, data)
    return y


if '__main__' == __name__:
    # load pupil_df
    pupil_df_path = r'X:\Dammy\mouse_pupillometry\full_pupil_dump\DO72_230919_230919_full_pupil.csv'
    pupil_df = pd.read_csv(pupil_df_path)
    pupil_df = pupil_df.rename(columns={'Unnamed: 0':'timestamp'})
    pupil_df['timestamp'] = pupil_df['timestamp']-pupil_df['timestamp'].min()

    pupil_df['dlc_radii_a_filtered_hpass2'] = butter_filter(pupil_df['dlc_radii_a_raw'].interpolate(limit_direction='both'),
                                                     [0.1,2], 90, filtype='band',)
    pupil_df['dlc_radii_a_filtered_hpass4'] = butter_filter(pupil_df['dlc_radii_a_raw'].interpolate(limit_direction='both'),
                                                     [0.1,4], 90, filtype='band',)
    pupil_df['dlc_radii_a_filtered_hpass1'] = butter_filter(pupil_df['dlc_radii_a_raw'].interpolate(limit_direction='both'),
                                                     [0.01,10], 90, filtype='band',)
    # pupil_df['dlc_radii_a_filtered'] = pupil_df['dlc_radii_a_processed'].interpolate(limit_direction='both')

    metrics = ['dlc_radii_a_raw','dlc_radii_a_no_outs', 'dlc_radii_a_int', 'dlc_radii_a_processed',
               'dlc_radii_a_zscored','dlc_radii_a_filtered_hpass4','dlc_radii_a_filtered_hpass2',
               'dlc_radii_a_filtered_hpass1']
    # metrics = ['dlc_radii_a_raw','dlc_radii_a_zscored',]
    # plots = plt.subplots(len(metrics),sharex='all')
    plots = plt.subplots()
    crop_plots = plt.subplots()
    t = int(0e5)
    t_points = int(0.5e5)
    fs = 1 / 90
    t_crop = int(211/fs)
    t_crop_points = int(5/fs)
    for metric,plot,c,lbls in zip(metrics,[plots[1]]*len(metrics),['darkgray','darkred'],['raw','z-scored']):
        plot.plot(np.arange(0,fs*t_points,fs),pupil_df[metric].values[t:t+t_points]-pupil_df[metric].mean(), label=lbls,
                  alpha=.75,c=c,lw=0.5)
        # plot.legend()
        crop_plots[1].plot(np.arange(0,5,fs),pupil_df[metric].values[t_crop:t_crop+t_crop_points]-pupil_df[metric].mean(),
                  label=lbls, alpha=.75,c=c)
        crop_plots[1].legend(loc='upper right')

    # plots[1].set_xlabel('Time (s)')
    # plots[1].set_ylabel('Pupil Size (a.u.)')
    plots[1].axvspan(211, 216,fc='m',alpha=.2)
    plots[0].set_size_inches(6,2)
    plots[0].set_layout_engine('tight')
    plots[0].show()
    crop_plots[0].set_size_inches(6,2)
    crop_plots[0].set_layout_engine('tight')
    crop_plots[0].show()


    # plot signal around outliers
    outliers = pupil_df[pupil_df['dlc_radii_a_isout'].astype(int).diff()!=0].index.values

    # outlier=np.argmin(pupil_df['dlc_radii_a_raw'].values)
    outlier=outliers[300]
    window = 50
    plots = plt.subplots(2)
    metrics = ['dlc_radii_a_raw','dlc_radii_a_no_outs','dlc_radii_a_int',]
    lss = [{'c':'r','ls':':'},{'c':'b','ls':'-'},]
    # lss = [{'c':'r','ls':':'},{'c':'k','ls':'-'},{'c':'b','ls':'-'}]
    for metric,plot,ls in zip(metrics[::-1],[plots[1][0]]*len(metrics),lss):

        # [plot.plot(np.arange(-window,window),pupil_df[metric][outlier-window:outlier+window].values,
        #           c='k',alpha=0.25)
        #  for outlier in outliers]
        # outlier = outliers[100]
        plot.plot(np.arange(-window*fs, window*fs,fs), pupil_df[metric][outlier - window:outlier + window].values,
                  alpha=1,**ls,label=metric)
        plot.axvline(0,c='k',ls='--')
        # plot.legend()
    plots[0].set_size_inches(10,10)
    plots[0].set_layout_engine('tight')
    plots[0].show()

    plot_ax2 = plots[1][1]
    plot_ax2.plot(np.arange(-window*fs, window*fs,fs), pupil_df['dlc_radii_a_zscored'][outlier - window:outlier + window].values,
                  c='#d6b340', alpha=1,ls='--')
    plots[0].set_layout_engine('tight')
    plots[0].set_size_inches(4,2)
    plots[0].show()
    plot_ax2.locator_params(axis='y', nbins=4)
    plot_ax2.set_ylim(np.array(plot_ax2.get_ylim())*0.99)
    plots[1].locator_params(axis='x', nbins=4)
    plots[1].locator_params(axis='y', nbins=3)
    # plot_ax2.set_ylabel('Pupil Size (a.u.)')
