import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path, PurePosixPath, PureWindowsPath, PosixPath
from tqdm import tqdm
import json
from datetime import timedelta
import pickle
from copy import copy
from scipy.signal import convolve
from scipy.signal.windows import gaussian
from scipy.stats.mstats import zscore
from decoder import run_decoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from copy import deepcopy as copy
# import scicomap as sc
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import svm
import multiprocessing
from functools import partial
from scipy.stats import ttest_ind
from matplotlib.colors import LogNorm
# import seaborn as sns
from os import register_at_fork


try:matplotlib.use('TkAgg')
except ImportError:pass


def init_pool_processes():
    np.random.seed()


def posix_from_win(path:str) -> Path:
    if ':\\' in path:
        path_bits = PureWindowsPath(path).parts
        path_bits = [bit for bit in path_bits if '\\' not in bit]
        return Path(PurePosixPath(*path_bits))


def get_spikedir(recdir,sorter='kilosort2_5') -> Path:
    recdir = Path(recdir)

    while not any(['sorting' in str(path) for path in list(recdir.iterdir())]):
        recdir = recdir.parent

    spikedir = recdir/'sorting'/sorter/'sorter_output'
    assert spikedir.is_dir(), Warning(f'spikedir {spikedir} not a dir, check path and sorter name')

    return spikedir


def gen_metadata(data_topology_filepath: [str,Path],ceph_dir,harp_bin_dir=r'X:\Dammy\harpbins'):
    if not isinstance(data_topology_filepath,Path):
        data_topology_filepath = Path(data_topology_filepath)
    data_topology = pd.read_csv(data_topology_filepath)
    for idx,sess in data_topology.iterrows():
        e_dir = ceph_dir/posix_from_win(sess['ephys_dir'])
        bin_stem = sess['beh_bin_stem']
        dir_files = list(e_dir.iterdir())
        if 'metadata.json' not in dir_files or True:
            harp_bin_dir = Path(harp_bin_dir)
            harp_writes = pd.read_csv(harp_bin_dir/f'{bin_stem}_write_data.csv')
            trigger_time = harp_writes[harp_writes['DO3']==True]['Times'][0]
            metadata = {'trigger_time': trigger_time}
            with open(e_dir/'metadata.json','w') as jsonfile:
                json.dump(metadata,jsonfile)


def load_spikes(spike_times_path:[Path|str], spike_clusters_path:[Path|str], parent_dir=None):
    spike_times_path,spike_clusters_path = Path(spike_times_path), Path(spike_clusters_path)
    assert spike_times_path.suffix == '.npy' and spike_clusters_path.suffix == '.npy', Warning('paths should be .npy')

    if parent_dir:
        spike_times_path,spike_clusters_path = [parent_dir / path or path for path in [spike_times_path,spike_clusters_path]
                                                if not path.is_absolute()]
    spike_times = np.load(spike_times_path)
    spike_clusters = np.load(spike_clusters_path)

    return spike_times, spike_clusters


def cluster_spike_times(spike_times:np.ndarray, spike_clusters:np.ndarray):
    assert spike_clusters.shape == spike_times.shape, Warning('spike times/ cluster arrays need to be same shape')
    cluster_spike_times_dict = {}
    for i in tqdm(np.unique(spike_clusters),desc='getting session cluster spikes times',
                  total=len(np.unique(spike_clusters))):
        cluster_spike_times_dict[i] = spike_times[spike_clusters == i]
    return cluster_spike_times_dict


def get_spike_times_in_window(spike_time_dict:dict,event_time:int, window:[list | np.ndarray],fs):
    window_spikes_dict = {}

    for cluster_id in tqdm(spike_time_dict, desc='getting spike times for event', total=len(spike_time_dict)):
        all_spikes = (spike_time_dict[cluster_id] - event_time)   # / fs

        window_spikes_dict[cluster_id] = all_spikes[(all_spikes >= window[0]) * (all_spikes <= window[1])]
    return window_spikes_dict


def gen_spike_matrix(spike_time_dict: dict, window, fs):
    precision = np.ceil(np.log10(fs)).astype(int)
    time_cols = np.round(np.arange(window[0],window[1]+1/fs,1/fs),precision)
    spike_matrix = pd.DataFrame(np.zeros((len(spike_time_dict), int((window[1]-window[0])*fs)+1)),
                                index=list(spike_time_dict.keys()),columns=time_cols)
    rounded_spike_dict = {}
    for cluster_id in tqdm(spike_time_dict, desc='rounding spike times for event', total=len(spike_time_dict)):
        rounded_spike_dict[cluster_id] = np.round(spike_time_dict[cluster_id],precision)
    for cluster_id in tqdm(rounded_spike_dict, desc='assigning spikes to matrix', total=len(spike_time_dict)):
        spike_matrix.loc[cluster_id][rounded_spike_dict[cluster_id]] = 1
    return spike_matrix


def plot_spike_time_raster(spike_time_dict: dict, ax=None,**pltkwargs):
    if not ax:
        fig,ax = plt.subplots()
    assert isinstance(ax, plt.Axes)
    for cluster_id in tqdm(spike_time_dict, desc='plotting spike times for event', total=len(spike_time_dict)):
        ax.scatter(spike_time_dict[cluster_id], [cluster_id]*len(spike_time_dict[cluster_id]),**pltkwargs)
        ax.invert_xaxis()


def gen_firing_rate_matrix(spike_matrix: pd.DataFrame, bin_dur=0.01, baseline_dur=0.0,
                           zscore_flag=False, gaus_std=0.04) -> pd.DataFrame:
    # print(f'zscore_flag = {zscore_flag}')
    guas_window = gaussian(int(gaus_std/bin_dur),int(gaus_std/bin_dur))
    spike_matrix.columns = pd.to_timedelta(spike_matrix.columns,'s')
    rate_matrix = spike_matrix.T.resample(f'{bin_dur}S').mean().T/bin_dur
    cols = rate_matrix.columns
    rate_matrix = np.array([convolve(row,guas_window,mode='same') for row in rate_matrix.values])
    assert not all([baseline_dur, zscore_flag])
    rate_matrix = pd.DataFrame(rate_matrix,columns=cols)
    if baseline_dur:
        rate_matrix = pd.DataFrame(rate_matrix, columns=cols)
        rate_matrix = rate_matrix.sub(rate_matrix.loc[:,timedelta(0,-baseline_dur):timedelta(0,0)].median(axis=1),axis=0)
    if zscore_flag:
        # rate_matrix = (rate_matrix.T - rate_matrix.mean(axis=1))/rate_matrix.std(axis=1)
        rate_matrix = zscore(rate_matrix,axis=1,)
    rate_matrix = rate_matrix.fillna(0)
    return rate_matrix


def load_sound_bin(bin_stem, bin_dir=Path(r'X:\Dammy\harpbins')):
    all_writes = pd.read_csv(bin_dir/f'{bin_stem}_write_indices.csv')
    return all_writes


def simple_beeswarm2(y, nbins=None, width=1.):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    """
    y = np.asarray(y)
    if nbins is None:
        # nbins = len(y) // 6
        nbins = np.ceil(len(y) / 6).astype(int)

    # Get upper bounds of bins
    x = np.zeros(len(y))

    nn, ybins = np.histogram(y, bins=nbins)
    nmax = nn.max()

    #Divide indices into bins
    ibs = []#np.nonzero((y>=ybins[0])*(y<=ybins[1]))[0]]
    for ymin, ymax in zip(ybins[:-1], ybins[1:]):
        i = np.nonzero((y>ymin)*(y<=ymax))[0]
        ibs.append(i)

    # Assign x indices
    dx = width / (nmax // 2)
    for i in ibs:
        yy = y[i]
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(yy)]
            a = i[j::2]
            b = i[j+1::2]
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x


def point_cloud(x_loc,y_ser, spread=0.1):
    return np.random.uniform(low=x_loc - spread, high=x_loc + spread, size=len(y_ser))
    # return np.random.normal(loc=x_loc, scale=spread, size=len(y_ser))


def plot_2d_array_with_subplots(array_2d: np.ndarray, cmap='viridis', cbar_width=0.03, cbar_height=0.8,
                                extent=None,aspect=1.0, vmin=None,vmax=None,vcenter=None) -> (plt.Figure, plt.Axes):
    """
    Create a matshow plot with a color bar legend for a 2D array using plt.subplots.

    Parameters:
    - array_2d (list of lists or numpy.ndarray): The 2D array to be plotted.
    - cmap (str, optional): The colormap to be used for the matshow plot. Default is 'viridis'.
    - cbar_width (float, optional): The width of the color bar as a fraction of the figure width.
                                    Default is 0.03.
    - cbar_height (float, optional): The height of the color bar as a fraction of the figure height.
                                     Default is 0.8.
    - extent (list or None, optional): The extent (left, right, bottom, top) of the matshow plot.
                                       If None, the default extent is used.
    """
    # Convert the input array to a NumPy array
    array_2d = np.array(array_2d)

    # Create a figure and axis using plt.subplots
    fig, ax = plt.subplots()

    # Create the matshow plot on the specified axis with the provided colormap and extent
    if vcenter:
        cax = ax.imshow(array_2d, cmap=cmap, extent=extent,
                         norm=matplotlib.colors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin,vmax=vmax))
    else:
        cax = ax.imshow(array_2d, cmap=cmap, extent=extent)
    # ax.set_aspect(array_2d.shape[1]/(array_2d.shape[0]*100*aspect))
    ax.set_aspect('auto')

    # Add a color bar legend using fig.colorbar with explicit width and height
    cbar = fig.colorbar(cax, fraction=cbar_width, pad=0.04, aspect=cbar_height,)

    # Show the plot
    # plt.show()
    return fig, ax, cbar


# def fixed_cmap(cmap):
#     div_map = sc.ScicoDiverging(cmap=cmap)
#     div_map.unif_sym_cmap(lift=15,
#                           bitonic=False,
#                           diffuse=True)


def load_session_pkls(pkl_path):
    with open(pkl_path, 'rb') as pklfile:
        sess_obj = pickle.load(pklfile)
        sess_obj.sound_event_dict = {}
        # sessions[sess_obj.sessname] = sess_obj
    return sess_obj



class SessionSpikes:
    def __init__(self, spike_times_path: [Path| str], spike_clusters_path: [Path| str], sess_start_time: float,
                 parent_dir=None, fs=3e4, resample_fs=1e3):
        self.spike_times_path = spike_times_path
        self.spike_clusters_path = spike_clusters_path
        self.start_time = sess_start_time
        self.fs = fs
        self.new_fs = resample_fs
        self.spike_times, self.clusters = load_spikes(spike_times_path,spike_clusters_path,parent_dir)
        self.spike_times = self.spike_times/fs + sess_start_time  # unnits seconds

        self.cluster_spike_times_dict = cluster_spike_times(self.spike_times, self.clusters)
        self.event_spike_matrices = {}
        self.event_cluster_spike_times = {}

    def get_event_spikes(self,event_times: [list|np.ndarray|pd.Series], event_name: str, window: [list| np.ndarray]):
        for event_time in event_times:
            if f'{event_name}_{event_time}' not in list(self.event_cluster_spike_times.keys()):
                self.event_cluster_spike_times[f'{event_name}_{event_time}'] = get_spike_times_in_window(self.cluster_spike_times_dict,event_time,window,self.new_fs)
            if f'{event_name}_{event_time}' not in list(self.event_spike_matrices.keys()):
                self.event_spike_matrices[f'{event_name}_{event_time}'] = gen_spike_matrix(self.event_cluster_spike_times[f'{event_name}_{event_time}'],
                                                                                           window,self.new_fs)


# def get_event_rate_mat(event_times, event_idx, spike_obj: SessionSpikes, event_window, bin_width=0.1) -> pd.DataFrame:
#     spike_obj.get_event_spikes(event_times, f'{event_idx}', event_window)
#     spikes = spike_obj.event_spike_matrices
#     all_events_list = [spikes[event].values for event in spikes if str(event_idx) == event.split('_')[0]]
#     # all_events_list = [event.values for event in spike_obj.event_spike_matrices.values()]
#     all_events = np.array(all_events_list)  # 3d array: trials x units x time
#     x_tseries = list(spike_obj.event_spike_matrices.values())[0].columns
#
#     # dataset
#     all_events_stacked = pd.DataFrame(np.vstack(all_events_list), columns=x_tseries)
#     all_events_rate_mat = gen_firing_rate_matrix(all_events_stacked, baseline_dur=0)
#     all_events_rate_mat.columns = x_tseries[::int(1/bin_width)]  # needs to be smarter
#     event_rate_mat = all_events_rate_mat
#
#     return event_rate_mat

def plot_decoder_accuracy(decoder_accuracy, labels, fig=None,ax=None, start_loc=0,
                          n_features=None,plt_kwargs=None) -> [plt.Figure,plt.Axes]:
    if not isinstance(ax, plt.Axes):
        fig, ax = plt.subplots()

    if not plt_kwargs:
        plt_kwargs = {}
    decoder_accuracy = np.array(decoder_accuracy)
    for li, label_res in enumerate(decoder_accuracy):
        ax.scatter(simple_beeswarm2(label_res.flatten(), width=0.1) + start_loc+li, label_res.flatten(),
                   label=labels[li], alpha=0.1,**plt_kwargs)
        # ax.scatter(point_cloud(0,label_res.flatten()) + start_loc+li, sorted(label_res.flatten()),
        #            label=labels[li], alpha=0.1,**plt_kwargs)
    ax.set_ylim(0, 1.19)
    ax.set_ylabel('decoder accuracy')

    if not n_features:
        n_features = 2
    ax.axhline(1/n_features, c='k', ls='--')

    ax.set_xticks(np.arange(0, len(labels), 1))
    ax.set_xticklabels(labels)
    ax.legend(loc=1)

    return fig,ax


def get_event_psth(sess_spike_obj:SessionSpikes,event_idx,event_times:[pd.Series,np.ndarray,list],
                   window:[float,float], event_lbl:str, baseline_dur=0.25,zscore_flag=False,iti_zscore=None) -> (np.ndarray,pd.DataFrame):
    if iti_zscore:
        zscore_flag = False
    sess_spike_obj.get_event_spikes(event_times, f'{event_idx}', window)
    all_event_list = [sess_spike_obj.event_spike_matrices[key]
                      for key in sess_spike_obj.event_spike_matrices if str(event_idx) in key.split('_')[0]]
    all_events_stacked = np.vstack(all_event_list)
    # all_event_mean = pd.DataFrame(np.array(all_event_list).mean(axis=0))
    all_event_mean = pd.DataFrame(all_events_stacked)
    all_event_mean.columns = np.linspace(window[0],window[1],all_event_mean.shape[1])

    rate_mat_stacked = gen_firing_rate_matrix(all_event_mean,baseline_dur=baseline_dur,zscore_flag=zscore_flag)   # -all_events.values[:,:1000].mean(axis=1,keepdims=True)
    ratemat_arr3d = rate_mat_stacked.values.reshape((len(all_event_list), -1, rate_mat_stacked.shape[1]))
    if iti_zscore:
        # iti_zscore[0][iti_zscore[0]==0]=np.nan
        # iti_zscore[1][iti_zscore[1]==0]=np.nan
        # ratemat_arr3d = np.transpose((ratemat_arr3d.T - iti_zscore[0].T)/iti_zscore[1].T)
        # for ti in range(iti_zscore[0].shape[0]):
        # for ui in range(iti_zscore[0].shape[1]):
        #     ratemat_arr3d[ui] = (ratemat_arr3d[ui]-np.nanmean(iti_zscore[0],axis=0)[ui])/\
        #                            np.nanmean(iti_zscore[1],axis=0)[ui]  # taking mean mean & std

        # ratemat_arr3d = ((ratemat_arr3d.T - iti_zscore[0].T)/iti_zscore[1].T).T
        ratemat_arr3d = zscore_w_iti(ratemat_arr3d,iti_zscore[0],iti_zscore[1])
        ratemat_arr3d[ratemat_arr3d==np.nan] = 0.0

        # for i, trial in enumerate(ratemat_arr3d):


    rate_mat = pd.DataFrame(ratemat_arr3d.mean(axis=0),
                            columns=rate_mat_stacked.columns)
    rate_mat.assign(m=rate_mat.loc[:,timedelta(0,0):timedelta(0,0.2)].mean(axis=1)).sort_values('m').drop('m', axis=1)
    # rate_mat = rate_mat.sort_values(by=timedelta(0, 0.2), ascending=False)

    return np.array(all_event_list),rate_mat


def plot_psth(psth_rate_mat, event_lbl,title=''):
    fig, ax, cbar = plot_2d_array_with_subplots(psth_rate_mat, cmap='bwr', extent=[-2, 3, psth_rate_mat.shape[0], 0],
                                          cbar_height=15,
                                          aspect=0.1)
    ax.set_ylabel('unit number', fontsize=14)
    ax.set_xlabel(f'time from {event_lbl} onset', fontsize=14)
    ax.set_title(title)
    cbar.ax.set_ylabel('firing rate (Hz)',rotation=270)
    for t in np.arange(0, 1, 1):
        ax.axvline(t, ls='--', c='k', lw=1)

    return fig,ax,cbar


class SoundEvent:
    def __init__(self,idx,times,lbl):
        self.idx = idx
        self.times = times
        self.lbl = lbl

        self.psth = None
        self.psth_plot = None

    def get_psth(self,sess_spike_obj:SessionSpikes,window,title='', redo_psth=False, redo_psth_plot=False,
                 baseline_dur=0.25,zscore_flag=False, iti_zscore=None):
        if not self.psth or redo_psth:
            self.psth = get_event_psth(sess_spike_obj, self.idx, self.times, window, self.lbl,
                                       baseline_dur=baseline_dur,zscore_flag=zscore_flag,iti_zscore=iti_zscore)
        if not self.psth_plot or redo_psth or redo_psth_plot:
            self.psth_plot = plot_psth(self.psth[1],self.lbl,title=title)
            if zscore_flag:
                self.psth_plot[2].ax.set_ylabel('zscored firing rate (au)',rotation=270)

    def save_plot_as_svg(self, figdir=Path(r'X:\Dammy\figures\ephys',), suffix=''):
        """
        Save the plot as an SVG file.

        Parameters:

        """
        filename = figdir/f'{self.lbl}_{suffix}.svg'
        if self.psth_plot:
            self.psth_plot[0].savefig(filename, format='svg')
            print(f"Plot saved as {filename}")
        else:
            print("No plot to save. Call 'plot_psth' first.")


class Decoder:
    def __init__(self,predictors,features,model_name, ):
        self.predictors = predictors
        self.features = features
        self.model_name = model_name
        self.models = None
        self.predictions = None
        self.accuracy = None
        self.fold_accuracy = None
        self.accuracy_plot = None
        self.cm = None
        self.cm_plot = None
        self.prediction_ts = None

    def decode(self,dec_kwargs,**kwargs):
        if not dec_kwargs.get('cv_folds',0):
            n_runs = kwargs.get('n_runs',100)
        else:
            n_runs=100
        # cv_folds = kwargs.get('cv_folds',None)

        with multiprocessing.Pool(initializer=init_pool_processes) as pool:
            results = list(tqdm(pool.imap(partial(run_decoder, features=self.features, model=self.model_name,
                                                  **dec_kwargs),
                                          [self.predictors] * n_runs), total=n_runs))
        # results = []
        # register_at_fork(after_in_child=np.random.seed)
        # with multiprocessing.Pool(initializer=np.random.seed) as pool:
        #     results = list(tqdm(pool.imap(partial(run_decoder, features=self.features, shuffle=to_shuffle,
        #                                           model=self.model_name, cv_folds=cv_folds),
        #                                   [self.predictors] * 1), total=n_runs))
        # for n in range(n_runs):
        #     results.append(run_decoder(self.predictors,self.features,model=self.model_name,**dec_kwargs))
        self.accuracy = [res[0] for res in results]
        self.models = [res[1] for res in results]
        self.fold_accuracy = [list(res[2] for res in results)]
        self.predictions = [list(res[3]) for res in results]

    def map_decoding_ts(self, t_ser, model_i=0,y_lbl=0):
        dec_models = np.array(self.models).flatten()
        # with multiprocessing.Pool() as pool:
        #     results = tqdm(pool.imap())
        self.prediction_ts = np.array([m.predict(t_ser.T) for m in dec_models])
        if isinstance(y_lbl,list):
            assert len(y_lbl) == self.prediction_ts.shape[0]
            _arr = np.array(row==lbl for row, lbl in zip(self.prediction_ts,y_lbl))
            self.prediction_ts = _arr
        return self.prediction_ts

    def plot_decoder_accuracy(self,labels, plt_kwargs=None, **kwargs):
        fig,ax = kwargs.get('plot',(None,None))
        start_loc = kwargs.get('start_loc',0)
        n_features = kwargs.get('n_features',None)

        if len(labels) <= 2:
            self.accuracy_plot = plot_decoder_accuracy(self.fold_accuracy, labels,fig=fig,ax=ax,plt_kwargs=plt_kwargs,
                                                       start_loc=start_loc,n_features=n_features)
        else:
            unique_lbls = np.arange(len(labels))
            y_tests_folds = [np.hstack(ee) for ee in [e[0] for e in self.predictions]]
            y_preds_folds = [np.hstack(ee) for ee in [e[1] for e in self.predictions]]

            lbl_accuracy_list = [[(fold_preds[fold_tests==lbl]==lbl).mean() for lbl in unique_lbls]
                                 for fold_tests, fold_preds in zip(y_tests_folds,y_preds_folds)]
            lbl_accuracy_list = np.array(lbl_accuracy_list).T
            # for lbl_i, (lbl, lbl_acc) in enumerate(zip(labels,lbl_accuracy_list)):
            self.accuracy_plot = plot_decoder_accuracy(lbl_accuracy_list, labels, fig=fig, ax=ax, plt_kwargs=plt_kwargs,
                                                       start_loc=start_loc, n_features=len(labels))
            self.accuracy_plot[1].legend(ncols=len(labels))

        self.accuracy_plot[0].show()

    def plot_confusion_matrix(self,labels,**kwargs):
        y_tests = np.hstack([np.hstack(ee) for ee in [e[0] for e in self.predictions]])
        y_preds = np.hstack([np.hstack(ee) for ee in [e[1] for e in self.predictions]])
        self.cm = confusion_matrix(y_tests,y_preds,normalize='true')
        cm_plot_ = ConfusionMatrixDisplay(self.cm,display_labels=labels,)
        cm_plot__ = cm_plot_.plot(**kwargs)
        self.cm_plot = cm_plot__.figure_,cm_plot__.ax_
        self.cm_plot[1].invert_yaxis()


class Session:
    def __init__(self,sessname,ceph_dir,pkl_dir='X:\Dammy\ephys_pkls',):
        self.iti_zscore = None
        self.td_df = pd.DataFrame()
        self.sessname = sessname
        self.spike_obj = None
        self.sound_event_dict = {}
        self.decoders = {}
        self.ceph_dir = ceph_dir
        # check for pkl
        class_pkl = (Path(pkl_dir)/ f'{sessname}.pkl')
        # if class_pkl.is_file():
        #     with open(class_pkl,'rb') as pklfile:
        #         temp_class = pickle.load(pklfile)
        #     self.sound_event_dict = temp_class.sound_event_dict
        #     self.decoders = temp_class.decoders

    def init_spike_obj(self,spike_times_path, spike_cluster_path, start_time, parent_dir):
        self.spike_obj = SessionSpikes(spike_times_path, spike_cluster_path, start_time, parent_dir)

    def init_sound_event_dict(self,sound_write_path_stem,sound_event_labels,tone_space=5,
                              harpbin_dir=Path(r'X:\Dammy\harpbins') ):
        sound_writes = load_sound_bin(sound_write_path_stem,bin_dir=harpbin_dir)
        # assign sounds to trials
        sound_writes['Trial_Number'] = np.full_like(sound_writes.index, -1)
        sound_writes_diff = sound_writes['Time_diff'] = sound_writes['Timestamp'].diff()
        long_dt = np.squeeze(np.argwhere(sound_writes_diff > 1))
        for n, idx in enumerate(long_dt):
            sound_writes.loc[idx:, 'Trial_Number'] = n
        sound_writes['Trial_Number'] = sound_writes['Trial_Number'] + 1
        all_pip_idxs = [e for e in sound_writes['Payload'].unique() if e > 7]
        base_pip_idx = min(all_pip_idxs)

        # get trial start times
        long_dt_df = sound_writes.iloc[long_dt]
        trial_start_times = long_dt_df['Timestamp']
        a_idx = base_pip_idx + 2
        # pattern_idxs = [a_idx + i * tone_space for i in range(4)]
        pattern_idxs = sorted([pip for pip in all_pip_idxs if pip != base_pip_idx])
        sound_events = pattern_idxs + [3, base_pip_idx,-1]
        sound_event_labels = ['A', 'B', 'C', 'D', 'X', 'base','trial_start']

        # get baseline pip
        sound_writes['idx_diff'] = sound_writes['Payload'].diff()
        writes_to_use = np.all([sound_writes['Payload'] == base_pip_idx, sound_writes['idx_diff'] == 0,
                                sound_writes['Trial_Number'] > 10, sound_writes['Time_diff'] < 0.3], axis=0)
        base_pips_times = base_pip_times_same_prev = sound_writes[writes_to_use]['Timestamp']
        if base_pips_times.empty:
            base_pips_times = sound_writes[sound_writes['Payload']==base_pip_idx]['Timestamp']

        # event sound times
        if sound_writes[sound_writes['Payload'] == 3].empty:
            sound_events.remove(3), sound_event_labels.remove('X')
        all_sound_times = [sound_writes[sound_writes['Payload'] == i]['Timestamp'] for i in sound_events
                           if i not in [base_pip_idx,-1]]
        if not base_pips_times.empty:
            idxss = np.random.choice(base_pips_times.index, min(base_pips_times.shape[0],
                                                                np.max([len(e) for e in all_sound_times])),
                                     replace=False)
            all_sound_times.append(sound_writes.loc[idxss,'Timestamp'])
        else:
            sound_event_labels.remove('base'), sound_events.remove(base_pip_idx)
        all_sound_times.append(trial_start_times)

        assert len(sound_events) == len(all_sound_times), Warning('sound event and labels must be equal length')
        for event, event_times, event_lbl in zip(sound_events, all_sound_times, sound_event_labels):
            self.sound_event_dict[event_lbl] = SoundEvent(event, event_times, event_lbl)

            self.sound_event_dict[event_lbl].trial_nums = sound_writes.loc[event_times.index]['Trial_Number'].values

    def get_sound_psth(self,psth_window,baseline_dur=0.25,zscore_flag=False,redo_psth=False,to_exclude=(None,)):
        if zscore_flag:
            baseline_dur = 0
        self.get_iti_baseline()
        for e in self.sound_event_dict:
            if e not in to_exclude:
                # iti_zscore = [ee[self.sound_event_dict[e].trial_nums-1] for ee in self.iti_zscore]  # if using mean why not use all trials
                self.sound_event_dict[e].get_psth(self.spike_obj,psth_window,title=f'session {self.sessname}',
                                                  baseline_dur=baseline_dur, zscore_flag=zscore_flag,
                                                  redo_psth=redo_psth,iti_zscore=self.iti_zscore)
                self.sound_event_dict[e].psth_plot[0].show()

    def get_iti_baseline(self):
        assert 'trial_start' in list(self.sound_event_dict.keys())
        t_cols = np.linspace(-2,0,2001)
        iti_psth = get_event_psth(self.spike_obj,-1,self.sound_event_dict['trial_start'].times,[-2,0],'iti',
                                  baseline_dur=0,zscore_flag=False)[0]
        iti_rate_mats = [gen_firing_rate_matrix(pd.DataFrame(e,columns=t_cols)).values for e in iti_psth]
        iti_mean = np.mean(iti_rate_mats,axis=2)
        iti_std = np.std(iti_rate_mats,axis=2)
        self.iti_zscore = iti_mean,iti_std

    def save_psth(self,keys='all',figdir=Path(r'X:\Dammy\figures\ephys'),**kwargs):
        if keys == 'all':
            keys2save = list(self.sound_event_dict.keys())
        else:
            keys2save = keys
        for key in keys2save:
            self.sound_event_dict[key].save_plot_as_svg(suffix=self.sessname,figdir=figdir)

    def pickle_obj(self, pkldir=Path(r'X:\Dammy\ephys_pkls')):
        if not pkldir.is_dir():
            pkldir.mkdir()
        with open(pkldir/ f'{self.sessname}.pkl', 'wb') as pklfile:
            to_save = copy(self)
            to_save.spike_obj = None
            pickle.dump(to_save,pklfile)

    def init_decoder(self, decoder_name: str, predictors, features, model_name='svc'):
        self.decoders[decoder_name] = Decoder(predictors,features,model_name)

    def run_decoder(self,decoder_name,labels,plot_flag=False,decode_shuffle_flag=False,dec_kwargs=None,**kwargs):
        self.decoders[decoder_name].decode(dec_kwargs=dec_kwargs,**kwargs)
        if plot_flag:
            self.decoders[decoder_name].plot_decoder_accuracy(labels=labels,)

    def map_preds2sound_ts(self, sound_event, decoders, psth_window, window,map_kwargs={}):
        sound_event_ts = get_predictor_from_psth(self, sound_event, psth_window, window,mean=None,mean_axis=0)
        # with multiprocessing.Pool() as pool:
        #     prediction_ts_list = tqdm()
        prediction_ts_list = [np.array([self.decoders[decoder].map_decoding_ts(trial_ts,**map_kwargs)
                                        for decoder in decoders]) for trial_ts in sound_event_ts]
        prediction_ts = np.array(prediction_ts_list)
        return prediction_ts

    def load_trial_data(self,tdfile_path,td_home, tddir_path=r'H:\data\Dammy'):
        tddir_path = posix_from_win(tddir_path)
        td_path = Path(td_home)/tddir_path/tdfile_path
        self.td_df = pd.read_csv(td_path)
        self.get_n_since_last()
        # self.get_local_rate()

    def get_n_since_last(self):
        self.td_df['n_since_last'] = np.arange(self.td_df.shape[0])
        since_last = self.td_df[self.td_df['Tone_Position'] == 0].index
        for t,tt in zip(since_last,np.pad(since_last,[1,0])):
            self.td_df.loc[tt+1:t,'n_since_last'] = self.td_df.loc[tt+1:t,'n_since_last']-tt
        self.td_df.loc[t+1:, 'n_since_last'] = self.td_df.loc[t+1:, 'n_since_last'] - t

    def get_local_rate(self,window=10):
        self.td_df['local_rate'] = self.td_df['PatternPresentation_Rate'].rolling(window=window).mean()


def get_predictor_from_psth(sess_obj:Session,event_key,psth_window, new_window, mean=np.mean,mean_axis=2) -> np.ndarray:
    td_start, td_end = [timedelta(0, t) for t in new_window]
    event_arr = sess_obj.sound_event_dict[event_key].psth[0]
    event_arr_2d = event_arr.reshape(-1, event_arr.shape[2])
    predictor_rate_mat = gen_firing_rate_matrix(pd.DataFrame(event_arr_2d,
                                                             columns=np.linspace(psth_window[0], psth_window[1],
                                                                                 event_arr_2d.shape[1])),
                                                baseline_dur=0,).loc[:, td_start:td_end]
    predictor = predictor_rate_mat.values.reshape((-1, event_arr.shape[1], predictor_rate_mat.shape[1]))  # units x trials x t steps
    predictor = zscore_w_iti(predictor,sess_obj.iti_zscore[0],sess_obj.iti_zscore[1])

    return predictor


def zscore_w_iti(rate_arr,iti_mean,iti_std):
    iti_mean[iti_mean == 0] = np.nan
    iti_std[iti_std == 0] = np.nan
    # ratemat_arr3d = np.transpose((ratemat_arr3d.T - iti_zscore[0].T)/iti_zscore[1].T)
    # for ti in range(iti_zscore[0].shape[0]):
    for ui in range(iti_mean.shape[1]):
        rate_arr[ui] = (rate_arr[ui] - np.nanmean(iti_mean, axis=0)[ui]) / \
                            np.nanmean(iti_std, axis=0)[ui]

    return rate_arr
