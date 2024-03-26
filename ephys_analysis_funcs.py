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
import scipy
from scipy.signal import savgol_filter
# import scicomap as sc
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import svm
import multiprocessing
from functools import partial
from scipy.stats import ttest_ind
from matplotlib.colors import LogNorm
# import seaborn as sns
# from os import register_at_fork


try:matplotlib.use('TkAgg')
except ImportError:pass


def init_pool_processes():
    np.random.seed()


def posix_from_win(path:str) -> Path:
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


def get_spikedir(recdir,sorter='kilosort2_5',sorting_dir_name='sorting') -> Path:
    """
    Returns the path to the spike sorting output directory for the given recording directory and sorter.

    Parameters:
    recdir (Path): The path to the recording directory.
    sorter (str): The spike sorter name (default is 'kilosort2_5').

    Returns:
    Path: The path to the spike sorting output directory.
    """
    recdir = Path(recdir)

    while not any([sorting_dir_name in str(path) for path in list(recdir.iterdir())]):
        recdir = recdir.parent

    spikedir = recdir/sorting_dir_name/sorter/'sorter_output'
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
    """
    A function to load spike times and clusters from the given paths.

    :param spike_times_path: Path to the spike times file
    :param spike_clusters_path: Path to the spike clusters file
    :param parent_dir: Optional parent directory for the spike times and clusters paths
    :return: Tuple containing spike times and spike clusters arrays
    """
    spike_times_path,spike_clusters_path = Path(spike_times_path), Path(spike_clusters_path)
    assert spike_times_path.suffix == '.npy' and spike_clusters_path.suffix == '.npy', Warning('paths should be .npy')

    if parent_dir:
        spike_times_path,spike_clusters_path = [parent_dir / path or path for path in [spike_times_path,spike_clusters_path]
                                                if not path.is_absolute()]
    spike_times = np.load(spike_times_path)
    spike_clusters = np.load(spike_clusters_path)

    return spike_times, spike_clusters


def cluster_spike_times(spike_times:np.ndarray, spike_clusters:np.ndarray)->dict:
    assert spike_clusters.shape == spike_times.shape, Warning('spike times/ cluster arrays need to be same shape')
    cluster_spike_times_dict = {}
    for i in tqdm(np.unique(spike_clusters),desc='getting session cluster spikes times',
                  total=len(np.unique(spike_clusters)),disable=True):
        cluster_spike_times_dict[i] = spike_times[spike_clusters == i]
    return cluster_spike_times_dict


def get_spike_times_in_window(event_time:int,spike_time_dict:dict, window:[list | np.ndarray],fs):
    """
    Get spike times in a specified window for a given event.

    Parameters:
    - event_time: int
    - spike_time_dict: dict
    - window: list or np.ndarray
    - fs: unspecified type

    Returns:
    - window_spikes_dict: dict
    """
    window_spikes_dict = {}

    for cluster_id in tqdm(spike_time_dict, desc='getting spike times for event', total=len(spike_time_dict),
                           disable=True):
        all_spikes = (spike_time_dict[cluster_id] - event_time)   # / fs

        window_spikes_dict[cluster_id] = all_spikes[(all_spikes >= window[0]) * (all_spikes <= window[1])]
    return window_spikes_dict


def gen_spike_matrix(spike_time_dict: dict, window, fs):
    precision = np.ceil(np.log10(fs)).astype(int)
    time_cols = np.round(np.arange(window[0],window[1]+1/fs,1/fs),precision)
    spike_matrix = pd.DataFrame(np.zeros((len(spike_time_dict), int((window[1]-window[0])*fs)+1)),
                                index=list(spike_time_dict.keys()),columns=time_cols)
    rounded_spike_dict = {}
    for cluster_id in tqdm(spike_time_dict, desc='rounding spike times for event', total=len(spike_time_dict),
                           disable=True):
        rounded_spike_dict[cluster_id] = np.round(spike_time_dict[cluster_id],precision)
    for cluster_id in tqdm(rounded_spike_dict, desc='assigning spikes to matrix', total=len(spike_time_dict),
                           disable=True):
        spike_matrix.loc[cluster_id][rounded_spike_dict[cluster_id]] = 1
    return spike_matrix


def plot_spike_time_raster(spike_time_dict: dict, ax=None,**pltkwargs):
    if not ax:
        fig,ax = plt.subplots()
    assert isinstance(ax, plt.Axes)
    for cluster_id in tqdm(spike_time_dict, desc='plotting spike times for event', total=len(spike_time_dict),
                           disable=True):
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


def plot_2d_array_with_subplots(array_2d: np.ndarray, cmap='viridis', cbar_width=0.1, cbar_height=0.8, plot_cbar=True,
                                extent=None,aspect=1.0,vcenter=None,plot=None,
                                **im_kwargs) -> (plt.Figure, plt.Axes):
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
    if not plot:
        fig, ax = plt.subplots()
    else:
        fig,ax = plot
    # Create the matshow plot on the specified axis with the provided colormap and extent
    if vcenter:
        cax = ax.imshow(array_2d, cmap=cmap, extent=extent,
                         norm=matplotlib.colors.TwoSlopeNorm(vcenter=vcenter, )) # vmin=vmin,vmax=vmax
    else:
        cax = ax.imshow(array_2d, cmap=cmap, extent=extent,**im_kwargs)
    # ax.set_aspect(array_2d.shape[1]/(array_2d.shape[0]*100*aspect))
    ax.set_aspect('auto')

    # Add a color bar legend using fig.colorbar with explicit width and height
    if plot_cbar:
        cbar = fig.colorbar(cax, fraction=cbar_width, pad=0.05, aspect=cbar_height,)
    else:
        cbar=None
    # Show the plot
    # plt.show()
    return fig, ax, cbar


# def fixed_cmap(cmap):
#     div_map = sc.ScicoDiverging(cmap=cmap)
#     div_map.unif_sym_cmap(lift=15,
#                           bitonic=False,
#                           diffuse=True)


def predict_1d(models, t_ser,y_lbl=0):
    # models = models
    # with multiprocessing.Pool() as pool:
    #     results = tqdm(pool.imap())
    prediction_ts = np.array([m.predict(t_ser.T) for m in models])
    if isinstance(y_lbl,list):
        assert len(y_lbl) == prediction_ts.shape[0]
        _arr = np.array(row==lbl for row, lbl in zip(prediction_ts,y_lbl))
        prediction_ts = _arr
    return prediction_ts


def load_session_pkls(pkl_path):
    with open(pkl_path, 'rb') as pklfile:
        sess_obj = pickle.load(pklfile)
        # sess_obj.sound_event_dict = {}
        # sessions[sess_obj.sessname] = sess_obj
    return sess_obj


class SessionSpikes:
    def __init__(self, spike_times_path: [Path| str], spike_clusters_path: [Path| str], sess_start_time: float,
                 parent_dir=Path(''), fs=3e4, resample_fs=1e3):
        """
        Initialize the SpikeSorter class.

        Parameters:
            spike_times_path (Path|str): The path to the spike times file.
            spike_clusters_path (Path|str): The path to the spike clusters file.
            sess_start_time (float): The start time of the session.
            parent_dir (optional): The parent directory. Defaults to None.
            fs (float): The sampling frequency. Defaults to 30000.0.
            resample_fs (float): The resampled sampling frequency. Defaults to 1000.0.

        Returns:
            None
        """
        self.spike_times_path = spike_times_path
        self.spike_clusters_path = spike_clusters_path
        self.start_time = sess_start_time
        self.fs = fs
        self.new_fs = resample_fs
        self.spike_times, self.clusters = load_spikes(spike_times_path,spike_clusters_path,parent_dir)
        self.spike_times = self.spike_times/fs + sess_start_time  # units seconds

        self.cluster_spike_times_dict = cluster_spike_times(self.spike_times, self.clusters)
        self.bad_units = set()
        # self.curate_units()
        # self.curate_units_by_rate()
        if (parent_dir/'good_units.csv').is_file():
            self.good_units =pd.read_csv(parent_dir/'good_units.csv').iloc[:,0].to_list()
            unit_ids = list(self.cluster_spike_times_dict.keys())
            for unit in unit_ids:
                if unit not in self.good_units:
                    self.cluster_spike_times_dict.pop(unit)
            print(f'good units: {self.good_units}')
        else:
            self.curate_units_by_rate()
        self.units = list(self.cluster_spike_times_dict.keys())
        self.event_spike_matrices = multiprocessing.Manager().dict()
        self.event_cluster_spike_times = multiprocessing.Manager().dict()

    def get_event_spikes(self,event_times: [list|np.ndarray|pd.Series], event_name: str, window: [list| np.ndarray]):
        # event_times2use = [event_time for event_time in event_times
        #                    if f'{event_name}_{event_time}' not in list(self.event_cluster_spike_times.keys())]
        # event_keys2use = [f'{event_name}_{event_time}' for event_time in event_times2use]
        # with multiprocessing.Pool() as pool:
        #     spikes_in_window_func = partial(get_spike_times_in_window,spike_time_dict=self.cluster_spike_times_dict,
        #                                     window=window,fs=self.new_fs)
        #     results = list(tqdm(pool.imap(spikes_in_window_func, event_times2use)))
        #     for ri, res in enumerate(results):
        #         self.event_cluster_spike_times[f'{event_name}_{event_times2use[ri]}'] = res
        #
        #     gen_matrix_func = partial(gen_spike_matrix,window=window,fs=self.new_fs)
        #     gen_matrix_iter = [self.event_cluster_spike_times[event_key] for event_key in event_keys2use]
        #     results = tqdm(pool.imap(gen_matrix_func, gen_matrix_iter))
        #     for ri, res in results:
        #         self.event_spike_matrices[event_keys2use[ri]] = res
        for event_time in event_times:
            if f'{event_name}_{event_time}' not in list(self.event_cluster_spike_times.keys()):
                self.event_cluster_spike_times[f'{event_name}_{event_time}'] = get_spike_times_in_window(event_time,self.cluster_spike_times_dict,window,self.new_fs)
            if f'{event_name}_{event_time}' not in list(self.event_spike_matrices.keys()):
                self.event_spike_matrices[f'{event_name}_{event_time}'] = gen_spike_matrix(self.event_cluster_spike_times[f'{event_name}_{event_time}'],
                                                                                           window,self.new_fs)

    def curate_units(self):
        # self.bad_units = set()
        for unit in self.cluster_spike_times_dict:
            d_spike_times = np.diff(self.cluster_spike_times_dict[unit])
            if np.mean(d_spike_times>10)>0.05:
                # self.cluster_spike_times_dict.pop(unit)
                self.bad_units.add(unit)
        print(f'popped units {self.bad_units}, remaining units: {len(self.cluster_spike_times_dict) - len(self.bad_units)}/{len(self.cluster_spike_times_dict)}')
        for unit in self.bad_units:
            self.cluster_spike_times_dict.pop(unit)

    def curate_units_by_rate(self):
        # self.bad_units = set()
        for unit in self.cluster_spike_times_dict:
            d_spike_times = np.diff(self.cluster_spike_times_dict[unit])
            if np.mean(d_spike_times) > (1/0.05):
                # self.cluster_spike_times_dict.pop(unit)
                self.bad_units.add(unit)
        print(f'popped units {self.bad_units}, remaining units: {len(self.cluster_spike_times_dict) - len(self.bad_units)}/{len(self.cluster_spike_times_dict)}')
        for unit in self.bad_units:
            self.cluster_spike_times_dict.pop(unit)


def plot_decoder_accuracy(decoder_accuracy, labels, fig=None,ax=None, start_loc=0,
                          n_features=None,plt_kwargs=None) -> [plt.Figure,plt.Axes]:
    if not isinstance(ax, plt.Axes):
        fig, ax = plt.subplots()

    if not plt_kwargs:
        plt_kwargs = {}
    decoder_accuracy = np.array(decoder_accuracy)
    for li, label_res in enumerate(decoder_accuracy):
        ax.scatter(simple_beeswarm2(label_res.flatten(), width=0.1) + start_loc+li, label_res.flatten(),
                   label=labels[li], alpha=0.5,**plt_kwargs)
        ax.scatter(li+start_loc,np.mean(label_res),marker='^',c='k',s=20)
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
    rate_mat.index = sess_spike_obj.units
    # rate_mat = rate_mat.assign(m=rate_mat.loc[:,timedelta(0,0):timedelta(0,0.2)].mean(axis=1)
    #                            ).sort_values('m',ascending=False).drop('m', axis=1)
    # sorted_arrs = [e.iloc[rate_mat.index.to_series()] for e in all_event_list]
    sorted_arrs = all_event_list
    # rate_mat = rate_mat.sort_values(by=timedelta(0, 0.2), ascending=False

    return np.array(sorted_arrs),rate_mat,rate_mat.index


def plot_psth(psth_rate_mat, event_lbl,window, title='', cbar_label=None,cmap='hot',**im_kwargs):
    if not cbar_label:
        cbar_label = 'firing rate (Hz)'
    fig, ax, cbar = plot_2d_array_with_subplots(psth_rate_mat, cmap=cmap, extent=[window[0], window[1], psth_rate_mat.shape[0], 0],
                                                cbar_height=50,aspect=0.1,**im_kwargs)
    ax.set_ylabel('unit number', fontsize=14)
    ax.set_xlabel(f'time from {event_lbl} onset (s)', fontsize=14)
    ax.set_title(title)
    if cbar:
        cbar.ax.set_ylabel(cbar_label,rotation=270, labelpad=5)
    for t in np.arange(0, 1, 1):
        ax.axvline(t, ls='--', c='white', lw=1)

    return fig,ax,cbar


def plot_psth_ts(psth_mat, x_ser, x_lbl='', y_lbl='', title='', plot=None,**plt_kwargs) -> [plt.Figure,plt.Axes]:
    if not plot:
        plot=plt.subplots()
    plot[1].plot(x_ser,psth_mat.mean(axis=0),**plt_kwargs)
    plot[1].set_title(title)
    plot[1].set_xlabel(x_lbl)
    plot[1].set_ylabel(y_lbl)
    return plot


def plot_ts_var(x_ser:np.ndarray|pd.Series, y_ser:np.ndarray|pd.Series,colour:str,plt_ax:plt.Axes):
    if isinstance(x_ser,pd.Series):
        x_ser = x_ser.to_numpy()
    if isinstance(y_ser,pd.Series):
        y_ser = y_ser.to_numpy()

    rand_npdample = [y_ser[np.random.choice(y_ser.shape[0], y_ser.shape[0], replace=True), :].mean(axis=0)
                     for i in range(500)]
    rand_npsample = np.array(rand_npdample)
    ci = np.apply_along_axis(mean_confidence_interval, axis=0, arr=rand_npsample).astype(float)

    plt_ax.fill_between(x_ser,ci[1], ci[2],alpha=0.1,fc=colour)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.nanmean(a),a.std()  # a.std()
    h = se * scipy.stats.t.ppf((1 - confidence) / 2., n-1)
    return m, m-h, m+h


def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))


def multi_pearsonr(x, y):
    xmean = x.mean(axis=1)
    ymean = y.mean()
    xm = x - xmean[:,None]
    ym = y - ymean
    normxm = np.linalg.norm(xm, axis=1)
    normym = np.linalg.norm(ym)
    return np.clip(np.dot(xm/normxm[:,None], ym/normym), -1.0, 1.0)


class SoundEvent:
    def __init__(self,idx,times,lbl):
        self.idx = idx
        self.times = times
        self.lbl = lbl

        self.psth = None
        self.psth_plot = None

    def get_psth(self,sess_spike_obj:SessionSpikes,window,title='', redo_psth=False, redo_psth_plot=False,
                 baseline_dur=0.25,zscore_flag=False, iti_zscore=None,reorder_idxs=None):

        if not self.psth or redo_psth:
            self.psth = get_event_psth(sess_spike_obj, self.idx, self.times, window, self.lbl,
                                       baseline_dur=baseline_dur,zscore_flag=zscore_flag,iti_zscore=iti_zscore)
        if not self.psth_plot or redo_psth or redo_psth_plot:

            self.psth_plot = plot_psth(self.psth[1],self.lbl,window,title=title)
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
            n_runs = kwargs.get('n_runs',1000)
        else:
            n_runs = kwargs.get('n_runs',1000)
        # cv_folds = kwargs.get('cv_folds',None)
        # if not kwargs.get('balance',False):
        if not isinstance(self.predictors,(list|tuple)):
            preds,feats = [self.predictors]*n_runs, [self.features]*n_runs
        else:
            n_run_pred = [balance_predictors(self.predictors,self.features) for i in range(n_runs)]
            preds = [run[0] for run in n_run_pred]
            feats = [run[1] for run in n_run_pred]
        with multiprocessing.Pool(initializer=init_pool_processes) as pool:
            results = list(tqdm(pool.starmap(partial(run_decoder, model=self.model_name,
                                                  **dec_kwargs),
                                             zip(preds,feats)), total=n_runs))
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

    def init_spike_obj(self,spike_times_path, spike_cluster_path, start_time, parent_dir,):
        """
        Initializes the spike_obj attribute with a SessionSpikes object.

        Args:
            spike_times_path (str): The file path to the spike times.
            spike_cluster_path (str): The file path to the spike clusters.
            start_time (int): The start time of the spikes.
            parent_dir (str): The parent directory of the spike files.
        """
        self.spike_obj = SessionSpikes(spike_times_path, spike_cluster_path, start_time, parent_dir)

    def init_sound_event_dict(self,sound_write_path_stem,sound_event_labels,tone_space=5,
                              harpbin_dir=Path(r'X:\Dammy\harpbins'), normal=None,new_normal=None):
        """
        Initialize a sound event dictionary using the given parameters.

        Args:
            sound_write_path_stem: The path stem for the sound writes.
            sound_event_labels: Labels for the sound events.
            tone_space: The space between tones.
            harpbin_dir: The directory for the harp bins.
            normal: The normal pattern values.
            new_normal: The new normal pattern values.

        Returns:
            None
        """
        sound_writes = load_sound_bin(sound_write_path_stem,bin_dir=harpbin_dir)
        # assign sounds to trials
        sound_writes['Trial_Number'] = np.full_like(sound_writes.index, -1)
        sound_writes_diff = sound_writes['Time_diff'] = sound_writes['Timestamp'].diff()
        long_dt = np.squeeze(np.argwhere(sound_writes_diff > 1))
        for n, idx in enumerate(long_dt):
            sound_writes.loc[idx:, 'Trial_Number'] = n
        sound_writes['Trial_Number'] = sound_writes['Trial_Number'] + 1
        if '240216a' in sound_write_path_stem:
            sound_writes = sound_writes[sound_writes['Trial_Number']>30]
        sound_writes_diff = sound_writes['Time_diff'] = sound_writes['Timestamp'].diff()
        sound_writes['Payload_diff'] = sound_writes['Payload'].diff()
        long_dt = np.squeeze(np.argwhere(sound_writes_diff > 1))


        # if passive:
        #     all_pip_idxs = [e for e in sound_writes['Payload'].unique() if e > 7]
        # else:
        all_pip_idxs = normal
        # base_pip_idx = min(all_pip_idxs)
        base_pip_idx = normal[0] - (-2 if (normal[0]-normal[1]>0) else 2)  # base to pip gap
        non_base_pips = sound_writes.query('Payload > 8 & Payload != @base_pip_idx')['Payload'].unique()
        sound_writes['pattern_pips'] = np.any((sound_writes['Payload_diff']==(-2 if (normal[0]-normal[1]>0) else 2),
                                              sound_writes['Payload'].isin(non_base_pips)),axis=0)
        sound_writes['pattern_start'] = sound_writes['pattern_pips']*sound_writes['pattern_pips'].diff()>0
        sound_writes['pip_counter'] = np.zeros_like(sound_writes['pattern_start']).astype(int)
        for i in sound_writes.query('pattern_start == True').index:
            sound_writes.loc[i:i+3,'pip_counter'] = np.cumsum(sound_writes['pattern_pips'].loc[i:i+3])\
                                                    *sound_writes['pattern_pips'].loc[i:i+3]
        print(base_pip_idx)

        # get trial start times
        long_dt_df = sound_writes.iloc[long_dt]
        trial_start_times = long_dt_df['Timestamp']
        # pattern_idxs = [a_idx + i * tone_space for i in range(4)]
        pattern_idxs = [pip for pip in all_pip_idxs if pip != base_pip_idx]
        sound_events = pattern_idxs + [3, base_pip_idx,-1]
        sound_event_labels = ['A', 'B', 'C', 'D', 'X', 'base','trial_start']

        # get baseline pip
        sound_writes['idx_diff'] = sound_writes['Payload'].diff()
        writes_to_use = np.all([sound_writes['Payload'] == base_pip_idx, sound_writes['idx_diff'] == 0,
                                sound_writes['Trial_Number'] > 10, sound_writes['Time_diff'] < 0.3], axis=0)
        base_pips_times = base_pip_times_same_prev = sound_writes[sorted(writes_to_use)]['Timestamp']
        if base_pips_times.empty:
            base_pips_times = sound_writes[sound_writes['Payload']==base_pip_idx]['Timestamp']

        # event sound times
        if sound_writes[sound_writes['Payload'] == 3].empty:
            sound_events.remove(3), sound_event_labels.remove('X')
        all_sound_times = [sound_writes[sound_writes['Payload'] == i].drop_duplicates(subset='Trial_Number')['Timestamp']
                           for i in sound_events if i not in [base_pip_idx,-1]]
        if not base_pips_times.empty:
            idxss = np.random.choice(base_pips_times.index, min(base_pips_times.shape[0],
                                                                np.max([len(e) for e in all_sound_times])),
                                     replace=False)
            all_sound_times.append(sound_writes.loc[idxss,'Timestamp'])
        else:
            sound_event_labels.remove('base'), sound_events.remove(base_pip_idx)
        all_sound_times.append(trial_start_times)
        if new_normal:
            new_As = sound_writes[sound_writes['pattern_start']]['Payload'].unique()[1:]
            for new_i, new_A in enumerate(new_As):
                if new_A > 8:
                    all_sound_times.append(sound_writes.query('Payload==@new_A & pattern_start==True')['Timestamp'])
                    sound_events.append(new_A)
                    sound_event_labels.append(f'A{new_i+1}')

        assert len(sound_events) == len(all_sound_times), Warning('sound event and labels must be equal length')
        for event, event_times, event_lbl in zip(sound_events, all_sound_times, sound_event_labels):
            self.sound_event_dict[event_lbl] = SoundEvent(event, event_times, event_lbl)

            self.sound_event_dict[event_lbl].trial_nums = sound_writes.loc[event_times.index]['Trial_Number'].values

    def get_sound_psth(self,psth_window,baseline_dur=0.25,zscore_flag=False,redo_psth=False,to_exclude=(None,),
                       use_iti_zscore=True,**psth_kwargs):
        """
        Generate the peristimulus time histogram (PSTH) for sound events.

        Args:
            psth_window: The window (in seconds) over which to calculate the PSTH.
            baseline_dur: The duration (in seconds) of the baseline period for z-score calculation.
            zscore_flag: A boolean indicating whether to z-score the PSTH.
            redo_psth: A boolean indicating whether to recalculate the PSTH.
            to_exclude: A tuple of sound events to exclude from the PSTH calculation.
            use_iti_zscore: A boolean indicating whether to use inter-trial-interval (ITI) z-score for baseline correction.
            **psth_kwargs: Additional keyword arguments to be passed to get_event_psth.

        Returns:
            None
        """
        if zscore_flag:
            baseline_dur = 0
        self.get_iti_baseline()
        for e in tqdm(self.sound_event_dict,desc='get sound psth',total=len(self.sound_event_dict)):
            if e not in to_exclude:
                # iti_zscore = [ee[self.sound_event_dict[e].trial_nums-1] for ee in self.iti_zscore]  # if using mean why not use all trials
                self.sound_event_dict[e].get_psth(self.spike_obj,psth_window,title=f'session {self.sessname}',
                                                  baseline_dur=baseline_dur, zscore_flag=zscore_flag,
                                                  redo_psth=redo_psth,
                                                  iti_zscore=(self.iti_zscore if use_iti_zscore else None),
                                                  **psth_kwargs)
                # self.sound_event_dict[e].psth_plot[0].show()

    def reorder_psth(self,ref_name,reorder_list):
        if ref_name in list(self.sound_event_dict.keys()):
            for e in reorder_list:
                reord = self.sound_event_dict[e].psth[1].loc[self.sound_event_dict[ref_name].psth[2]]
                self.sound_event_dict[e].psth = (self.sound_event_dict[e].psth[0],reord,self.sound_event_dict[e].psth[2])

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
        self.get_local_rate()

    def get_n_since_last(self):
        self.td_df['n_since_last'] = np.arange(self.td_df.shape[0])
        since_last = self.td_df[self.td_df['Tone_Position'] == 0].index
        for t,tt in zip(since_last,np.pad(since_last,[1,0])):
            self.td_df.loc[tt+1:t,'n_since_last'] = self.td_df.loc[tt+1:t,'n_since_last']-tt
        self.td_df.loc[t+1:, 'n_since_last'] = self.td_df.loc[t+1:, 'n_since_last'] - t

    def get_local_rate(self,window=5):
        self.td_df['local_rate'] = self.td_df['Tone_Position'].rolling(window=window).mean()


def get_predictor_from_psth(sess_obj:Session,event_key,psth_window, new_window, mean=np.mean,mean_axis=2) -> np.ndarray:
    """
    Generate a predictor array from a PSTH (peristimulus time histogram) using the provided session object, event key,
    PSTH window, new window, and optional mean and mean axis parameters. Return the predictor array as a NumPy array.
    """
    td_start, td_end = [timedelta(0, t) for t in new_window]
    event_arr = sess_obj.sound_event_dict[event_key].psth[0]
    event_arr_2d = event_arr.reshape(-1, event_arr.shape[2])
    predictor_rate_mat = gen_firing_rate_matrix(pd.DataFrame(event_arr_2d,
                                                             columns=np.linspace(psth_window[0], psth_window[1],
                                                                                 event_arr_2d.shape[1])),
                                                baseline_dur=0,).loc[:, td_start:td_end]
    predictor = predictor_rate_mat.values.reshape((-1, event_arr.shape[1], predictor_rate_mat.shape[1]))  # units x trials x t steps
    predictor = zscore_w_iti(predictor,sess_obj.iti_zscore[0],sess_obj.iti_zscore[1])
    if mean:
        predictor = mean(predictor,axis=mean_axis)

    return predictor


def zscore_w_iti(rate_arr,iti_mean,iti_std):
    iti_mean[iti_mean == 0] = np.nan
    iti_std[iti_std == 0] = np.nan
    # ratemat_arr3d = np.transpose((ratemat_arr3d.T - iti_zscore[0].T)/iti_zscore[1].T)
    # for ti in range(iti_zscore[0].shape[0]):
    for ui in range(iti_mean.shape[1]):
        rate_arr[:,ui] = (rate_arr[:,ui] - np.nanmean(iti_mean, axis=0)[ui]) / \
                            np.nanmean(iti_std, axis=0)[ui]

    return rate_arr


def balance_predictors(list_predictors,list_features)->[[np.ndarray],[np.ndarray]]:
    min_pred_len = min([e.shape[0] for e in list_predictors])
    idx_subsets = [np.random.choice(e.shape[0],min_pred_len,replace=False) for e in list_predictors]
    # assert len(np.unique(idx_subset)) == min_pred_len
    predictors = [e[idxs] for e,idxs in zip(list_predictors,idx_subsets)]
    features = [e[idxs] for e,idxs in zip(list_features,idx_subsets)]

    return np.vstack(predictors),np.hstack(features)


def find_ramp_start(signal:np.ndarray,):
    t_peak = np.argmax(signal)
    d2_signal = np.diff(savgol_filter(np.diff(signal[:t_peak]),50,2))
    return d2_signal[d2_signal>d2_signal.mean()+2*d2_signal.std()]
