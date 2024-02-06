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
# import scicomap as sc
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import svm
import multiprocessing
from decoder import run_decoder
from functools import partial
from scipy.stats import ttest_ind
from matplotlib.colors import LogNorm
# import seaborn as sns

try:matplotlib.use('TkAgg')
except ImportError:pass


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
    print(f'zscore_flag = {zscore_flag}')
    guas_window = gaussian(int(gaus_std/bin_dur),int(gaus_std/bin_dur))
    spike_matrix.columns = pd.to_timedelta(spike_matrix.columns,'s')
    rate_matrix = spike_matrix.T.resample(f'{bin_dur}S').mean().T/bin_dur
    cols = rate_matrix.columns
    rate_matrix = np.array([convolve(row,guas_window,mode='same') for row in rate_matrix.values])
    rate_matrix = pd.DataFrame(rate_matrix,columns=cols)
    assert not all([baseline_dur, zscore_flag])
    if baseline_dur:
        rate_matrix = rate_matrix.sub(rate_matrix.loc[:,timedelta(0,-baseline_dur):timedelta(0,0)].median(axis=1),axis=0)
    if zscore_flag:
        rate_matrix = (rate_matrix-rate_matrix.mean())/rate_matrix.std
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


def get_event_rate_mat(event_times, event_idx, spike_obj: SessionSpikes, event_window, bin_width=0.1) -> pd.DataFrame:
    spike_obj.get_event_spikes(event_times, f'{event_idx}', event_window)
    spikes = spike_obj.event_spike_matrices
    all_events_list = [spikes[event].values for event in spikes if str(event_idx) == event.split('_')[0]]
    # all_events_list = [event.values for event in spike_obj.event_spike_matrices.values()]
    all_events = np.array(all_events_list)  # 3d array: trials x units x time
    x_tseries = list(spike_obj.event_spike_matrices.values())[0].columns

    # dataset
    all_events_stacked = pd.DataFrame(np.vstack(all_events_list), columns=x_tseries)
    all_events_rate_mat = gen_firing_rate_matrix(all_events_stacked, baseline_dur=0)
    all_events_rate_mat.columns = x_tseries[::int(1/bin_width)]  # needs to be smarter
    event_rate_mat = all_events_rate_mat

    return event_rate_mat


def plot_decoder_accuracy(decoder_accuracy, labels, fig=None,ax=None, start_loc=0,
                          n_features=None) -> [plt.Figure,plt.Axes]:
    if not isinstance(ax,plt.Axes):
        fig,ax = plt.subplots()
    decoder_accuracy = np.array(decoder_accuracy)
    for li, label_res in enumerate(decoder_accuracy):
        ax.scatter(simple_beeswarm2(label_res.flatten(), width=0.1) + start_loc+li, label_res.flatten(),
                   label=labels[li], alpha=0.1)
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
                   window:[float,float], event_lbl:str, baseline_dur=0.25,zscore_flag=False) -> [plt.Figure,plt.Axes,np.ndarray]:

    sess_spike_obj.get_event_spikes(event_times, f'{event_idx}', window)
    all_event_list = [sess_spike_obj.event_spike_matrices[key]
                      for key in sess_spike_obj.event_spike_matrices if str(event_idx) in key.split('_')[0]]
    all_events_stacked = np.vstack(all_event_list)
    # all_event_mean = pd.DataFrame(np.array(all_event_list).mean(axis=0))
    all_event_mean = pd.DataFrame(all_events_stacked)
    all_event_mean.columns = np.linspace(window[0],window[1],all_event_mean.shape[1])
    rate_mat_stacked = gen_firing_rate_matrix(all_event_mean,baseline_dur=baseline_dur,zscore_flag=zscore_flag)   # -all_events.values[:,:1000].mean(axis=1,keepdims=True)
    rate_mat = pd.DataFrame(rate_mat_stacked.values.reshape((len(all_event_list),-1,rate_mat_stacked.shape[1])).mean(axis=0),
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

    return fig,ax


class SoundEvent:
    def __init__(self,idx,times,lbl):
        self.idx = idx
        self.times = times
        self.lbl = lbl

        self.psth = None
        self.psth_plot = None

    def get_psth(self,sess_spike_obj:SessionSpikes,window,title='', redo_psth=False, redo_psth_plot=False,
                 baseline_dur=0.25,zscore_flag=False):
        if not self.psth or redo_psth:
            self.psth = get_event_psth(sess_spike_obj, self.idx, self.times, window, self.lbl,
                                       baseline_dur=baseline_dur,zscore_flag=zscore_flag)
        if not self.psth_plot or redo_psth or redo_psth_plot:
            self.psth_plot = plot_psth(self.psth[1],self.lbl,title=title)

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

    def decode(self,cv_folds=None,**kwargs):
        to_shuffle = kwargs.get('to_shuffle',False)
        n_runs = kwargs.get('n_runs',100)
        # cv_folds = kwargs.get('cv_folds',None)
        print(cv_folds)

        with multiprocessing.Pool() as pool:
            results = list(tqdm(pool.imap(partial(run_decoder, features=self.features, shuffle=to_shuffle,
                                                  model=self.model_name, cv_folds=cv_folds),
                                          [self.predictors] * n_runs), total=n_runs))
        self.accuracy = [res[0] for res in results]
        self.models = [list(res[1] for res in results)]
        self.fold_accuracy = [list(res[2] for res in results)]

    def plot_decoder_accuracy(self,labels, **kwargs):
        fig,ax = kwargs.get('plot',(None,None))
        start_loc = kwargs.get('start_loc',0)
        n_features = kwargs.get('n_features',None)

        self.accuracy_plot = plot_decoder_accuracy(self.fold_accuracy, labels,fig=fig,ax=ax,
                                                   start_loc=start_loc,n_features=n_features)
        self.accuracy_plot[0].show()

    def plot_confusion_matrix(self):
        for run_i, model_run in enumerate(np.array(self.models).flatten()):
            pass


class Session:
    def __init__(self,sessname,ceph_dir,pkl_dir='X:\Dammy\ephys_pkls',):
        self.td_df = pd.DataFrame()
        self.sessname = sessname
        self.spike_obj = None
        self.sound_event_dict = {}
        self.decoders = {}
        self.ceph_dir = ceph_dir
        # check for pkl
        class_pkl = (Path(pkl_dir)/ f'{sessname}.pkl')
        if class_pkl.is_file():
            with open(class_pkl,'rb') as pklfile:
                temp_class = pickle.load(pklfile)
            self.sound_event_dict = temp_class.sound_event_dict
            self.decoders = temp_class.decoders

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
        base_pip_idx = sound_writes['Payload'].mode()[0]

        # get trial start times
        long_dt_df = sound_writes.iloc[long_dt]
        trial_start_times = long_dt_df['Timestamp']
        a_idx = base_pip_idx + 2
        pattern_idxs = [a_idx + i * tone_space for i in range(4)]
        sound_events = pattern_idxs + [3, base_pip_idx,-1]
        sound_event_labels = ['A', 'B', 'C', 'D', 'X', 'base','trial_start']

        # get baseline pip
        sound_writes['idx_diff'] = sound_writes['Payload'].diff()
        writes_to_use = np.all([sound_writes['Payload'] == base_pip_idx, sound_writes['idx_diff'] == 0,
                                sound_writes['Trial_Number'] > 10, sound_writes['Time_diff'] < 0.3], axis=0)
        base_pips_times = base_pip_times_same_prev = sound_writes[writes_to_use]['Timestamp']

        # event sound times
        if sound_writes[sound_writes['Payload'] == 3].empty:
            sound_events.remove(3), sound_event_labels.remove('X')
        all_sound_times = [sound_writes[sound_writes['Payload'] == i]['Timestamp'] for i in sound_events
                           if i != base_pip_idx]
        if not base_pips_times.empty:
            all_sound_times.append(
                np.random.choice(base_pips_times, np.max([len(e) for e in all_sound_times]), replace=False))
        all_sound_times.append(trial_start_times)

        assert len(sound_events) == len(all_sound_times), Warning('sound event and labels must be equal length')
        for event, event_times, event_lbl in zip(sound_events, all_sound_times, sound_event_labels):
            self.sound_event_dict[event_lbl] = SoundEvent(event, event_times, event_lbl)

    def get_sound_psth(self,psth_window,baseline_dur=0.25,zscore_flag=False,redo_psth=False):
        if zscore_flag:
            baseline_dur = 0
        for e in self.sound_event_dict:
            self.sound_event_dict[e].get_psth(self.spike_obj,psth_window,title=f'session {self.sessname}',
                                              baseline_dur=baseline_dur, zscore_flag=zscore_flag,redo_psth=redo_psth)
            self.sound_event_dict[e].psth_plot[0].show()

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

    def run_decoder(self,decoder_name,labels,cv_folds=None,plot_flag=False,**kwargs):
        self.decoders[decoder_name].decode(cv_folds=cv_folds)
        if plot_flag:
            self.decoders[decoder_name].plot_decoder_accuracy(labels=labels,kwargs=kwargs)

    def load_trial_data(self,tdfile_path,td_home, tddir_path=r'H:\data\Dammy'):
        tddir_path = posix_from_win(tddir_path)
        td_path = Path(td_home)/tddir_path/tdfile_path
        self.td_df = pd.read_csv(td_path)
        self.get_n_since_last()

    def get_n_since_last(self):
        self.td_df['n_since_last'] = np.arange(self.td_df.shape[0])
        since_last = self.td_df[self.td_df['Tone_Position'] == 0].index
        for t,tt in zip(since_last,np.pad(since_last,[1,0])):
            self.td_df.loc[tt+1:t,'n_since_last'] = self.td_df.loc[tt+1:t,'n_since_last']-tt
        self.td_df.loc[t+1:, 'n_since_last'] = self.td_df.loc[t+1:, 'n_since_last'] - t



def get_predictor_from_psth(sess_obj:Session,event_key,psth_window, new_window) -> np.ndarray:
    td_start, td_end = [timedelta(0, t) for t in new_window]
    event_arr = sess_obj.sound_event_dict[event_key].psth[0]
    event_arr_2d = event_arr.reshape(-1, event_arr.shape[2])
    predictor_rate_mat = gen_firing_rate_matrix(pd.DataFrame(event_arr_2d,
                                                             columns=np.linspace(psth_window[0], psth_window[1],
                                                                                 event_arr_2d.shape[1])),
                                                baseline_dur=0,zscore_flag=True).loc[:, td_start:td_end]
    predictor = predictor_rate_mat.values.reshape((-1, event_arr.shape[1], predictor_rate_mat.shape[1]))  # units x trials x t steps
    predictor_mean = predictor.mean(axis=2)

    return predictor_mean
