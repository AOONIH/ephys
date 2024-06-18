import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path, PurePosixPath, PureWindowsPath
from tqdm import tqdm
import json
from datetime import timedelta
import pickle
from scipy.signal import convolve
from scipy.signal.windows import gaussian
from scipy.stats.mstats import zscore
from decoder import run_decoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from copy import deepcopy as copy
import scipy
from scipy.signal import savgol_filter
# import scicomap as sc
import multiprocessing
from functools import partial
from mpl_toolkits.axes_grid1 import make_axes_locatable
from elephant.statistics import instantaneous_rate
from quantities import s, ms
from neo import SpikeTrain
from elephant.kernels import GaussianKernel
from generate_synthetic_spikes import gen_responses

# import seaborn as sns
# from os import register_at_fork

# try:matplotlib.use('TkAgg')
# except ImportError:pass


def init_pool_processes():
    np.random.seed()


def posix_from_win(path: str, ceph_linux_dir='/ceph/akrami') -> Path:
    """
    Convert a Windows path to a Posix path.

    Args:
        path (str): The input Windows path.
        :param ceph_linux_dir:

    Returns:
        Path: The converted Posix path.
    """
    if ':\\' in path:
        path_bits = PureWindowsPath(path).parts
        path_bits = [bit for bit in path_bits if '\\' not in bit]
        return Path(PurePosixPath(*path_bits))
    else:
        assert ceph_linux_dir
        return Path(path).relative_to(ceph_linux_dir)


def get_spikedir(recdir, sorter='kilosort2_5', sorting_dir_name='sorting') -> Path:
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

    spikedir = recdir / sorting_dir_name / sorter / 'sorter_output'
    assert spikedir.is_dir(), Warning(f'spikedir {spikedir} not a dir, check path and sorter name')

    return spikedir


def gen_metadata(data_topology_filepath: [str, Path], ceph_dir, col_name='sound_bin_stem',
                 harp_bin_dir=r'X:\Dammy\harpbins'):
    if not isinstance(data_topology_filepath, Path):
        data_topology_filepath = Path(data_topology_filepath)
    data_topology = pd.read_csv(data_topology_filepath)
    for idx, sess in data_topology.iterrows():

        e_dir = next(posix_from_win(sess['ephys_dir']).rglob('continuous')).parent
        bin_stem = sess[col_name]
        dir_files = list(e_dir.iterdir())
        if 'metadata.json' not in dir_files or True:
            if harp_bin_dir and isinstance(harp_bin_dir, Path):
                harp_bin = harp_bin_dir / f'{bin_stem}_write_data.csv'
            else:
                bin_path = Path(bin_stem)
                harp_bin = bin_path.with_stem(f'{bin_path.stem}_write_data').with_suffix('.csv')
            # harp_bin_dir = Path(harp_bin_dir)
            harp_writes = pd.read_csv(harp_bin)
            trigger_time = harp_writes[harp_writes['DO3'] == True]['Times'].values[0]
            metadata = {'trigger_time': trigger_time}
            with open(e_dir / 'metadata.json', 'w') as jsonfile:
                json.dump(metadata, jsonfile)


def load_spikes(spike_times_path: [Path | str], spike_clusters_path: [Path | str], parent_dir=None):
    """
    A function to load spike times and clusters from the given paths.

    :param spike_times_path: Path to the spike times file
    :param spike_clusters_path: Path to the spike clusters file
    :param parent_dir: Optional parent directory for the spike times and clusters paths
    :return: Tuple containing spike times and spike clusters arrays
    """
    spike_times_path, spike_clusters_path = Path(spike_times_path), Path(spike_clusters_path)
    assert spike_times_path.suffix == '.npy' and spike_clusters_path.suffix == '.npy', Warning('paths should be .npy')

    if parent_dir:
        spike_times_path, spike_clusters_path = [next(parent_dir.rglob(path.name)) or path
                                                 for path in [spike_times_path, spike_clusters_path]
                                                 if not path.is_absolute()]
    spike_times = np.load(spike_times_path)
    spike_clusters = np.load(spike_clusters_path)

    return spike_times, spike_clusters


def load_pupil_data(pupil_data_path: [Path | str], parent_dir=None):
    """
    A function to load pupil data from the given path.

    :param pupil_data_path: Path to the pupil data file
    :param parent_dir: Optional parent directory for the pupil data path
    :return: Pupil data array
    """
    pupil_data_path = Path(pupil_data_path)
    with open(pupil_data_path, 'rb') as f:
        pupil_data = pickle.load(f)
    for sess in list(pupil_data.keys()):
        if pupil_data[sess].pupildf is None:
            pupil_data.pop(sess)
        else:
            pupil_data[sess].pupildf.index = pupil_data[sess].pupildf.index
    return pupil_data


def cluster_spike_times(spike_times: np.ndarray, spike_clusters: np.ndarray) -> dict:
    assert spike_clusters.shape == spike_times.shape, Warning('spike times/ cluster arrays need to be same shape')
    cluster_spike_times_dict = {}
    for i in tqdm(np.unique(spike_clusters), desc='getting session cluster spikes times',
                  total=len(np.unique(spike_clusters)), disable=True):
        cluster_spike_times_dict[i] = spike_times[spike_clusters == i]
    return cluster_spike_times_dict


def get_times_in_window(all_times: np.ndarray,window: [list | np.ndarray]) -> np.ndarray:
    return all_times[(all_times >= window[0]) * (all_times <= window[1])]


def get_spike_rate_in_window(spike_times: np.ndarray, window: [list | np.ndarray], fs):
    spike_train = SpikeTrain(spike_times*s,t_start=window[0],t_stop=window[1])
    bin_firing_rate = np.squeeze(np.array(instantaneous_rate(spike_train,sampling_period=100*ms,
                                                             kernel=GaussianKernel(40*ms))))

    return bin_firing_rate


def get_spike_times_in_window(event_time: int, spike_time_dict: dict, window: [list | np.ndarray], fs):
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
        all_spikes = (spike_time_dict[cluster_id] - event_time)  # / fs

        window_spikes_dict[cluster_id] = all_spikes[(all_spikes >= window[0]) * (all_spikes <= window[1])]


    return window_spikes_dict


def gen_spike_matrix_old(spike_time_dict: dict, window, fs):
    precision = np.ceil(np.log10(fs)).astype(int)
    time_cols = np.round(np.arange(window[0], window[1] + 1 / fs, 1 / fs), precision)
    spike_matrix = pd.DataFrame(np.zeros((len(spike_time_dict), int((window[1] - window[0]) * fs) + 1)),
                                index=list(spike_time_dict.keys()), columns=time_cols)
    rounded_spike_dict = {}
    for cluster_id in tqdm(spike_time_dict, desc='rounding spike times for event', total=len(spike_time_dict),
                           disable=True):
        rounded_spike_dict[cluster_id] = np.round(spike_time_dict[cluster_id], precision)
    for cluster_id in tqdm(rounded_spike_dict, desc='assigning spikes to matrix', total=len(spike_time_dict),
                           disable=True):
        spike_matrix.loc[cluster_id][rounded_spike_dict[cluster_id]] = 1
    return spike_matrix


def gen_spike_matrix(spike_time_dict: dict, window, fs):
    fs = 100
    precision = np.ceil(np.log10(fs)).astype(int)
    time_cols = np.round(np.arange(window[0], window[1] + 1 / fs, 1 / fs), precision)
    # spike_matrix = pd.DataFrame(np.zeros((len(spike_time_dict), int((window[1] - window[0]) * fs) + 1)),
    #                             index=list(spike_time_dict.keys()), columns=time_cols)


    event_psth = [np.squeeze(np.array(instantaneous_rate(SpikeTrain(c_spiketimes * s,
                                                                    t_start=window[0], t_stop=window[-1]*s+0.011*s),
                                                         sampling_period=10 * ms,
                                                         kernel=GaussianKernel(40 * ms))))
                  for c_spiketimes in spike_time_dict.values()]
    spike_matrix = pd.DataFrame(event_psth,columns=time_cols,index=list(spike_time_dict.keys()))
    spike_matrix.columns = pd.to_timedelta(spike_matrix.columns, 's')

    return spike_matrix


def check_unique_across_dim(arr: [np.ndarray, list],dim2check=0):
    if isinstance(arr, list):
        arr = np.vstack(arr)
    return len(np.unique(arr, axis=dim2check)) == len(arr)


def plot_spike_time_raster(spike_time_dict: dict, ax=None, **pltkwargs):
    if not ax:
        fig, ax = plt.subplots()
    assert isinstance(ax, plt.Axes)
    for cluster_id in tqdm(spike_time_dict, desc='plotting spike times for event', total=len(spike_time_dict),
                           disable=True):
        ax.scatter(spike_time_dict[cluster_id], [cluster_id] * len(spike_time_dict[cluster_id]), **pltkwargs)
        ax.invert_xaxis()


def gen_firing_rate_matrix(spike_matrix: pd.DataFrame, bin_dur=0.01, baseline_dur=0.0,
                           zscore_flag=False, gaus_std=0.04) -> pd.DataFrame:
    # print(f'zscore_flag = {zscore_flag}')
    if gaus_std:
        gaus_window = gaussian(int(gaus_std / bin_dur*2), int(gaus_std / bin_dur*2))
        gaus_window = np.array_split(gaus_window,2)[1]
    spike_matrix.columns = pd.to_timedelta(spike_matrix.columns, 's')
    rate_matrix = spike_matrix.T.resample(f'{bin_dur}S').mean().T / bin_dur
    cols = rate_matrix.columns
    if gaus_std:
        rate_matrix = np.array([convolve(row, gaus_window, mode='same') for row in rate_matrix.values])
    assert not all([baseline_dur, zscore_flag])
    rate_matrix = pd.DataFrame(rate_matrix, columns=cols)
    if baseline_dur:
        rate_matrix = pd.DataFrame(rate_matrix, columns=cols)
        rate_matrix = rate_matrix.sub(np.mean(rate_matrix.loc[:, timedelta(0, -baseline_dur):timedelta(0, 0)],axis=1),
                                      axis=0)
    if zscore_flag:
        # rate_matrix = (rate_matrix.T - rate_matrix.mean(axis=1))/rate_matrix.std(axis=1)
        rate_matrix = zscore(rate_matrix, axis=1, )
    rate_matrix = rate_matrix.fillna(0)
    return rate_matrix


# def load_sound_bin(fpath:Pathbin_stem, bin_dir=Path(r'X:\Dammy\harpbins')):
def load_sound_bin(binpath: Path):
    # all_writes = pd.read_csv(bin_dir/f'{bin_stem}_write_indices.csv')
    all_writes = pd.read_csv(binpath)
    return all_writes


def format_sound_writes(sound_writes_df: pd.DataFrame, patterns: [[int, ], ], ) -> pd.DataFrame:
    # assign sounds to trials
    sound_writes_df = sound_writes_df.drop_duplicates(subset='Timestamp', keep='first').copy()
    sound_writes_df = sound_writes_df.reset_index(drop=True)
    # sound_writes_df['Timestamp'] = sound_writes_df['Timestamp']
    sound_writes_df['Trial_Number'] = np.full_like(sound_writes_df.index, -1)
    sound_writes_diff = sound_writes_df['Time_diff'] = sound_writes_df['Timestamp'].diff()
    # print(sound_writes_diff[:10])
    long_dt = np.squeeze(np.argwhere(sound_writes_diff.values > 1))
    for n, idx in enumerate(long_dt):
        sound_writes_df.loc[idx:, 'Trial_Number'] = n
    sound_writes_df['Trial_Number'] = sound_writes_df['Trial_Number'] + 1

    sound_writes_df['Time_diff'] = sound_writes_df['Timestamp'].diff()
    sound_writes_df['Payload_diff'] = sound_writes_df['Payload'].diff()

    tones = sound_writes_df['Payload'].unique()
    pattern_tones = tones[tones >= 8]
    base_pip_idx = [e for e in pattern_tones if e not in sum(patterns, [])]
    if len(base_pip_idx) > 1:
        base_pip_idx = sound_writes_df.query('Payload in @base_pip_idx')['Payload'].mode().iloc[0]
    else:
        base_pip_idx = base_pip_idx[0]

    normal_patterns = [pattern for pattern in patterns if (np.diff(pattern) > 0).all()]
    deviant_patterns = [pattern for pattern in patterns if pattern not in normal_patterns]

    sound_writes_df['pattern_pips'] = sound_writes_df['Payload'].isin(sum(patterns, []))
    sound_writes_df['pattern_start'] = sound_writes_df['Payload_diff'].isin(np.array(patterns)[:, 0] - base_pip_idx) * \
                                       sound_writes_df['pattern_pips']
    sound_writes_df['pip_counter'] = np.zeros_like(sound_writes_df['pattern_start']).astype(int)
    patterns_presentations = [sound_writes_df.iloc[idx:idx+4]
                              for idx in sound_writes_df.query('pattern_start == True').index]
    sound_writes_df['ptype'] = [np.nan]*sound_writes_df.shape[0]
    for pattern_pres in patterns_presentations:
        patt = pattern_pres['Payload'].to_list()
        ptype = 0 if patt in normal_patterns else 1 if patt in deviant_patterns else -1
        sound_writes_df.loc[pattern_pres.index, 'ptype'] = ptype

    for i in sound_writes_df.query('pattern_start == True').index:
        sound_writes_df.loc[i:i + 3, 'pip_counter'] = np.cumsum(sound_writes_df['pattern_pips'].loc[i:i + 3]) \
                                                      * sound_writes_df['pattern_pips'].loc[i:i + 3]

    # base_pip_idx = normal[0] - (-2 if (normal[0] - normal[1] > 0) else 2)  # base to pip gap
    # non_base_pips = sound_writes_df.query('Payload > 8 & Payload != @base_pip_idx')['Payload'].unique()
    # sound_writes_df['pattern_pips'] = np.any((sound_writes_df['Payload_diff'] == (-2 if (normal[0] - normal[1] > 0) else 2),
    #                                           sound_writes_df['Payload'].isin(non_base_pips)), axis=0)
    # sound_writes_df['pattern_start'] = sound_writes_df['pattern_pips'] * sound_writes_df['pattern_pips'].diff() > 0
    # sound_writes_df['pip_counter'] = np.zeros_like(sound_writes_df['pattern_start']).astype(int)
    # for i in sound_writes_df.query('pattern_start == True').index:
    #     sound_writes_df.loc[i:i + 3, 'pip_counter'] = np.cumsum(sound_writes_df['pattern_pips'].loc[i:i + 3]) \
    #                                                   * sound_writes_df['pattern_pips'].loc[i:i + 3]

    return sound_writes_df


def read_lick_times(beh_event_path: Path):
    beh_events = pd.read_csv(beh_event_path)
    lick_times = beh_events.query('Payload == 0')['Timestamp'].values
    return lick_times


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

    # Divide indices into bins
    ibs = []  # np.nonzero((y>=ybins[0])*(y<=ybins[1]))[0]]
    for ymin, ymax in zip(ybins[:-1], ybins[1:]):
        i = np.nonzero((y > ymin) * (y <= ymax))[0]
        ibs.append(i)

    # Assign x indices
    dx = width / (nmax // 2)
    for i in ibs:
        yy = y[i]
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(yy)]
            a = i[j::2]
            b = i[j + 1::2]
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x


def point_cloud(x_loc, y_ser, spread=0.1):
    return np.random.uniform(low=x_loc - spread, high=x_loc + spread, size=len(y_ser))
    # return np.random.normal(loc=x_loc, scale=spread, size=len(y_ser))


def plot_2d_array_with_subplots(array_2d: np.ndarray, cmap='viridis', cbar_width=0.1, cbar_height=0.8, plot_cbar=True,
                                extent=None, aspect=1.0, vcenter=None, plot=None,
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
        fig, ax = plot
    # Create the matshow plot on the specified axis with the provided colormap and extent
    divider = make_axes_locatable(ax)
    if vcenter:
        im = ax.imshow(array_2d, cmap=cmap, extent=extent,
                        norm=matplotlib.colors.TwoSlopeNorm(vcenter=vcenter, ))  # vmin=vmin,vmax=vmax
    else:
        im = ax.imshow(array_2d, cmap=cmap, extent=extent, **im_kwargs)
    # ax.set_aspect(array_2d.shape[1]/(array_2d.shape[0]*100*aspect))
    ax.set_aspect('auto')

    # Add a color bar legend using fig.colorbar with explicit width and height
    if plot_cbar:
        cax = divider.append_axes('right', size='7.5%', pad=0.1)
        cbar = fig.colorbar(im, cax=cax, fraction=cbar_width, aspect=cbar_height, )
    else:
        cbar = None
    # Show the plot
    # plt.show()
    return fig, ax, cbar


# def fixed_cmap(cmap):
#     div_map = sc.ScicoDiverging(cmap=cmap)
#     div_map.unif_sym_cmap(lift=15,
#                           bitonic=False,
#                           diffuse=True)


def predict_1d(models, t_ser, y_lbl=0):
    # models = models
    # with multiprocessing.Pool() as pool:
    #     results = tqdm(pool.imap())
    prediction_ts = np.array([m.predict(t_ser.T) for m in models])
    if isinstance(y_lbl, list):
        assert len(y_lbl) == prediction_ts.shape[0]
        _arr = np.array(row == lbl for row, lbl in zip(prediction_ts, y_lbl))
        prediction_ts = _arr
    return prediction_ts


def load_session_pkls(pkl_path):
    with open(pkl_path, 'rb') as pklfile:
        sess_obj = pickle.load(pklfile)
        # sess_obj.sound_event_dict = {}
        # sessions[sess_obj.sessname] = sess_obj
    return sess_obj


class SessionSpikes:
    def __init__(self, spike_times_path: [Path | str], spike_clusters_path: [Path | str], sess_start_time: float,
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
        self.spike_times, self.clusters = load_spikes(spike_times_path, spike_clusters_path, parent_dir)
        self.spike_times = self.spike_times / fs + sess_start_time  # units seconds
        self.duration = self.spike_times[-1] - self.start_time

        self.cluster_spike_times_dict = cluster_spike_times(self.spike_times, self.clusters)
        self.bad_units = set()
        # self.curate_units()
        # self.curate_units_by_rate()
        if (parent_dir / 'good_units.csv').is_file():
            self.good_units = pd.read_csv(parent_dir / 'good_units.csv').iloc[:, 0].to_list()
            unit_ids = list(self.cluster_spike_times_dict.keys())
            for unit in unit_ids:
                if unit not in self.good_units:
                    self.cluster_spike_times_dict.pop(unit)
            print(f'good units: {self.good_units}')
        else:
            self.curate_units_by_rate()
        self.unit_means = self.get_unit_mean_std()
        self.units = list(self.cluster_spike_times_dict.keys())
        self.event_spike_matrices = multiprocessing.Manager().dict()
        self.event_cluster_spike_times = multiprocessing.Manager().dict()

    def get_event_spikes(self, event_times: [list | np.ndarray | pd.Series], event_name: str,
                         window: [list | np.ndarray], get_spike_matrix=True):
        if self.event_cluster_spike_times is None:
            self.event_cluster_spike_times = multiprocessing.Manager().dict()
            for event_time in event_times:
                if f'{event_name}_{event_time}' not in list(self.event_cluster_spike_times.keys()):
                    self.event_cluster_spike_times[f'{event_name}_{event_time}'] = get_spike_times_in_window(event_time,
                                                                                                             self.cluster_spike_times_dict,
                                                                                                             window,
                                                                                                             self.new_fs)
                if get_spike_matrix:
                    if self.event_spike_matrices is None:
                        self.event_spike_matrices = multiprocessing.Manager().dict()
                    if f'{event_name}_{event_time}' not in list(self.event_spike_matrices.keys()):
                        self.event_spike_matrices[f'{event_name}_{event_time}'] = gen_spike_matrix(
                            self.event_cluster_spike_times[f'{event_name}_{event_time}'],
                            window, self.new_fs)


    def curate_units(self):
        # self.bad_units = set()
        for unit in self.cluster_spike_times_dict:
            d_spike_times = np.diff(self.cluster_spike_times_dict[unit])
            if np.mean(d_spike_times > 10) > 0.05:
                # self.cluster_spike_times_dict.pop(unit)
                self.bad_units.add(unit)
        print(
            f'popped units {self.bad_units}, remaining units: {len(self.cluster_spike_times_dict) - len(self.bad_units)}/{len(self.cluster_spike_times_dict)}')
        for unit in self.bad_units:
            self.cluster_spike_times_dict.pop(unit)

    def curate_units_by_rate(self):
        # self.bad_units = set()
        for unit in self.cluster_spike_times_dict:
            d_spike_times = np.diff(self.cluster_spike_times_dict[unit])
            if np.mean(d_spike_times) > (1 / 0.05):
                # self.cluster_spike_times_dict.pop(unit)
                self.bad_units.add(unit)
        print(
            f'popped units {self.bad_units}, remaining units: {len(self.cluster_spike_times_dict) - len(self.bad_units)}/{len(self.cluster_spike_times_dict)}')
        for unit in self.bad_units:
            self.cluster_spike_times_dict.pop(unit)

    def get_unit_mean_std(self,time_bins_idx=None):

        unit_means = np.full(len(self.cluster_spike_times_dict), np.nan)
        unit_stds = np.full(len(self.cluster_spike_times_dict), np.nan)

        for i, unit in sorted(list(enumerate(self.cluster_spike_times_dict))):
            # unit_means[i] = (self.cluster_spike_times_dict[unit].shape / self.duration)
            time_bin = 1
            # unit_rates = [get_times_in_window(self.cluster_spike_times_dict[unit], [i, i + time_bin]).shape[0]/time_bin
            #               for i in np.arange(int(self.start_time), int(self.start_time) + int(self.duration), time_bin)]
            spike_train = SpikeTrain(self.cluster_spike_times_dict[unit]*s,t_start=self.start_time,
                                     t_stop=self.start_time + self.duration)
            unit_rates = np.squeeze(np.array(instantaneous_rate(spike_train,sampling_period=100*ms,
                                                                kernel=GaussianKernel(40*ms))))
            unit_means[i] = np.mean(unit_rates)
            # unit_rates = np.array(unit_rates)
            if time_bins_idx is not None:
                assert len(time_bins_idx) == len(unit_rates)
                unit_rates = unit_rates[time_bins_idx]
                unit_means[i] = np.mean(unit_rates)
            unit_stds[i] = np.std(unit_rates) if np.std(unit_rates) > 0.01 else 1

        assert np.isnan(unit_means).sum() == 0 + np.isnan(unit_stds).sum() == 0, 'Nans in mean and std'
        # return 0.1*unit_means, 0.1*unit_stds
        return unit_means, unit_stds


def plot_decoder_accuracy(decoder_accuracy, labels, fig=None, ax=None, start_loc=0,
                          n_features=None, plt_kwargs=None) -> [plt.Figure, plt.Axes]:
    if not isinstance(ax, plt.Axes):
        fig, ax = plt.subplots()

    if not plt_kwargs:
        plt_kwargs = {}
    decoder_accuracy = np.array(decoder_accuracy)
    for li, label_res in enumerate(decoder_accuracy):
        ax.scatter(simple_beeswarm2(label_res.flatten(), width=0.1) + start_loc + li, label_res.flatten(),
                   label=labels[li], alpha=0.5, **plt_kwargs)
        ax.scatter(li + start_loc, np.mean(label_res), marker='^', c='k', s=20)
        # ax.scatter(point_cloud(0,label_res.flatten()) + start_loc+li, sorted(label_res.flatten()),
        #            label=labels[li], alpha=0.1,**plt_kwargs)
    ax.set_ylim(0, 1.19)
    ax.set_ylabel('decoder accuracy')

    if not n_features:
        n_features = 2
    ax.axhline(1 / n_features, c='k', ls='--')

    ax.set_xticks(np.arange(0, len(labels), 1))
    ax.set_xticklabels(labels)
    ax.legend(loc=1)

    return fig, ax


def get_event_psth(sess_spike_obj: SessionSpikes, event_idx, event_times: [pd.Series, np.ndarray, list],
                   window: [float, float], event_lbl: str, baseline_dur=0.25, zscore_flag=False, iti_zscore=None,
                   gaus_std=0.04, synth_data=None) -> (
        np.ndarray, pd.DataFrame):
    if iti_zscore:
        zscore_flag = False

    assert not (zscore_flag and iti_zscore), 'zscore_flag and iti_zscore cannot both be True'

    sess_spike_obj.get_event_spikes(event_times, f'{event_idx}', window)
    event_keys = [f'{event_idx}_{t}' for t in event_times]
    if synth_data is None:
        all_event_list = [sess_spike_obj.event_spike_matrices[key]for key in event_keys]
    else:
        unit_rates = np.random.rand(len(sess_spike_obj.clusters) * 40)
        synth_times = gen_responses(unit_rates, len(event_times), np.arange(window[0],window[1],0.002))
        spike_trains = [[SpikeTrain(ee, t_start=window[0], t_stop=window[1]*s+0.11*s, units=s) for ee in e] for e in synth_times]
        all_event_list = np.array([instantaneous_rate(st, 100 * ms, kernel=GaussianKernel(40*ms)).T for st in spike_trains])

    # all_events_stacked = np.vstack(all_event_list)
    # all_event_mean = pd.DataFrame(np.array(all_event_list).mean(axis=0))
    # all_event_mean = pd.DataFrame(all_events_stacked)
    # all_event_mean.columns = np.linspace(window[0], window[1], all_event_mean.shape[1])

    # rate_mat_stacked = gen_firing_rate_matrix(all_event_mean, baseline_dur=baseline_dur,
    #                                           zscore_flag=False,gaus_std=gaus_std)  # -all_events.values[:,:1000].mean(axis=1,keepdims=True)
    # ratemat_arr3d = rate_mat_stacked.values.reshape((len(all_event_list), -1, rate_mat_stacked.shape[1]))
    ratemat_arr3d = np.array(all_event_list) if isinstance(all_event_list, list) else all_event_list
    x_ser = np.linspace(window[0], window[1], ratemat_arr3d.shape[-1])
    rate_mat = pd.DataFrame(ratemat_arr3d.mean(axis=0),
                            columns=x_ser)
    rate_mat.index = sess_spike_obj.units
    # rate_mat = rate_mat.assign(m=rate_mat.loc[:,timedelta(0,0):timedelta(0,0.2)].mean(axis=1)
    #                            ).sort_values('m',ascending=False).drop('m', axis=1)
    # sorted_arrs = [e.iloc[rate_mat.index.to_series()] for e in all_event_list]
    # sorted_arrs = all_event_list
    # rate_mat = rate_mat.sort_values(by=timedelta(0, 0.2), ascending=False

    return ratemat_arr3d, rate_mat, rate_mat.index


def plot_psth(psth_rate_mat, event_lbl, window, title='', cbar_label=None, cmap='hot', **im_kwargs):
    if not cbar_label:
        cbar_label = 'firing rate (Hz)'
    fig, ax, cbar = plot_2d_array_with_subplots(psth_rate_mat, cmap=cmap,
                                                extent=[window[0], window[1], psth_rate_mat.shape[0], 0],
                                                cbar_height=50, aspect=0.1, **im_kwargs)
    ax.set_ylabel('unit number', fontsize=14)
    ax.set_xlabel(f'time from {event_lbl} onset (s)', fontsize=14)
    ax.set_title(title)
    if cbar:
        cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=5)
    for t in np.arange(0, 1, 1):
        ax.axvline(t, ls='--', c='white', lw=1)

    return fig, ax, cbar


def plot_psth_ts(psth_mat, x_ser, x_lbl='', y_lbl='', title='', plot=None, **plt_kwargs) -> [plt.Figure, plt.Axes]:
    if not plot:
        plot = plt.subplots()
    plot[1].plot(x_ser, psth_mat.mean(axis=0), **plt_kwargs)
    plot[1].set_title(title)
    plot[1].set_xlabel(x_lbl)
    plot[1].set_ylabel(y_lbl)
    return plot


def plot_ts_var(x_ser: np.ndarray | pd.Series, y_ser: np.ndarray | pd.Series, colour: str, plt_ax: plt.Axes,
                n=500,ci_kwargs=None):
    if isinstance(x_ser, pd.Series):
        x_ser = x_ser.to_numpy()
    if isinstance(y_ser, pd.Series):
        y_ser = y_ser.to_numpy()

    rand_npdample = [y_ser[np.random.choice(y_ser.shape[0], y_ser.shape[0], replace=True), :].mean(axis=0)
                     for i in range(n)]
    rand_npsample = np.array(rand_npdample)
    ci = np.apply_along_axis(mean_confidence_interval, axis=0, arr=rand_npsample,
                             **ci_kwargs if ci_kwargs else {}).astype(float)

    plt_ax.fill_between(x_ser, ci[1], ci[2], alpha=0.1, fc=colour)


def mean_confidence_interval(data, confidence=0.95,var_func=np.std):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.nanmean(a), var_func(a)  # a.std()
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m - h, m + h


def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))


def multi_pearsonr(x, y):
    xmean = x.mean(axis=1)
    ymean = y.mean()
    xm = x - xmean[:, None]
    ym = y - ymean
    normxm = np.linalg.norm(xm, axis=1)
    normym = np.linalg.norm(ym)
    return np.clip(np.dot(xm / normxm[:, None], ym / normym), -1.0, 1.0)


class SoundEvent:
    def __init__(self, idx, times, lbl):
        self.idx = idx
        self.times = times
        self.lbl = lbl

        self.psth = None
        self.psth_plot = None

    def get_psth(self, sess_spike_obj: SessionSpikes, window, title='', redo_psth=False, redo_psth_plot=False,
                 baseline_dur=0.25, zscore_flag=False, iti_zscore=None, reorder_idxs=None,synth_data=None):

        if not self.psth or redo_psth:
            self.psth = get_event_psth(sess_spike_obj, self.idx, self.times, window, self.lbl,
                                       baseline_dur=baseline_dur, zscore_flag=zscore_flag, iti_zscore=None,
                                       synth_data=synth_data)
            if iti_zscore:
                self.psth = (self.psth[0],zscore_by_unit(self.psth[1],unit_means=iti_zscore[0],unit_stds=iti_zscore[1]),
                             self.psth[2])
                # self.psth[1][self.psth[1] == np.nan] = 0.0
            if zscore_flag:
                means,stds = sess_spike_obj.unit_means
                self.psth = (self.psth[0],zscore_by_unit(self.psth[1], means, stds),self.psth[2])

        if not self.psth_plot or redo_psth or redo_psth_plot:

            self.psth_plot = plot_psth(self.psth[1], self.lbl, window, title=title)
            if zscore_flag:
                self.psth_plot[2].ax.set_ylabel('zscored firing rate (au)', rotation=270)

    def save_plot_as_svg(self, figdir=Path(r'X:\Dammy\figures\ephys', ), suffix=''):
        """
        Save the plot as an SVG file.

        Parameters:

        """
        filename = figdir / f'{self.lbl}_{suffix}.svg'
        if self.psth_plot:
            self.psth_plot[0].savefig(filename, format='svg')
            print(f"Plot saved as {filename}")
        else:
            print("No plot to save. Call 'plot_psth' first.")


class Decoder:
    def __init__(self, predictors, features, model_name, ):
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

    def decode(self, dec_kwargs, **kwargs):
        if not dec_kwargs.get('cv_folds', 0):
            n_runs = kwargs.get('n_runs', 500)
        else:
            n_runs = kwargs.get('n_runs', 500)
        # cv_folds = kwargs.get('cv_folds',None)
        # if not kwargs.get('balance',False):
        if not isinstance(self.predictors, (list | tuple)):
            preds, feats = [self.predictors] * n_runs, [self.features] * n_runs
        else:
            n_run_pred = [balance_predictors(self.predictors, self.features) for i in range(n_runs)]
            preds = [run[0] for run in n_run_pred]
            feats = [run[1] for run in n_run_pred]
        with multiprocessing.Pool(initializer=init_pool_processes) as pool:
            results = list(tqdm(pool.starmap(partial(run_decoder, model=self.model_name,
                                                     **dec_kwargs),
                                             zip(preds, feats)), total=n_runs))
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

    def map_decoding_ts(self, t_ser, model_i=0, y_lbl=0):
        dec_models = np.array(self.models).flatten()
        # with multiprocessing.Pool() as pool:
        #     results = tqdm(pool.imap())
        self.prediction_ts = np.array([m.predict(t_ser.T) for m in dec_models])
        if isinstance(y_lbl, list):
            assert len(y_lbl) == self.prediction_ts.shape[0]
            _arr = np.array(row == lbl for row, lbl in zip(self.prediction_ts, y_lbl))
            self.prediction_ts = _arr
        return self.prediction_ts

    def plot_decoder_accuracy(self, labels, plt_kwargs=None, **kwargs):
        fig, ax = kwargs.get('plot', (None, None))
        start_loc = kwargs.get('start_loc', 0)
        n_features = kwargs.get('n_features', None)

        if len(labels) <= 2:
            self.accuracy_plot = plot_decoder_accuracy(self.fold_accuracy, labels, fig=fig, ax=ax,
                                                       plt_kwargs=plt_kwargs,
                                                       start_loc=start_loc, n_features=n_features)
        else:
            unique_lbls = np.arange(len(labels))
            y_tests_folds = [np.hstack(ee) for ee in [e[0] for e in self.predictions]]
            y_preds_folds = [np.hstack(ee) for ee in [e[1] for e in self.predictions]]

            lbl_accuracy_list = [[(fold_preds[fold_tests == lbl] == lbl).mean() for lbl in unique_lbls]
                                 for fold_tests, fold_preds in zip(y_tests_folds, y_preds_folds)]
            lbl_accuracy_list = np.array(lbl_accuracy_list).T
            # for lbl_i, (lbl, lbl_acc) in enumerate(zip(labels,lbl_accuracy_list)):
            self.accuracy_plot = plot_decoder_accuracy(lbl_accuracy_list, labels, fig=fig, ax=ax, plt_kwargs=plt_kwargs,
                                                       start_loc=start_loc, n_features=len(labels))
            self.accuracy_plot[1].legend(ncols=len(labels))

        self.accuracy_plot[0].show()

    def plot_confusion_matrix(self, labels, **kwargs):
        y_tests = np.hstack([np.hstack(ee) for ee in [e[0] for e in self.predictions]])
        y_preds = np.hstack([np.hstack(ee) for ee in [e[1] for e in self.predictions]])
        self.cm = confusion_matrix(y_tests, y_preds, normalize='true')
        cm_plot_ = ConfusionMatrixDisplay(self.cm, display_labels=labels, )
        cm_plot__ = cm_plot_.plot(**kwargs)
        self.cm_plot = cm_plot__.figure_, cm_plot__.ax_
        self.cm_plot[1].invert_yaxis()


class SessionLicks:
    def __init__(self, lick_times: np.ndarray, sound_writes: pd.DataFrame, fs=1e3, resample_fs=1e2, ):

        self.event_lick_plots = {}
        self.event_licks = {}
        self.fs = fs
        self.new_fs = resample_fs
        self.spike_times, self.clusters = lick_times, np.zeros_like(lick_times)

        self.cluster_spike_times_dict = cluster_spike_times(self.spike_times, self.clusters)

        self.event_spike_matrices = dict()
        self.event_cluster_spike_times = dict()
        self.sound_writes = sound_writes

    def get_event_spikes(self, event_idx: int, event_name: str, window: [list | np.ndarray], sessname: str):
        event_times = self.sound_writes.query(f'Payload == {event_idx}')['Timestamp'].values
        event_trialnum = self.sound_writes.query(f'Payload == {event_idx}')['Trial_Number'].values
        self.event_spike_matrices[event_name] = dict()
        self.event_cluster_spike_times[event_name] = dict()
        for event_time in event_times:
            if f'{event_name}_{event_time}' not in list(self.event_cluster_spike_times.keys()):
                self.event_cluster_spike_times[event_name][f'{event_name}_{event_time}'] = get_spike_times_in_window(
                    event_time, self.cluster_spike_times_dict, window, self.new_fs)
            if f'{event_name}_{event_time}' not in list(self.event_spike_matrices.keys()):
                self.event_spike_matrices[event_name][f'{event_name}_{event_time}'] = gen_spike_matrix(
                    self.event_cluster_spike_times[event_name][f'{event_name}_{event_time}'],
                    window, self.new_fs)
        # self.event_licks[f'{event_name}_licks'] = pd.concat([self.event_spike_matrices[e_key] for e_key in self.event_spike_matrices
        #                                                      if event_name in e_key])
        self.event_licks[f'{event_name}_licks'] = pd.concat(self.event_spike_matrices[event_name])
        # multi index for licks
        self.event_licks[f'{event_name}_licks'].index = pd.MultiIndex.from_arrays([event_times, event_trialnum,
                                                                                   [sessname] * len(event_trialnum)],
                                                                                  names=['time', 'trial', 'sess'])

    def plot_licks(self, event_name, window=(-3, 3)):
        licks_to_event = self.event_licks[f'{event_name}_licks']
        lick_plot = plot_2d_array_with_subplots(licks_to_event, cmap='binary', extent=[window[0], window[1],
                                                                                       licks_to_event.shape[0], 0],
                                                plot_cbar=False)
        lick_plot[1].axvline(0, c='k', ls='--')
        lick_plot[1].set_ylabel('Trials')
        lick_plot[1].set_xlabel(f'time since {event_name} (s)')

        self.event_lick_plots[f'licks_to_{event_name}'] = lick_plot


class SessionPupil:
    def __init__(self, pupil_data, sound_events, beh_events):
        # sess_idx = ord(sessname[1]) - ord('a')
        self.aligned_pupil = {}
        self.pupil_data = pupil_data
        self.sound_writes = sound_events
        self.beh_events = beh_events

    def align2events(self, event_idx: int, event_name: str, window, sessname: str,
                     baseline_dur=0.0, size_col='dlc_radii_a_zscored'):
        """
            Aligns pupil data to sound events.

            Parameters
            ----------
            event_idx : int
                Index of the sound event to align to.
            event_name : str
                Name of the sound event to align to.
            window : tuple
                Time window to align to the sound event.
            sessname : str
                Name of the session.
            baseline_dur : float, optional
                Duration of the baseline period to subtract from the aligned pupil data.
            size_col : str, optional
                Name of the column containing pupil size data.

            Returns
            -------
            None
            """

        pupil_size = self.pupil_data[size_col]
        start, end = pupil_size.index[0], pupil_size.index[-1]
        event_times = self.sound_writes.query(f'Payload == {event_idx}')['Timestamp'].values
        event_trialnums = self.sound_writes.query(f'Payload == {event_idx}')['Trial_Number'].values
        dt = np.nanmedian(np.diff(pupil_size.index))
        event_tdeltas = np.round(np.arange(window[0], window[1], dt), 2)

        # aligned_epochs = [[pupil_size.loc[eventtime + tt], np.nan) for tt in event_tdeltas]
        #                   for eventtime in event_times]
        # aligned_epochs = [pupil_size.loc[eventtime+window[0]: eventtime+window[1]] for eventtime in event_times]
        aligned_epochs = []
        for eventtime in event_times:
            a = pupil_size.loc[eventtime + window[0]: eventtime + window[1]]
            if a.isna().all():
                print(f'{start<eventtime<end= } {eventtime} not found')
                aligned_epochs.append(pd.Series(np.full_like(event_tdeltas, np.nan)))
            else:
                aligned_epochs.append(pupil_size.loc[eventtime + window[0]: eventtime + window[1]])
        for aligned_epoch, eventtime in zip(aligned_epochs, event_times):
            aligned_epoch.index = np.round(aligned_epoch.index - eventtime, 2)
        sessname_list = [sessname] * len(event_times)
        aligned_epochs_df = pd.DataFrame(aligned_epochs, columns=event_tdeltas,
                                         index=pd.MultiIndex.from_tuples(list(zip(event_times, event_trialnums,
                                                                                  sessname_list)),
                                                                         names=['time', 'trial', 'sess']))
        aligned_epochs_df = aligned_epochs_df.dropna(axis=0, how='all')
        # for col in aligned_epochs_df.columns:
        #     aligned_epochs_df[col] = pd.to_numeric(aligned_epochs_df[col],errors='coerce')
        aligned_epochs_df = aligned_epochs_df.ffill(axis=1, )
        aligned_epochs_df = aligned_epochs_df.bfill(axis=1, )
        if baseline_dur:
            epoch_baselines = aligned_epochs_df.loc[:, -baseline_dur:0.0]
            aligned_epochs_df = aligned_epochs_df.sub(epoch_baselines.mean(axis=1), axis=0)

        self.aligned_pupil[event_name] = aligned_epochs_df


class Session:
    def __init__(self, sessname, ceph_dir, pkl_dir='X:\Dammy\ephys_pkls', ):
        self.pupil_obj = None
        self.lick_obj = None
        self.iti_zscore = None
        self.td_df = pd.DataFrame()
        self.sessname = sessname
        self.spike_obj = None
        self.sound_event_dict = {}
        self.decoders = {}
        self.ceph_dir = ceph_dir
        self.sound_writes_df = pd.DataFrame


    def init_spike_obj(self, spike_times_path, spike_cluster_path, start_time, parent_dir, ):
        """
        Initializes the spike_obj attribute with a SessionSpikes object.

        Args:
            spike_times_path (str): The file path to the spike times.
            spike_cluster_path (str): The file path to the spike clusters.
            start_time (int): The start time of the spikes.
            parent_dir (str): The parent directory of the spike files.
        """
        self.spike_obj = SessionSpikes(spike_times_path, spike_cluster_path, start_time, parent_dir)

    def init_sound_event_dict(self, sound_write_path, patterns=None):
        """
        Initialize a sound event dictionary using the given parameters.

        Args:
            sound_write_path: The path stem for the sound writes.
            sound_event_labels: Labels for the sound events.
            tone_space: The space between tones.
            harpbin_dir: The directory for the harp bins.
            normal: The normal pattern values.
            new_normal: The new normal pattern values.

        Returns:
            None
        """
        if not sound_write_path:
            assert self.sound_writes_df is not None
            sound_writes = self.sound_writes_df
        else:
            sound_writes = load_sound_bin(sound_write_path)
            sound_writes = format_sound_writes(sound_writes, patterns)
            self.sound_writes_df = sound_writes

        # assign sounds to trials
        # sound_writes['Trial_Number'] = np.full_like(sound_writes.index, -1)
        # sound_writes_diff = sound_writes['Time_diff'] = sound_writes['Timestamp'].diff()
        # print(sound_writes_diff[:10])
        # long_dt = np.squeeze(np.argwhere(sound_writes_diff.values > 1))
        # for n, idx in enumerate(long_dt):
        #     sound_writes.loc[idx:, 'Trial_Number'] = n
        # sound_writes['Trial_Number'] = sound_writes['Trial_Number'] + 1
        # if '240216a' in str(sound_write_path):
        #     sound_writes = sound_writes[sound_writes['Trial_Number'] > 30]
        # sound_writes_diff = sound_writes['Time_diff'] = sound_writes['Timestamp'].diff()
        # sound_writes['Payload_diff'] = sound_writes['Payload'].diff()
        # long_dt = np.squeeze(np.argwhere(sound_writes_diff.values > 1))
        #
        # tones = sound_writes['Payload'].unique()
        #
        # pattern_tones = tones[tones >= 8]
        # base_pip_idx = [e for e in pattern_tones if e not in sum(patterns, [])]
        # if len(base_pip_idx) > 1:
        #     base_pip_idx = sound_writes.query('Payload in @base_pip_idx')['Payload'].mode().iloc[0]
        # else:
        #     base_pip_idx = base_pip_idx[0]
        #
        # normal_patterns = [pattern for pattern in patterns if (np.diff(pattern) > 0).all()]
        # deviant_patterns = [pattern for pattern in patterns if pattern not in normal_patterns]
        #
        # all_pip_idxs = normal
        # sound_writes['pattern_pips'] = sound_writes['Payload'].isin(sum(patterns, []))
        # sound_writes['pattern_start'] = sound_writes['Payload_diff'].isin(np.array(patterns)[:, 0] - base_pip_idx) * \
        #                                 sound_writes['pattern_pips']
        # sound_writes['pip_counter'] = np.zeros_like(sound_writes['pattern_start']).astype(int)
        # for i in sound_writes.query('pattern_start == True').index:
        #     sound_writes.loc[i:i + 3, 'pip_counter'] = np.cumsum(sound_writes['pattern_pips'].loc[i:i + 3]) \
        #                                                * sound_writes['pattern_pips'].loc[i:i + 3]
        # print(base_pip_idx)
        base_pip_idx = [e for e in sound_writes['Payload'].unique() if e not in sum(patterns, []) and e >= 8]
        if len(base_pip_idx) > 1:
            base_pip_idx = sound_writes.query('Payload in @base_pip_idx')['Payload'].mode().iloc[0]
        else:
            base_pip_idx = base_pip_idx[0]
        patterns_df = pd.DataFrame(patterns, columns=list('ABCD'))
        event_sound_times = {}
        event_sound_idx = {}
        if 3 in sound_writes['Payload'].values:
            for idx, p in patterns_df.iterrows():
                for pi, pip in enumerate(p):
                    timesss = sound_writes.query(f'Payload == {pip} and pip_counter == {pi + 1} '
                                                 f'and ptype == @idx%2')['Timestamp']
                    if not timesss.empty:
                        event_sound_times[f'{p.index[pi]}-{idx}'] = timesss
                        event_sound_idx[f'{p.index[pi]}-{idx}'] = pip
        else:
            for lbl, pip in zip(patterns_df.columns, patterns_df.loc[0]):
                event_sound_times[f'{lbl}-0'] = \
                    sound_writes.query(f'Payload == {pip}')['Timestamp']
                event_sound_idx[f'{lbl}-0'] = pip
        # IMPLEMENT: trials with incomplete patterns

        # get trial start times
        long_dt = np.squeeze(np.argwhere(sound_writes['Time_diff'].values > 1))
        long_dt_df = sound_writes.iloc[long_dt]
        trial_start_times = long_dt_df['Timestamp']
        # pattern_idxs = [a_idx + i * tone_space for i in range(4)]

        sound_events = [3, base_pip_idx, -1]
        sound_event_labels = ['X', 'base', 'trial_start']

        # get baseline pip
        sound_writes['idx_diff'] = sound_writes['Payload'].diff()
        writes_to_use = np.all([sound_writes['Payload'] == base_pip_idx, sound_writes['idx_diff'] == 0,
                                sound_writes['Trial_Number'] > 10, sound_writes['Time_diff'] < 0.3], axis=0)
        base_pips_times = base_pip_times_same_prev = sound_writes[sorted(writes_to_use)]['Timestamp']
        if base_pips_times.empty:
            base_pips_times = sound_writes[sound_writes['Payload'] == base_pip_idx]['Timestamp']

        # event sound times
        if sound_writes[sound_writes['Payload'] == 3].empty:
            sound_events.remove(3), sound_event_labels.remove('X')
        for i, lbl in zip(sound_events, sound_event_labels):
            if i in [base_pip_idx, -1]:
                continue
            event_sound_times[lbl] = sound_writes.query('Payload == @i').drop_duplicates(subset='Trial_Number')[
                'Timestamp']
            event_sound_idx[lbl] = i

        if not base_pips_times.empty:
            idxss = np.random.choice(base_pips_times.index, min(base_pips_times.shape[0],
                                                                np.max([len(e) for e in event_sound_times.values()])),
                                     replace=False)
            event_sound_times['base'] = sound_writes.loc[idxss, 'Timestamp']
            event_sound_idx['base'] = base_pip_idx
        else:
            sound_event_labels.remove('base'), sound_events.remove(base_pip_idx)
        event_sound_times['trial_start'] = trial_start_times
        event_sound_idx['trial_start'] = -1

        assert len(event_sound_times) == len(event_sound_idx), Warning('sound event and labels must be equal length')
        for event_lbl in event_sound_times:
            self.sound_event_dict[event_lbl] = SoundEvent(event_sound_idx[event_lbl], event_sound_times[event_lbl],
                                                          event_lbl)

            self.sound_event_dict[event_lbl].trial_nums = np.squeeze(
                sound_writes.loc[event_sound_times[event_lbl].index, ['Trial_Number']].values)

    def get_event_free_zscore(self,event_free_time_window=2):
        spike_dict = self.spike_obj.cluster_spike_times_dict

        event_times = self.sound_writes_df['Timestamp'].values
        zscored_comp_plot = plt.subplots(figsize=(20, 10))
        mean_unit_mean = []
        mean_unit_std = []

        time_bin = 0.1
        event_free_time_bins = [get_times_in_window(event_times, [t-time_bin, t+time_bin]).size == 0
                                for t in np.arange(self.spike_obj.start_time, self.spike_obj.start_time
                                                   + self.spike_obj.duration-time_bin, time_bin)]
        event_free_time_bins = np.array(event_free_time_bins)
        print('old means')
        print(self.spike_obj.unit_means)
        self.spike_obj.unit_means = self.spike_obj.get_unit_mean_std(event_free_time_bins)
        print('new means')
        print(self.spike_obj.unit_means)

    def get_sound_psth(self, psth_window, baseline_dur=0.25, zscore_flag=False, redo_psth=False, to_exclude=(None,),
                       use_iti_zscore=True, **psth_kwargs):
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
        # if use_iti_zscore:
        self.get_iti_baseline()

        for e in tqdm(self.sound_event_dict, desc='get sound psth', total=len(self.sound_event_dict)):
            if e not in to_exclude:
                # iti_zscore = [ee[self.sound_event_dict[e].trial_nums-1] for ee in self.iti_zscore]  # if using mean why not use all trials
                self.sound_event_dict[e].get_psth(self.spike_obj, psth_window, title=f'session {self.sessname}',
                                                  baseline_dur=baseline_dur, zscore_flag=zscore_flag,
                                                  redo_psth=redo_psth,
                                                  iti_zscore=(self.iti_zscore if use_iti_zscore else None),
                                                  **psth_kwargs)
                # self.sound_event_dict[e].psth_plot[0].show()

    def reorder_psth(self, ref_name, reorder_list):
        if ref_name in list(self.sound_event_dict.keys()):
            for e in reorder_list:
                reord = self.sound_event_dict[e].psth[1].loc[self.sound_event_dict[ref_name].psth[2]]
                self.sound_event_dict[e].psth = (
                    self.sound_event_dict[e].psth[0], reord, self.sound_event_dict[e].psth[2])

    def get_iti_baseline(self):
        assert 'trial_start' in list(self.sound_event_dict.keys())
        # t_cols = np.linspace(-2, 0, 2001)
        # iti_psth = get_event_psth(self.spike_obj, -1, self.sound_event_dict['trial_start'].times, [-2, 0], 'iti',
        #                           baseline_dur=0, zscore_flag=False)[0]
        # iti_rate_mats = iti_psth
        # iti_mean = np.mean(iti_rate_mats, axis=2)
        # iti_std = np.std(iti_rate_mats, axis=2)
        #
        # z_df = pd.DataFrame([iti_mean.mean(axis=0),iti_std.mean(axis=0)]).T
        # z_df.columns = ['iti_mean','iti_std']
        # z_df['event_free_mean'],z_df['event_free_std'] = self.spike_obj.unit_means
        # z_df['global_mean'],z_df['global_std'] = self.spike_obj.get_unit_mean_std()
        trial_starts = self.sound_event_dict['trial_start'].times.values
        iti_means = []
        iti_stds = []
        all_spikes = self.spike_obj.cluster_spike_times_dict

        for ui, unit in tqdm(enumerate(all_spikes), total=len(all_spikes), desc='get_iti_baseline'):
            unit_spikes = [(all_spikes[unit][np.logical_and(all_spikes[unit] >= t-2,all_spikes[unit] <=t)].shape[0])
                           for t in trial_starts]
            unit_mean_rate = np.array([np.nanmean(e)/2 for e in unit_spikes])
            # nan_idx = np.logical_and(np.isnan(unit_mean_rate),np.isinf(unit_mean_rate))
            # unit_mean_rate[nan_idx] = 0
            # unit_std[nan_idx] = 0
            iti_means.append(unit_mean_rate)
            iti_stds.append(unit_mean_rate.std())
        iti_means = np.vstack(iti_means).T  # rate per 0.1 secs
        iti_stds = np.vstack(iti_stds).T
        # z_df['alt_iti_mean'],z_df['alt_iti_std'] = iti_means.mean(axis=0), iti_stds.T

        self.iti_zscore = iti_means.mean(axis=0), np.squeeze(iti_stds)

    def save_psth(self, keys='all', figdir=Path(r'X:\Dammy\figures\ephys'), **kwargs):
        if keys == 'all':
            keys2save = list(self.sound_event_dict.keys())
        else:
            keys2save = keys
        for key in keys2save:
            self.sound_event_dict[key].save_plot_as_svg(suffix=self.sessname, figdir=figdir)

    def pickle_obj(self, pkldir=Path(r'X:\Dammy\ephys_pkls')):
        if not pkldir.is_dir():
            pkldir.mkdir()
        with open(pkldir / f'{self.sessname}.pkl', 'wb') as pklfile:
            to_save = copy(self)
            to_save.spike_obj.event_spike_matrices = None
            to_save.spike_obj.event_cluster_spike_times = None
            pickle.dump(to_save, pklfile)

    def init_decoder(self, decoder_name: str, predictors, features, model_name='logistic'):
        self.decoders[decoder_name] = Decoder(predictors, features, model_name)

    def run_decoder(self, decoder_name, labels, plot_flag=False, decode_shuffle_flag=False, dec_kwargs=None, **kwargs):
        self.decoders[decoder_name].decode(dec_kwargs=dec_kwargs, **kwargs)
        if plot_flag:
            self.decoders[decoder_name].plot_decoder_accuracy(labels=labels, )

    def map_preds2sound_ts(self, sound_event, decoders, psth_window, window, map_kwargs={}):
        sound_event_ts = get_predictor_from_psth(self, sound_event, psth_window, window, mean=None, mean_axis=0)
        # with multiprocessing.Pool() as pool:
        #     prediction_ts_list = tqdm()
        prediction_ts_list = [np.array([self.decoders[decoder].map_decoding_ts(trial_ts, **map_kwargs)
                                        for decoder in decoders]) for trial_ts in sound_event_ts]
        prediction_ts = np.array(prediction_ts_list)
        return prediction_ts

    def load_trial_data(self, tdfile_path):
        # tddir_path = posix_from_win(tddir_path)
        # td_path = Path(td_home)/tddir_path/tdfile_path
        self.td_df = pd.read_csv(tdfile_path)
        self.get_n_since_last()
        self.get_local_rate()

    def get_n_since_last(self):
        self.td_df['n_since_last'] = np.arange(self.td_df.shape[0])
        since_last = self.td_df[self.td_df['Tone_Position'] == 0].index
        if not since_last.empty:
            for t, tt in zip(since_last, np.pad(since_last, [1, 0])):
                self.td_df.loc[tt + 1:t, 'n_since_last'] = self.td_df.loc[tt + 1:t, 'n_since_last'] - tt
            self.td_df.loc[t + 1:, 'n_since_last'] = self.td_df.loc[t + 1:, 'n_since_last'] - t

    def get_local_rate(self, window=5):
        self.td_df['local_rate'] = self.td_df['Tone_Position'].rolling(window=window).mean()

    def init_lick_obj(self, lick_times_path, sound_events_path, normal):
        licks = read_lick_times(lick_times_path)
        sound_events = pd.read_csv(sound_events_path)
        sound_events = format_sound_writes(sound_events, normal)
        self.lick_obj = SessionLicks(licks, sound_events)

    def get_licks_to_event(self, event_idx, event_name, window=(-3, 3), plot_kwargs=None):
        self.lick_obj.get_event_spikes(event_idx, event_name, window, self.sessname)
        self.lick_obj.plot_licks(event_name, window)

    def init_pupil_obj(self, pupil_data, sound_events_path, beh_events_path, normal) -> SessionPupil:
        sound_events = pd.read_csv(sound_events_path)
        beh_events = pd.read_csv(beh_events_path)
        sound_events = format_sound_writes(sound_events, normal)
        self.pupil_obj = SessionPupil(pupil_data, sound_events, beh_events)

    def get_pupil_to_event(self, event_idx, event_name, window=(-3, 3), align_kwargs=None, plot_kwargs=None):
        self.pupil_obj.align2events(event_idx, event_name, window, self.sessname,
                                    **align_kwargs if align_kwargs else {})
        # self.pupil_obj.plot_pupil(event_name, window)


def get_predictor_from_psth(sess_obj: Session, event_key, psth_window, new_window, mean=np.mean, mean_axis=2,
                            use_iti_zscore=False,use_unit_zscore=True) -> np.ndarray:
    """
    Generate a predictor array from a PSTH (peristimulus time histogram) using the provided session object, event key,
    PSTH window, new window, and optional mean and mean axis parameters. Return the predictor array as a NumPy array.
    """
    assert not (use_iti_zscore and use_unit_zscore), 'Can only use one zscore option at a time'
    # td_start, td_end = [timedelta(0, t) for t in new_window]
    event_arr = sess_obj.sound_event_dict[event_key].psth[0]
    event_arr_tseries = np.linspace(psth_window[0], psth_window[1], event_arr.shape[-1])
    time_window_idx = np.logical_and(event_arr_tseries >= new_window[0], event_arr_tseries <= new_window[1])

    # event_arr_2d = event_arr.reshape(-1, event_arr.shape[2])
    # predictor_rate_mat = gen_firing_rate_matrix(pd.DataFrame(event_arr_2d,
    #                                                          columns=np.linspace(psth_window[0], psth_window[1],
    #                                                                              event_arr_2d.shape[1])),
    #                                             baseline_dur=0, ).loc[:, td_start:td_end]
    # predictor = predictor_rate_mat.values.reshape(
    #     (-1, event_arr.shape[1], predictor_rate_mat.shape[1]))  # units x trials x t steps
    predictor = event_arr[:, :, time_window_idx]
    if use_iti_zscore:
        predictor = zscore_by_unit(predictor, sess_obj.iti_zscore[0], sess_obj.iti_zscore[1])
    if use_unit_zscore:
        predictor = zscore_by_unit(predictor, sess_obj.spike_obj.unit_means[0], sess_obj.spike_obj.unit_means[0])
    if mean:
        predictor = mean(predictor, axis=mean_axis)

    return predictor


def zscore_w_iti(rate_arr, iti_mean, iti_std):
    iti_mean[iti_mean == 0] = np.nan
    iti_std[iti_std == 0] = np.nan
    # ratemat_arr3d = np.transpose((ratemat_arr3d.T - iti_zscore[0].T)/iti_zscore[1].T)
    # for ti in range(iti_zscore[0].shape[0]):
    for ui in range(iti_mean.shape[1]):
        rate_arr[:, ui] = (rate_arr[:, ui] - np.nanmean(iti_mean, axis=0)[ui]) / \
                          np.nanmean(iti_std, axis=0)[ui]
    assert np.isnan(rate_arr).sum() == 0
    return rate_arr


def zscore_by_unit(rate_arr,unit_means,unit_stds):
    f,a = plt.subplots(ncols=2)
    # plot_2d_array_with_subplots(rate_arr, plot=(f, a[0]))
    if unit_means.ndim == 2:
        unit_means[unit_means == 0] = np.nan
        unit_stds[unit_stds == 0] = np.nan
        unit_means, unit_stds = [np.nanmean(e,axis=0) for e in [unit_means, unit_stds]]

    assert rate_arr.shape[-2] == unit_means.shape[0]
    assert rate_arr.ndim in [2,3]

    if isinstance(rate_arr, pd.DataFrame):
        rate_arr_df = rate_arr
        rate_arr = rate_arr.values
    else:
        rate_arr_df = None
    for ui, (u_mean, u_std) in enumerate(zip(unit_means, unit_stds)):
        if rate_arr.ndim == 3:
            rate_arr[:, ui, :] = (rate_arr[:, ui, :] - u_mean) / u_std
        else:
            rate_arr[ui, :] = (rate_arr[ui, :] - u_mean) / u_std
    # print(f'{u_mean, u_std = }')
    assert np.isnan(rate_arr).sum() == 0
    # plot_2d_array_with_subplots(rate_arr, plot=(f, a[1]))
    # f.show()
    if isinstance(rate_arr_df, pd.DataFrame):
        rate_arr = pd.DataFrame(rate_arr, columns=rate_arr_df.columns, index=rate_arr_df.index)
    return rate_arr


def balance_predictors(list_predictors, list_features) -> [[np.ndarray], [np.ndarray]]:
    min_pred_len = min([e.shape[0] for e in list_predictors])
    idx_subsets = [np.random.choice(e.shape[0], min_pred_len, replace=False) for e in list_predictors]
    # assert len(np.unique(idx_subset)) == min_pred_len
    predictors = [e[idxs] for e, idxs in zip(list_predictors, idx_subsets)]
    features = [e[idxs] for e, idxs in zip(list_features, idx_subsets)]

    return np.vstack(predictors), np.hstack(features)


def find_ramp_start(signal: np.ndarray, ):
    t_peak = np.argmax(signal)
    d2_signal = np.diff(savgol_filter(np.diff(signal[:t_peak]), 50, 2))
    return d2_signal[d2_signal > d2_signal.mean() + 2 * d2_signal.std()]


def get_main_sess_patterns(name='', date='', main_sess_td_name='', home_dir=Path(''),td_df=None) -> [int,int,int,int]:
    if isinstance(td_df, pd.DataFrame):
        main_sess_td = td_df
    else:
        main_sess_td = get_main_sess_td_df(name, date, main_sess_td_name, home_dir)[0]

    # try:main_pattern = main_sess_td.query('Session_Block>=0 & Tone_Position==0')['PatternID'].mode().iloc[0]
    # except IndexError: main_pattern = None
    main_patterns = main_sess_td.query('Session_Block>=0 & Tone_Position==0')['PatternID'].unique()
    main_patterns = [[int(e) for e in main_pattern.split(';')] for main_pattern in main_patterns]

    return sorted(main_patterns, key=lambda x: x[0])


def get_main_sess_td_df(name, date, main_sess_td_name, home_dir):
    td_path_pattern = 'data/Dammy/<name>/TrialData'
    td_path = td_path_pattern.replace('<name>', main_sess_td_name.split('_')[0])
    abs_td_path_dir = home_dir / td_path
    abs_td_path = next(abs_td_path_dir.glob(f'{name}_TrialData_{date}*.csv'))
    print(f'{abs_td_path = }')
    main_sess_td = pd.read_csv(abs_td_path)
    return main_sess_td, abs_td_path


def get_decoder_accuracy(sess_obj: Session, decoder_name):
    return sess_obj.decoders[decoder_name].accuracy


def load_sess_pkl(pkl_path):
    with open(pkl_path, 'rb') as pklfile:
        sess_obj: Session = pickle.load(pklfile)
    return sess_obj


def get_decoder_accuracy_from_pkl(pkl_path):
    try:
        sess_obj = load_sess_pkl(pkl_path)
    except:
        print(f'{pkl_path} error')
        return None
    print(f'extracting decoder accuracies from {pkl_path}')
    res = [get_decoder_accuracy(sess_obj, decoder_name) for decoder_name in sess_obj.decoders.keys()]
    keys = list(sess_obj.decoders.keys())
    return dict(zip(keys, res))


def get_property_from_decoder_pkl(pkl_path: str, decoder_name: str, property_name: str):
    try:
        sess_obj = load_sess_pkl(pkl_path)
    except:
        print(f'{pkl_path} error')
        return None
    return getattr(sess_obj.decoders[decoder_name], property_name)


def get_pip_desc(pip: str, pip_idx_dict: dict, pip_lbl_dict: dict, n_per_rule: int) -> [int, int, str, str, int,str]:
    position = ord(pip.split('-')[0]) - ord('A') + 1
    idx = pip_idx_dict[pip]
    name = pip_lbl_dict[idx]
    ptype = 'ABCD' if int(pip.split('-')[1]) % n_per_rule==0 else 'ABBA'
    ptype_i = 0 if int(pip.split('-')[1]) % n_per_rule==0 else 1
    group = int(int(pip.split('-')[1]) / n_per_rule)
    full_desc = f'pip {position}\n {ptype}({group})'
    return {'position': position, 'idx': idx, 'name': name, 'ptype': ptype,'ptype_i': ptype_i,
            'group': group, 'desc': full_desc}


def permute_pip_preds(preds_dict:dict):
    n_by_pip = [preds_dict[e].shape[0] for e in preds_dict]
    all_pip_arr = np.vstack(list(preds_dict.values()))
    all_pip_arr_shuffled = all_pip_arr[np.random.permutation(all_pip_arr.shape[0])]
    shuffled_split = np.array_split(all_pip_arr_shuffled, np.cumsum(n_by_pip))[:-1]
    return shuffled_split


def get_pip_info(sound_event_dict,normal_patterns,n_patts_per_rule):
    pip_idxs = {event_lbl: sound_event_dict[event_lbl].idx
                for event_lbl in sound_event_dict
                if any(char in event_lbl for char in 'ABCD')}
    pip_lbls = list(pip_idxs.keys())
    pip_desc = {pip: {} for pip in pip_idxs}  # position in pattern, label, index
    pip_names = {idx: lbl for idx, lbl in zip(sum(normal_patterns, []), 'ABCD' * len(normal_patterns))}
    [pip_desc.update({pip: get_pip_desc(pip, pip_idxs, pip_names, n_patts_per_rule)}) for pip in pip_idxs]
    return pip_desc, pip_lbls,pip_names
