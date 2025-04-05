import platform
import warnings

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
import multiprocessing
from functools import partial
from mpl_toolkits.axes_grid1 import make_axes_locatable
from elephant.statistics import instantaneous_rate
from quantities import s, ms
from neo import SpikeTrain
from elephant.kernels import GaussianKernel, ExponentialKernel
from generate_synthetic_spikes import gen_responses, gen_patterned_unit_rates, gen_patterned_time_offsets


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
        try:
            e_dir = next((ceph_dir / posix_from_win(sess['ephys_dir'])).rglob('continuous')).parent
            bin_stem = sess[col_name]
            dir_files = list(e_dir.iterdir())
            if 'metadata.json' not in dir_files or True:
                if harp_bin_dir and isinstance(harp_bin_dir, Path):
                    harp_bin = harp_bin_dir / f'{bin_stem}_write_data.csv'
                else:
                    bin_path = Path(bin_stem)
                    harp_bin = bin_path.with_stem(f'{bin_path.stem}_write_data').with_suffix('.csv')
                harp_writes = pd.read_csv(harp_bin)
                trigger_time = harp_writes[harp_writes['DO3'] == True]['Times'].values[0]
                metadata = {'trigger_time': trigger_time}
                with open(e_dir / 'metadata.json', 'w') as jsonfile:
                    json.dump(metadata, jsonfile)
        except:
            continue


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
        print(f'parent_dir = {parent_dir}')
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
        pupil_data = {}
        while True:
            try:
                y = (pickle.load(f))
                y = {k: y[k] for k in y.keys() if y[k].pupildf is not None}
                z = {**pupil_data, **y}
                pupil_data = z
            except EOFError:
                print(f'end of file {pupil_data.keys()}')
                break
    for sess in list(pupil_data.keys()):
        if pupil_data[sess].pupildf is None:
            pupil_data.pop(sess)
        else:
            pupil_data[sess].pupildf.index = pupil_data[sess].pupildf.index
    return pupil_data


def check_unique_across_dim(arr: [np.ndarray, list], dim2check=0):
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


def unique_legend(plotfig: (plt.figure().figure, list, tuple), **leg_kwargs):
    if isinstance(plotfig, (tuple, list)):
        if isinstance(plotfig[1], np.ndarray):
            plotaxes2use = plotfig[1].flatten()
        elif isinstance(plotfig[1], dict):
            plotaxes2use = plotfig[1].values()
        else:
            print('wrong figure used, returning none')
            plotaxes2use = None
    elif isinstance(plotfig, np.ndarray):
        plotaxes2use = plotfig.flatten()
    elif isinstance(plotfig[1], dict):
        plotaxes2use = plotfig[1].values()
    else:
        plotaxes2use = None
        print('wrong figure used, returning none')
    for axis in plotaxes2use:
        handle, label = axis.get_legend_handles_labels()
        axis.legend(pd.Series(handle).unique(), pd.Series(label).unique(), **leg_kwargs)


def load_sound_bin(binpath: Path):
    all_writes = pd.read_csv(binpath)
    return all_writes


def format_sound_writes(sound_writes_df: pd.DataFrame, patterns: [[int, ], ], normal_patterns=None,
                        ptypes=None) -> pd.DataFrame:
    sound_writes_df = sound_writes_df.drop_duplicates(subset='Timestamp', keep='first').copy()
    sound_writes_df = sound_writes_df.reset_index(drop=True)
    sound_writes_df['rand_n'] = np.random.randint(0, 8, len(sound_writes_df)).astype(int)
    sound_writes_df['Trial_Number'] = np.full_like(sound_writes_df.index, -1)
    sound_writes_diff = sound_writes_df['Time_diff'] = sound_writes_df['Timestamp'].diff()
    if 3 in sound_writes_df['Payload'].values:
        long_dt = np.squeeze(np.argwhere(sound_writes_diff.values > 2))
    else:
        long_dt = np.squeeze(np.argwhere(sound_writes_diff.values > 1))

    for n, idx in enumerate(long_dt):
        sound_writes_df.loc[idx:, 'Trial_Number'] = n
    sound_writes_df['Trial_Number'] = sound_writes_df['Trial_Number'] + 1

    sound_writes_df['Time_diff'] = sound_writes_df['Timestamp'].diff()
    sound_writes_df['Payload_diff'] = sound_writes_df['Payload'].diff()
    matrix_d_X_times = np.array(np.matrix(sound_writes_df['Timestamp'].values).T -
                                sound_writes_df.query('Payload == 3')['Timestamp'].values)
    matrix_d_X_times[matrix_d_X_times > 0] = 9999
    matrix_d_X_times = np.min(np.abs(matrix_d_X_times), axis=1)
    sound_writes_df['d_X_times'] = matrix_d_X_times

    tones = sound_writes_df['Payload'].unique()
    pattern_tones = tones[tones >= 8]
    if patterns is not None:
        base_pip_idx = [e for e in pattern_tones if e not in sum(patterns, [])]
        if len(base_pip_idx) > 1:
            base_pip_idx = sound_writes_df.query('Payload in @base_pip_idx')['Payload'].mode().iloc[0]
        else:
            base_pip_idx = base_pip_idx[0]
    else:
        base_pip_idx = sound_writes_df['Payload'].mode().iloc[0]

    if patterns is None:
        normal_patterns, deviant_patterns = None, None
        sound_writes_df['pattern_pips'] = [np.nan] * sound_writes_df.shape[0]
        sound_writes_df['pattern_start'] = [np.nan] * sound_writes_df.shape[0]
        sound_writes_df['pip_counter'] = [np.nan] * sound_writes_df.shape[0]
        sound_writes_df['ptype'] = [np.nan] * sound_writes_df.shape[0]
    else:
        if normal_patterns is None:
            normal_patterns = [pattern for pattern in patterns if (np.diff(pattern) > 0).all()]
        deviant_patterns = [pattern for pattern in patterns if pattern not in normal_patterns]

        sound_writes_df['pattern_pips'] = sound_writes_df['Payload'].isin(sum(patterns, []))
        sound_writes_df['pattern_start'] = sound_writes_df['Payload_diff'].isin(np.array(patterns)[:, 0] - base_pip_idx) * \
                                           sound_writes_df['pattern_pips']
        sound_writes_df['pip_counter'] = np.zeros_like(sound_writes_df['pattern_start']).astype(int)
        patterns_presentations = [sound_writes_df.iloc[idx:idx + 4]
                                  for idx in sound_writes_df.query('pattern_start == True').index
                                  if base_pip_idx not in sound_writes_df.iloc[idx:idx + 4]['Payload'].values
                                  and 3 not in sound_writes_df.iloc[idx:idx + 4]['Payload'].values]
        sound_writes_df['ptype'] = [np.nan] * sound_writes_df.shape[0]
        for pattern_pres in patterns_presentations:
            patt = pattern_pres['Payload'].to_list()
            if ptypes is not None:
                ptype = ptypes['_'.join(str(e) for e in patt)]
            else:
                ptype = 0 if patt in normal_patterns else 1 if patt in deviant_patterns else -1
            sound_writes_df.loc[pattern_pres.index, 'ptype'] = ptype

        for i in sound_writes_df.query('pattern_start == True').index:
            sound_writes_df.loc[i:i + 3, 'pip_counter'] = np.cumsum(sound_writes_df['pattern_pips'].loc[i:i + 3]) \
                                                          * sound_writes_df['pattern_pips'].loc[i:i + 3]

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
        nbins = np.ceil(len(y) / 6).astype(int)

    x = np.zeros(len(y))

    nn, ybins = np.histogram(y, bins=nbins)
    nmax = nn.max()

    ibs = []
    for ymin, ymax in zip(ybins[:-1], ybins[1:]):
        i = np.nonzero((y > ymin) * (y <= ymax))[0]
        ibs.append(i)

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


def plot_2d_array_with_subplots(array_2d: np.ndarray, cmap='viridis', cbar_width=0.1, cbar_height=0.8,
                                extent=None, aspect=1.0, vcenter=None, plot=None,
                                **im_kwargs) -> (plt.Figure, plt.Axes):
    plot_cbar = im_kwargs.get('plot_cbar', True)
    im_kwargs.pop('plot_cbar', None)

    array_2d = np.array(array_2d)

    if not plot:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot
    divider = make_axes_locatable(ax)
    if vcenter:
        im = ax.imshow(array_2d, cmap=cmap, extent=extent,
                       norm=matplotlib.colors.TwoSlopeNorm(vcenter=vcenter, ))
    else:
        im = ax.imshow(array_2d, cmap=cmap, extent=extent, **im_kwargs)
    ax.set_aspect('auto')

    if plot_cbar:
        cax = divider.append_axes('right', size='7.5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, fraction=cbar_width, aspect=cbar_height, )
    else:
        cbar = None
    return fig, ax, cbar


def predict_1d(models, t_ser, y_lbl=0):
    prediction_ts = np.array([m.predict(t_ser.T) for m in models])
    if isinstance(y_lbl, list):
        assert len(y_lbl) == prediction_ts.shape[0]
        _arr = np.array(row == lbl for row, lbl in zip(prediction_ts, y_lbl))
        prediction_ts = _arr
    return prediction_ts


def load_session_pkls(pkl_path):
    with open(pkl_path, 'rb') as pklfile:
        sess_obj = pickle.load(pklfile)
    return sess_obj


class SpikeAnalysis:
    def __init__(self, spike_times, spike_clusters, fs, resample_fs):
        self.spike_times = spike_times
        self.spike_clusters = spike_clusters
        self.fs = fs
        self.resample_fs = resample_fs
        self.cluster_spike_times_dict = self._cluster_spike_times()

    def _cluster_spike_times(self):
        cluster_spike_times_dict = {}
        for cluster_id in np.unique(self.spike_clusters):
            cluster_spike_times_dict[cluster_id] = self.spike_times[self.spike_clusters == cluster_id]
        return cluster_spike_times_dict

    def get_spike_times_in_window(self, event_time, window):
        window_spikes_dict = {}
        for cluster_id, spikes in self.cluster_spike_times_dict.items():
            all_spikes = spikes - event_time
            window_spikes_dict[cluster_id] = all_spikes[(all_spikes >= window[0]) & (all_spikes <= window[1])]
        return window_spikes_dict

    def generate_spike_matrix(self, window, kernel_width=40):
        time_cols = np.arange(window[0], window[1], 1 / self.resample_fs)
        spike_matrix = pd.DataFrame(index=self.cluster_spike_times_dict.keys(), columns=time_cols)
        for cluster_id, spikes in self.cluster_spike_times_dict.items():
            spike_train = SpikeTrain(spikes * s, t_start=window[0], t_stop=window[1] * s)
            firing_rate = instantaneous_rate(spike_train, sampling_period=10 * ms, kernel=ExponentialKernel(kernel_width * ms))
            spike_matrix.loc[cluster_id] = firing_rate
        return spike_matrix


class PupilAnalysis:
    def __init__(self, pupil_data):
        self.pupil_data = pupil_data

    def align_to_events(self, event_times, window, size_col='dlc_radii_a_zscored'):
        aligned_epochs = []
        for event_time in event_times:
            epoch = self.pupil_data[size_col].loc[event_time + window[0]: event_time + window[1]]
            if not epoch.empty:
                aligned_epochs.append(epoch)
        return pd.DataFrame(aligned_epochs)


class LickAnalysis:
    def __init__(self, lick_times):
        self.lick_times = lick_times

    def align_to_events(self, event_times, window):
        aligned_licks = []
        for event_time in event_times:
            licks = self.lick_times[(self.lick_times >= event_time + window[0]) & (self.lick_times <= event_time + window[1])]
            aligned_licks.append(licks)
        return aligned_licks


class DecoderAnalysis:
    def __init__(self, predictors, features, model_name='logistic'):
        self.predictors = predictors
        self.features = features
        self.model_name = model_name
        self.models = []
        self.accuracy = []

    def run_decoder(self, n_runs=100):
        for _ in range(n_runs):
            model = run_decoder(self.predictors, self.features, model=self.model_name)
            self.models.append(model)
            self.accuracy.append(model.score(self.predictors, self.features))
        return self.accuracy

    def plot_accuracy(self):
        plt.plot(self.accuracy)
        plt.title('Decoder Accuracy')
        plt.xlabel('Run')
        plt.ylabel('Accuracy')
        plt.show()