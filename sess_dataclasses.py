import pickle
import warnings
from copy import deepcopy as copy
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

import plot_funcs
from decoding_funcs import Decoder
from generate_synthetic_spikes import gen_patterned_unit_rates, gen_patterned_time_offsets
from io_utils import load_sound_bin, format_sound_writes, read_lick_times

from plot_funcs import plot_psth, plot_2d_array_with_subplots
from spike_time_utils import SessionSpikes, get_event_psth, cluster_spike_times, \
    get_spike_times_in_window, gen_spike_matrix, get_times_in_window


class SoundEvent:
    def __init__(self, idx, times, lbl):
        self.idx = idx
        self.times = times
        self.lbl = lbl

        self.psth = None
        self.psth_plot = None
        self.synth_params = None

    def get_psth(self, sess_spike_obj: SessionSpikes, window, title='', redo_psth=False, redo_psth_plot=False,
                 baseline_dur=0.25, zscore_flag=False, iti_zscore=None, reorder_idxs=None, synth_data=None):

        if not self.psth or redo_psth:
            self.psth = get_event_psth(sess_spike_obj, self.idx, self.times, window, self.lbl,
                                       baseline_dur=baseline_dur, zscore_flag=zscore_flag, iti_zscore=None,
                                       synth_data=synth_data, synth_params=self.synth_params if synth_data else None, )
            if iti_zscore:
                self.psth = (self.psth[0],
                             zscore_by_unit(self.psth[1], unit_means=iti_zscore[0], unit_stds=iti_zscore[1]),
                             self.psth[2])
                # self.psth[1][self.psth[1] == np.nan] = 0.0
            if zscore_flag:
                means,stds = sess_spike_obj.unit_means
                self.psth = (self.psth[0], zscore_by_unit(self.psth[1], means, stds), self.psth[2])

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

    def get_event_spikes_slow(self, event_idx: int, event_name: str, window: [list | np.ndarray], sessname: str,
                         sound_df_query='',**kwargs):
        sound_df_query = f'Payload == {event_idx} & {sound_df_query}' if sound_df_query else f'Payload == {event_idx}'
        event_times = self.sound_writes.query(sound_df_query)['Timestamp'].values
        event_trialnum = self.sound_writes.query(sound_df_query)['Trial_Number'].values
        self.event_spike_matrices[event_name] = dict()
        self.event_cluster_spike_times[event_name] = dict()
        for event_time in event_times:
            if f'{event_name}_{event_time}' not in list(self.event_cluster_spike_times.keys()):
                self.event_cluster_spike_times[event_name][f'{event_name}_{event_time}'] = get_spike_times_in_window(
                    event_time, self.cluster_spike_times_dict, window, self.new_fs)
            if f'{event_name}_{event_time}' not in list(self.event_spike_matrices.keys()):
                self.event_spike_matrices[event_name][f'{event_name}_{event_time}'] = gen_spike_matrix(
                    self.event_cluster_spike_times[event_name][f'{event_name}_{event_time}'],
                    window, self.new_fs, kwargs.get('kernel_width',40))
        # self.event_licks[f'{event_name}_licks'] = pd.concat([self.event_spike_matrices[e_key] for e_key in self.event_spike_matrices
        #                                                      if event_name in e_key])
        self.event_licks[f'{event_name}_licks'] = pd.concat(self.event_spike_matrices[event_name])
        # multi index for licks
        self.event_licks[f'{event_name}_licks'].index = pd.MultiIndex.from_arrays([event_times, event_trialnum,
                                                                                   [sessname] * len(event_trialnum)],
                                                                                  names=['time', 'trial', 'sess'])

    def get_event_spikes(self, event_idx: int, event_name: str, window: [list | np.ndarray], sessname: str,
                         sound_df_query='', **kwargs):

        # Construct the query string
        query_str = f'Payload == {event_idx}'
        if sound_df_query:
            query_str += f' & {sound_df_query}'

        # Query the sound writes DataFrame once
        queried_sound_writes = self.sound_writes.query(query_str)
        event_times = queried_sound_writes['Timestamp'].values
        event_trialnum = queried_sound_writes['Trial_Number'].values

        # Initialize the dictionaries if they do not exist
        if event_name not in self.event_spike_matrices:
            self.event_spike_matrices[event_name] = {}
        if event_name not in self.event_cluster_spike_times:
            self.event_cluster_spike_times[event_name] = {}

        # Precompute kernel width
        kernel_width = kwargs.get('kernel_width', 40)

        # Get unique event times that need to be processed
        existing_keys = self.event_cluster_spike_times[event_name].keys()

        new_event_times = [et for et in event_times if f'{event_name}_{et}' not in existing_keys]

        # Process each unique event time
        for event_time in new_event_times:
            event_key = f'{event_name}_{event_time}'

            # Get spike times in the window
            self.event_cluster_spike_times[event_name][event_key] = get_spike_times_in_window(
                event_time, self.cluster_spike_times_dict, window, self.new_fs)

            # Generate spike matrix
            self.event_spike_matrices[event_name][event_key] = gen_spike_matrix(
                self.event_cluster_spike_times[event_name][event_key], window, self.new_fs, kernel_width)

        # Concatenate spike matrices
        self.event_licks[f'{event_name}_licks'] = pd.concat(self.event_spike_matrices[event_name].values(), axis=0)

        # Create multi-index for licks
        multi_index = pd.MultiIndex.from_arrays([event_times, event_trialnum, [sessname] * len(event_trialnum)],
                                                names=['time', 'trial', 'sess'])
        self.event_licks[f'{event_name}_licks'].index = multi_index

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
                     sound_df_query='', baseline_dur=0.0, size_col='dlc_radii_a_zscored',event_shifts=None):
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
        pupil_isout = self.pupil_data['isout']
        start, end = pupil_size.index[0], pupil_size.index[-1]
        dt = np.round(np.nanmedian(np.diff(pupil_size.index)),2)
        event_tdeltas = np.round(np.arange(window[0], window[1], dt), 2)

        sound_df_query = f'Payload == {event_idx} & {sound_df_query}' if sound_df_query else f'Payload == {event_idx}'
        event_times = self.sound_writes.query(sound_df_query)['Timestamp'].values
        event_trialnums = self.sound_writes.query(sound_df_query)['Trial_Number'].values
        if event_shifts is not None:
            event_times += event_shifts

        # aligned_epochs = [[pupil_size.loc[eventtime + tt], np.nan) for tt in event_tdeltas]
        #                   for eventtime in event_times]
        # aligned_epochs = [pupil_size.loc[eventtime+window[0]: eventtime+window[1]] for eventtime in event_times]
        aligned_epochs = []
        epoch_isout = []
        for eventtime in event_times:
            try:
                a = pupil_size.loc[eventtime + window[0]: eventtime + window[1]]
            except KeyError:
                print(f'{sessname}: {start<eventtime<end= } {eventtime}  {event_name} not found. '
                      f'Event {np.where(event_times==eventtime)[0][0]+1}/{len(event_times)}')
                continue
            if a.isna().all():
                print(f'{sessname}: {start<eventtime<end= } {eventtime} {event_name} not found. '
                      f'Event {np.where(event_times == eventtime)[0][0] + 1}/{len(event_times)}')
                aligned_epochs.append(pd.Series(np.full_like(event_tdeltas, np.nan)))
            else:
                if pupil_isout.loc[eventtime + window[0]: eventtime + window[1]].mean() > 1.5:
                    warnings.warn(f'{sessname}: {eventtime} {event_name} has many outliers. Not using')
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
        aligned_epochs_df = aligned_epochs_df.interpolate(limit_direction='both',axis=1 )
        # aligned_epochs_df = aligned_epochs_df.ffill(axis=1, )
        # aligned_epochs_df = aligned_epochs_df.bfill(axis=1, )
        if baseline_dur:
            epoch_baselines = aligned_epochs_df.loc[:, -baseline_dur:0.0]
            aligned_epochs_df = aligned_epochs_df.sub(epoch_baselines.mean(axis=1), axis=0)

        self.aligned_pupil[event_name] = aligned_epochs_df

    def align2events_w_td_df(self,td_df: pd.DataFrame, td_df_column: str, td_df_query: str, window, sessname: str,
                     event_idx: int, event_name: str,
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
        dt = np.round(np.nanmedian(np.diff(pupil_size.index)), 2)
        event_tdeltas = np.round(np.arange(window[0], window[1], dt), 2)

        event_trials = td_df.query(td_df_query)
        # convert to harp time
        if 'Bonsai_Time_dt' in event_trials.columns:
            event_trials_tdeltas = event_trials.eval(f'{td_df_column} - Bonsai_Time_dt',inplace=False)
            event_times = event_trials['Harp_Time'] + event_trials_tdeltas.dt.total_seconds()
        else:
            # assume index is frame time as secs from 1900 1 1
            eventtimes_w_tz = [dt.to_pydatetime().astimezone() for dt in event_trials[td_df_column]]
            eventtimes_as_ts = [dt.replace(tzinfo=timezone.utc).timestamp() for dt in eventtimes_w_tz]
            event_times = eventtimes_as_ts


        event_trialnums = event_trials.index.get_level_values('trial_num')

        aligned_epochs = []
        for eventtime in event_times:
            a = pupil_size.loc[eventtime + window[0]: eventtime + window[1]]
            if a.isna().all():
                print(f'{sessname}: {start<eventtime<end= } {eventtime} not found')
                aligned_epochs.append(pd.Series(np.full_like(event_tdeltas, np.nan)))
            else:
                aligned_epochs.append(pupil_size.loc[eventtime + window[0]: eventtime + window[1]])
        for aligned_epoch, eventtime in zip(aligned_epochs, event_times):
            aligned_epoch.index = np.round(aligned_epoch.index - eventtime, 2)
        sessname_list = [sessname] * len(event_times)
        try:
            aligned_epochs_df = pd.DataFrame(aligned_epochs, columns=event_tdeltas,
                                             index=pd.MultiIndex.from_tuples(list(zip(event_times, event_trialnums,
                                                                                      sessname_list)),
                                                                             names=['time', 'trial', 'sess']))
        except pd.errors.InvalidIndexError:
            print(f'{sessname} error')
            print(event_times, event_trialnums, sessname_list)
            raise pd.errors.InvalidIndexError
        # aligned_epochs_df = pd.DataFrame(aligned_epochs, columns=event_tdeltas,
        #                                  index=pd.MultiIndex.from_tuples(list(zip(event_times, event_trialnums,
        #                                                                           sessname_list)),
        #                                                                  names=['time', 'trial', 'sess']))
        aligned_epochs_df = aligned_epochs_df.dropna(axis=0, how='all')
        # for col in aligned_epochs_df.columns:
        #     aligned_epochs_df[col] = pd.to_numeric(aligned_epochs_df[col],errors='coerce')
        aligned_epochs_df = aligned_epochs_df.ffill(axis=1, )
        aligned_epochs_df = aligned_epochs_df.bfill(axis=1, )
        if baseline_dur:
            epoch_baselines = aligned_epochs_df.loc[:, -baseline_dur:0.0]
            aligned_epochs_df = aligned_epochs_df.sub(epoch_baselines.mean(axis=1), axis=0)

        self.aligned_pupil[event_name] = aligned_epochs_df

    def align2times(self,event_times,size_col,window, sessname, event_name, baseline_dur=0.0):

        pupil_size = self.pupil_data[size_col]
        pupil_isout = self.pupil_data['isout']
        start, end = pupil_size.index[0], pupil_size.index[-1]
        dt = np.round(np.nanmedian(np.diff(pupil_size.index)), 2)
        event_tdeltas = np.round(np.arange(window[0], window[1], dt), 2)

        event_trialnums = np.full_like(event_times, np.nan)
        # aligned_epochs = [[pupil_size.loc[eventtime + tt], np.nan) for tt in event_tdeltas]
        #                   for eventtime in event_times]
        # aligned_epochs = [pupil_size.loc[eventtime+window[0]: eventtime+window[1]] for eventtime in event_times]
        aligned_epochs = []
        epoch_isout = []
        for eventtime in event_times:
            try:
                a = pupil_size.loc[eventtime + window[0]: eventtime + window[1]]
            except KeyError:
                print(f'{sessname}: {start<eventtime<end= } {eventtime}  {event_name} not found. '
                      f'Event {np.where(event_times == eventtime)[0][0] + 1}/{len(event_times)}')
                continue
            if a.isna().all():
                print(f'{sessname}: {start<eventtime<end= } {eventtime} {event_name} not found. '
                      f'Event {np.where(event_times == eventtime)[0][0] + 1}/{len(event_times)}')
                aligned_epochs.append(pd.Series(np.full_like(event_tdeltas, np.nan)))
            else:
                if pupil_isout.loc[eventtime + window[0]: eventtime + window[1]].mean() > 1.5:
                    warnings.warn(f'{sessname}: {eventtime} {event_name} has many outliers. Not using')
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
        aligned_epochs_df = aligned_epochs_df.interpolate(limit_direction='both', axis=1)
        # aligned_epochs_df = aligned_epochs_df.ffill(axis=1, )
        # aligned_epochs_df = aligned_epochs_df.bfill(axis=1, )
        if baseline_dur:
            epoch_baselines = aligned_epochs_df.loc[:, -baseline_dur:0.0]
            aligned_epochs_df = aligned_epochs_df.sub(epoch_baselines.mean(axis=1), axis=0)

        self.aligned_pupil[event_name] = aligned_epochs_df


class Session:
    def __init__(self, sessname, ceph_dir, pkl_dir='X:\Dammy\ephys_pkls', ):
        self.pupil_obj = None
        self.lick_obj = None
        self.iti_zscore = None
        self.pip_desc =  None
        self.td_df = pd.DataFrame()
        self.sessname = sessname
        self.spike_obj = None
        self.sound_event_dict = {}
        self.beh_event_dict = {}
        self.decoders = {}
        self.ceph_dir = ceph_dir
        self.sound_writes_df = pd.DataFrame

    def init_spike_obj(self, spike_times_path, spike_cluster_path, start_time, parent_dir,**kwargs ):
        """
        Initializes the spike_obj attribute with a SessionSpikes object.

        Args:
            spike_times_path (str): The file path to the spike times.
            spike_cluster_path (str): The file path to the spike clusters.
            start_time (int): The start time of the spikes.
            parent_dir (str): The parent directory of the spike files.
        """
        self.spike_obj = SessionSpikes(spike_times_path, spike_cluster_path, start_time, parent_dir,**kwargs)

    def init_sound_event_dict(self, sound_write_path, **format_kwargs):
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
        patterns = format_kwargs.get('patterns', None)
        normal_patterns = format_kwargs.get('normal_patterns', None)
        if normal_patterns is None:
            raise ValueError('No normal patterns provided')
        if patterns is None:
            raise ValueError('No patterns provided')

        patterns_df = pd.DataFrame(patterns, columns=list('ABCD'))
        ABCD_patts = [pattern for pattern in patterns if np.all(np.diff(pattern)>0)]
        non_ABCD_patts = [pattern for pattern in patterns if not np.all(np.diff(pattern)>0)]
        patt_rules = [0 if patt in ABCD_patts else 1 for patt in patterns]
        patt_group = [np.where(patterns_df['A']==patt[0])[0][0] for patt in patterns]
        ptypes = {'_'.join([str(e) for e in patt]): int(f'{group}{rule}')
                  for patt, group, rule in zip(patterns, patt_group, patt_rules)}

        if not sound_write_path:
            assert self.sound_writes_df is not None
            sound_writes = self.sound_writes_df
        else:
            sound_writes = load_sound_bin(sound_write_path)
            sound_writes = format_sound_writes(sound_writes, ptypes=ptypes, **format_kwargs if format_kwargs else {})
            self.sound_writes_df = sound_writes

        base_pip_idx = [e for e in sound_writes['Payload'].unique() if e not in sum(patterns, []) and e >= 8]
        if len(base_pip_idx) > 1:
            if 3 not in sound_writes['Payload'].values:
                base_pip_idx = min(base_pip_idx)
            else:
                base_pip_idx = sound_writes.query('Payload in @base_pip_idx')['Payload'].mode().iloc[0]
        else:
            base_pip_idx = base_pip_idx[0]
        non_base_pip_idx = sorted([e for e in sound_writes['Payload'].unique() if e not in [base_pip_idx,3]])

        event_sound_times = {}
        event_sound_idx = {}
        if 3 in sound_writes['Payload'].values:
            for idx, p in patterns_df.iterrows():
                for pi, pip in enumerate(p):
                    ptype= ptypes['_'.join([str(e) for e in p])]
                    timesss = sound_writes.query(f'Payload == {pip} and pip_counter == {pi + 1} '
                                                 f'and ptype == @ptype')['Timestamp']
                    if not timesss.empty:
                        event_sound_times[f'{p.index[pi]}-{idx}'] = timesss
                        event_sound_idx[f'{p.index[pi]}-{idx}'] = pip
        else:
            # for lbl, pip in zip(patterns_df.columns, patterns_df.loc[0]):
            for pip_i, pip in enumerate(non_base_pip_idx):
                lbl = chr(ord('A') + pip_i)
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
        event_free_time_bins = [
            get_times_in_window(event_times, [t - event_free_time_window, t + event_free_time_window]).size == 0
            for t in np.arange(self.spike_obj.start_time, self.spike_obj.start_time
                                                   + self.spike_obj.duration-time_bin, time_bin)]
        event_free_time_bins = np.array(event_free_time_bins)
        print('old means')
        print(self.spike_obj.unit_means)
        self.spike_obj.unit_means = self.spike_obj.get_unit_mean_std(event_free_time_bins)
        print('new means')
        print(self.spike_obj.unit_means)

    def get_grouped_rates_by_property(self, pip_desc, pip_prop, group_noise):
        groups = np.unique([e[pip_prop] for e in pip_desc.values()])
        print(groups)
        assert group_noise
        for group in groups:
            group_pips = [k for k, v in pip_desc.items() if v[pip_prop] == group]
            unit_rates = gen_patterned_unit_rates(len(self.spike_obj.cluster_spike_times_dict), len(group_pips), group_noise)
            unit_times_offsets = gen_patterned_time_offsets(len(self.spike_obj.cluster_spike_times_dict), len(group_pips),0.1)
            for k, v,t in zip(group_pips, unit_rates, unit_times_offsets):
                self.sound_event_dict[k].synth_params = {'unit_rates': v, 'unit_time_offsets': t}

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
            plot_funcs.plot_decoder_accuracy(self.decoders[decoder_name].accuracy, labels=labels, )

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
        sessname = self.sessname
        name, date = sessname.split('_')
        if not date.isnumeric():
            date = date[:-1]

        if 'Session_Block' not in self.td_df.columns:
            if 'WarmUp' not in self.td_df.columns:
                self.td_df['WarmUp'] = np.full_like(self.td_df.index, False)
            first_dev_trial = self.td_df.query('Pattern_Type != 0').index[0] if len(self.td_df.query('Pattern_Type != 0'))>0 else self.td_df.shape[0]
            if 'Stage' not in self.td_df.columns:
                if len(self.td_df['Pattern_Type'].unique())>1:
                    self.td_df['Stage'] = 3
                else:
                    self.td_df['Stage'] = 4
            sess_block = [-1 if r['WarmUp'] else 0 if r['Stage'] <= 3 else 2 if r['Stage']==4 and idx<first_dev_trial
                          else 3 if r['Stage']==4 and idx>=first_dev_trial else 0 for idx, r in self.td_df.iterrows()]
            self.td_df['Session_Block'] = sess_block

        self.td_df.index = pd.MultiIndex.from_arrays(
            [[sessname] * len(self.td_df), [name] * len(self.td_df), [date] * len(self.td_df),
             self.td_df.reset_index().index+1],
            names=['sess', 'name', 'date', 'trial_num'])


    def get_n_since_last(self):
        self.td_df['n_since_last'] = np.arange(self.td_df.shape[0])
        since_last = self.td_df[self.td_df['Tone_Position'] == 0].index
        if not since_last.empty:
            for t, tt in zip(since_last, np.pad(since_last, [1, 0])):
                self.td_df.loc[tt + 1:t, 'n_since_last'] = self.td_df.loc[tt + 1:t, 'n_since_last'] - tt
            self.td_df.loc[t + 1:, 'n_since_last'] = self.td_df.loc[t + 1:, 'n_since_last'] - t

    def get_local_rate(self, window=10):
        self.td_df['local_rate'] = self.td_df['Tone_Position'].rolling(window=window).mean()

    def init_lick_obj(self, lick_times_path, sound_events_path, normal):
        licks = read_lick_times(lick_times_path)
        sound_events = pd.read_csv(sound_events_path)
        sound_events = format_sound_writes(sound_events, normal)
        self.lick_obj = SessionLicks(licks, sound_events)

    def get_licks_to_event(self, event_idx, event_name, window=(-3, 3), align_kwargs=None,plot=False):
        self.lick_obj.get_event_spikes(event_idx, event_name, window, self.sessname,
                                       **align_kwargs if align_kwargs else {})
        if plot:
            self.lick_obj.plot_licks(event_name, window)

    def init_pupil_obj(self, pupil_data, sound_events_path, beh_events_path, normal):
        if sound_events_path is None:
            sound_events = None
        else:
            sound_events = pd.read_csv(sound_events_path)
            sound_events = format_sound_writes(sound_events, normal)

        if beh_events_path is None:
            beh_events = None
        else:
            beh_events = pd.read_csv(beh_events_path)
        self.pupil_obj = SessionPupil(pupil_data, sound_events, beh_events)

    def get_pupil_to_event(self, event_idx, event_name, window=(-3, 3), align_kwargs=None,alignmethod='w_soundcard',
                           plot_kwargs=None):
        if alignmethod == 'w_soundcard':
            self.pupil_obj.align2events(event_idx, event_name, window, self.sessname,
                                        **align_kwargs if align_kwargs else {})
        elif alignmethod == 'w_td_df':
            if align_kwargs and 'sound_df_query' in align_kwargs:
                align_kwargs.pop('sound_df_query')
            if event_name == 'X':
                col2use = 'Gap_Time_dt'
                td_query = 'Trial_Outcome in [0,1]'
            elif event_name == 'A':
                col2use = 'ToneTime_dt'
                td_query = 'Tone_Position == 0 & N_TonesPlayed > 0'
            else:
                return
            try:
                self.pupil_obj.align2events_w_td_df(self.td_df,col2use, td_query,
                                                    window, self.sessname,event_idx, event_name,
                                                    **align_kwargs if align_kwargs else {})
            except KeyError:
                print(f'Could not align to event {event_name} in session {self.sessname}')
        else:
            raise NotImplementedError


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


def get_predictor_from_psth(sess_obj: Session, event_key, psth_window, new_window, mean=np.mean, mean_axis=2,
                            use_iti_zscore=False, use_unit_zscore=True, baseline=0) -> np.ndarray:
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
    if baseline:
        baseline_window_idx = np.logical_and(event_arr_tseries >= -baseline, event_arr_tseries <=0)
        event_arr = event_arr - np.mean(event_arr[:, :, baseline_window_idx], axis=2, keepdims=True)
    predictor = event_arr[:, :, time_window_idx]
    if use_iti_zscore:
        predictor = zscore_by_unit(predictor, sess_obj.iti_zscore[0], sess_obj.iti_zscore[1])
    if use_unit_zscore:
        predictor = zscore_by_unit(predictor, sess_obj.spike_obj.unit_means[0], sess_obj.spike_obj.unit_means[1])
    if mean:
        predictor = mean(predictor, axis=mean_axis)

    return predictor
