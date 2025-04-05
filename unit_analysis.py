from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ephys_analysis_funcs import plot_2d_array_with_subplots


def get_participation_rate(resp_mat: np.ndarray, resp_window: tuple | list, activity_window_S: list | tqdm,
                           active_thresh: float, **kwargs):
    # get response x series and window in idxs
    x_ser = np.round(np.linspace(*resp_window, resp_mat.shape[-1]),2)
    x_ser = np.round(x_ser, 2)
    window_idxs = [np.where(x_ser == t)[0][0] for t in activity_window_S]
    # get active units
    max_func = partial(kwargs.get('max_func', np.max), axis=-1)
    max_activity = max_func(resp_mat[:,window_idxs[0]:window_idxs[1]] if resp_mat.ndim == 2 else
                            resp_mat[:, :, window_idxs[0]:window_idxs[1]])
    participation_rate_arr = max_activity > active_thresh
    if resp_mat.ndim == 3:
        participation_rate_arr = participation_rate_arr.mean(axis=0)

    return participation_rate_arr


class UnitAnalysis:
    def __init__(self,resp_dict:dict, stim_names: [str,], resp_window: [float,float]):
        self.participation_rate_by_pip_arr = None
        self.participation_rate_by_pip = None

        self.participation_rate_plots = {}

        self.sessnames = list(resp_dict.keys())
        self.responses = resp_dict
        self.stim_names = stim_names

        assert isinstance(resp_window, list)
        assert len(resp_window) == 2 and all([isinstance(t, int) for t in resp_window])
        self.resp_window = resp_window

    def get_participation_rate(self, activity_window_S:[float,float], active_threshold:float,
                         sess_filt_kwargs=None, **kwargs):
        assert isinstance(activity_window_S,list)
        assert len(activity_window_S) == 2 and all([isinstance(t,float) for t in activity_window_S])

        assert isinstance(active_threshold,float)

        if sess_filt_kwargs is None:
            sess_filt_kwargs = {}
        animals = sess_filt_kwargs.get('animals')
        active_units_by_pip = {pip: np.hstack([get_participation_rate(self.responses[sess][pip],
                                                                      self.resp_window, activity_window_S, active_threshold,
                                                                      max_func=kwargs.get('max_func',np.max))
                                               for sess in self.responses
                                               if (any(e in sess for e in animals) if animals else True)])
                               for pip in self.stim_names}

        self.participation_rate_by_pip = active_units_by_pip
        self.participation_rate_by_pip_arr = np.array(list(self.participation_rate_by_pip.values()))

    def plot_participation_rate(self, plot_name:str, plot_kwargs=None, **r_maps_kwargs):
        if plot_kwargs is None:
            plot_kwargs = {}

        plot = self.participation_rate_plots[plot_name] = plt.subplots()
        participation_rate_arr = self.participation_rate_by_pip_arr
        plot_recursive = r_maps_kwargs.get('recursive',False)
        if r_maps_kwargs.get('sort_by_max', True):
            participation_rate_sorted_arr_list = [participation_rate_arr.T[np.argsort(participation_rate_arr.T[:, pip_i])[::-1]]
                                                   for pip_i in range(participation_rate_arr.shape[0] if plot_recursive else 1)]
            participation_rate_arr_sorted = np.vstack(participation_rate_sorted_arr_list)
        else:
            participation_rate_arr_sorted = participation_rate_arr

        plot_2d_array_with_subplots(np.vstack(participation_rate_arr_sorted),
                                    interpolation='none', plot=plot, **plot_kwargs)
        plot[1].set_xticks(np.arange(participation_rate_arr_sorted.shape[1]))
        plot[1].set_xticklabels([e.split('-')[0] for e in self.stim_names])
        [plot[1].axvline(i + 0.5, c='k', lw=0.5) for i in range(participation_rate_arr_sorted.shape[1])]
        
        plot[1].set_ytick_labels([])
        if plot_recursive:
            [plot[1].axhline(i * participation_rate_arr_sorted.shape[0], c='k', lw=0.5) for i in
             range(len(self.stim_names))]

    def get_cross_event_active_units(self):
        cross_cond_units = np.all(
            [self.participation_rate_by_pip_arr.T[:, stim_i] > 0.5 for stim_i, _ in enumerate(self.stim_names)],
            axis=0)
