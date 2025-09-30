from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from plot_funcs import plot_2d_array_with_subplots


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
    def __init__(self,resp_dict:dict, stim_names: list, resp_window: [float,float]):
        self.participation_rate_by_sess_by_pip = None
        self.participation_rate_by_pip_arr = None
        self.participation_rate_by_pip = None

        self.participation_rate_plots = {}

        self.sessnames = list(resp_dict.keys())
        self.responses = resp_dict
        self.stim_names: list = stim_names

        assert isinstance(resp_window, list)
        assert len(resp_window) == 2 and all([isinstance(t, float) for t in resp_window])
        self.resp_window = resp_window

    def get_participation_rate(self, activity_window_s:[float, float], active_threshold:float,
                               **kwargs):
        assert isinstance(activity_window_s, list)
        assert len(activity_window_s) == 2 and all([isinstance(t, float) for t in activity_window_s])

        assert isinstance(active_threshold,(float|int))


        participation_rate_by_pip = {pip: np.hstack([get_participation_rate(self.responses[sess][pip],
                                                                            self.resp_window, activity_window_s,
                                                                            active_threshold,
                                                                            max_func=kwargs.get('max_func',np.max))
                                                     for sess in self.responses])
                                     for pip in self.stim_names}

        self.participation_rate_by_pip = participation_rate_by_pip
        self.participation_rate_by_pip_arr = np.array(list(self.participation_rate_by_pip.values()))

        participation_rate_by_sess_by_pip = {sess: {pip: get_participation_rate(self.responses[sess][pip],
                                                                                self.resp_window, activity_window_s,
                                                                                active_threshold,
                                                                                max_func=kwargs.get('max_func',np.max))
                                                     for pip in self.stim_names}
                                               for sess in self.responses}
        self.participation_rate_by_sess_by_pip = participation_rate_by_sess_by_pip

    def filter_by_prc_rate(self, **kwargs):
        prc_thresh = kwargs.get('prc_threshold', 0.5)

        prc_pips = kwargs.get('prc_pips', self.stim_names)
        if prc_pips is None:
            prc_pips = self.stim_names
        assert len(prc_pips) == len(self.stim_names)

        if prc_thresh is None:
            prc_thresh = 0.5

        units_to_keep = {sess: {pip: self.participation_rate_by_sess_by_pip[sess][prc_pip] > prc_thresh
                              for pip,prc_pip in zip(self.stim_names,prc_pips)}
                        for sess in self.responses.keys()}
        if kwargs.get('prc_mutual'):
            units_to_keep = {sess: {pip:np.all(list(units_to_keep[sess].values()),axis=0)
                                    for pip in units_to_keep[sess].keys()}
                             for sess in units_to_keep.keys()}
        subset_resp_dict = {sess: {pip:self.responses[sess][pip][:,units_to_keep[sess][pip]]
                                    for pip in units_to_keep[sess].keys()}
                             for sess in units_to_keep.keys()}
        return subset_resp_dict

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
