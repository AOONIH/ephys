# matplotlib.use('TkAgg')
import numpy as np
from matplotlib import pyplot as plt, lines as mlines
from scipy.linalg import orthogonal_procrustes
from sklearn.cross_decomposition import CCA

from ephys_analysis_funcs import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import euclidean
import argparse
import yaml
import platform
from elephant.gpfa import GPFA
import neo
from sklearn.model_selection import cross_val_score
from os import cpu_count
from itertools import combinations

from behviour_analysis_funcs import get_sess_name_date_idx
from io_utils import posix_from_win
from plot_funcs import plot_ts_var, format_axis
from neural_similarity_funcs import compare_pip_sims_2way
from sess_dataclasses import Session, get_predictor_from_psth


class PopPCA:

    def __init__(self, responses_by_cond: dict):
        self.scatter_plot = None
        self.proj_2d_plot = None
        self.proj_3d_plot = None
        self.eig_vals = None
        self.projected_pca_ts_by_cond = None
        self.pca_ts_plot = None
        self.Xa_trial_averaged_pca = None
        assert isinstance(list(responses_by_cond.values())[0], dict)
        self.responses_by_cond = responses_by_cond
        self.conds = list(responses_by_cond.keys())
        self.events = list(responses_by_cond[self.conds[0]].keys())
        self.event_concatenated_responses = self.get_event_concatenated_responses()
        self.get_eig_vals()

    def get_event_concatenated_responses(self):
        event_concatenated_responses = np.hstack(
            [np.hstack(
                [e_responses for e_name, e_responses in cond_responses.items()])
                for cond_responses in self.responses_by_cond.values()])
        event_concatenated_responses = np.squeeze(event_concatenated_responses)
        event_concatenated_responses = event_concatenated_responses - np.nanmean(event_concatenated_responses, axis=1,
                                                                                 keepdims=True)
        return event_concatenated_responses

    def get_eig_vals(self):
        self.eig_vals = compute_eig_vals(self.event_concatenated_responses, plot_flag=True)
        self.eig_vals[2][1].set_xlim(0, 30)
        self.eig_vals[2][1].set_ylabel('PC component')
        self.eig_vals[2][1].set_xlabel('Proportion of variance explained')
        # self.eig_vals[2][0].show()

    def get_trial_averaged_pca(self, n_components=15, standardise=True):
        self.Xa_trial_averaged_pca = compute_trial_averaged_pca(self.event_concatenated_responses,
                                                                n_components=n_components, standardise=standardise)

    def get_projected_pca_ts(self, standardise=True):
        self.projected_pca_ts_by_cond = {cond: {event_name: project_pca(event_response, self.Xa_trial_averaged_pca,
                                                                        standardise=standardise)
                                                for event_name, event_response in cond_responses.items()}
                                         for cond, cond_responses in self.responses_by_cond.items()}

    def plot_pca_ts(self, event_window, n_comp_toplot=5, plot_separately=False, fig_kwargs=None,
                    conds2plot=None, **kwargs):
        if conds2plot is None:
            conds2plot = self.conds
        if kwargs.get('events2plot', None) is None:
            events2plot = {cond: list(self.projected_pca_ts_by_cond[cond].keys()) for cond in conds2plot}
        else:
            events2plot = kwargs.get('events2plot')
        if kwargs.get('plot', None) is None:
            self.pca_ts_plot = plt.subplots(len(self.events) if plot_separately else 1, n_comp_toplot, squeeze=False,
                                            **(fig_kwargs if fig_kwargs is not None else {}))
        else:
            self.pca_ts_plot = kwargs.get('plot')

        axes = self.pca_ts_plot[1] if plot_separately else [self.pca_ts_plot[1][0]] * len(self.events)
        lss = kwargs.get('lss', ['-', '--', ':', '-.'])
        plt_cols = kwargs.get('plt_cols', ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'])
        [[plot_pca_ts([projected_responses], [f'{cond} {event}'], event_window, n_components=n_comp_toplot,
                      plot=[self.pca_ts_plot[0], axes[ei]], plot_kwargs={'ls': lss[cond_i],  # 'c': plt_cols[ei],
                                                                         'label': f'{cond} {event}'})
          for ei, (event, projected_responses) in
          enumerate(zip(events2plot[cond], [self.projected_pca_ts_by_cond[cond][e] for e in events2plot[cond]]))]
         for cond_i, cond in enumerate(conds2plot)]
        [row_axes[0].set_ylabel('PC component') for row_axes in self.pca_ts_plot[1]]
        [row_axes[0].legend(loc='upper center', ncol=4) for row_axes in self.pca_ts_plot[1].T]
        # [ax.legend() for ax in self.pca_ts_plot[1]]
        [ax.set_xlabel('Time from stimulus onset (s)') for ax in self.pca_ts_plot[1][-1]]
        self.pca_ts_plot[0].show()

    def scatter_pca_points(self, prop: str, t_s: list, x_ser: np.ndarray, **kwargs):
        pca_comps_2plot = kwargs.get('pca_comps_2plot', [0, 1])
        if kwargs.get('plot', None) is None:
            fig, ax = plt.subplots(**kwargs.get('fig_kwargs', {}))
        else:
            fig, ax = kwargs.get('plot')

        t_idxs = [np.where(x_ser == t)[0][0] for t in t_s]
        markers = kwargs.get('markers', list(mlines.Line2D.markers.keys())[:len(t_idxs)])
        proj_pcas_t = {e: {t: [projs[pca_comp][t_idx] for pca_comp in pca_comps_2plot]
                           for t_idx, t in zip(t_idxs, t_s)}
                       for e, projs in self.projected_pca_ts_by_cond[prop].items()}
        for pi, pip in enumerate(proj_pcas_t):
            for ti, t in enumerate(t_s):
                ax.scatter(*proj_pcas_t[pip][t], marker=markers[ti], c=f'C{pi}', label=pip, s=50)

        ax.legend()
        format_axis(ax)
        ax.set_xlabel(f'PC {pca_comps_2plot[0]}')
        ax.set_ylabel(f'PC {pca_comps_2plot[1]}')

        fig.set_layout_engine('tight')
        self.scatter_plot = fig, ax

    def plot_3d_pca_ts(self, prop, event_window, **kwargs):
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.ndimage import gaussian_filter1d

        pca_comps_2plot = kwargs.get('pca_comps_2plot', [0, 1, 2])
        minimal_axes = kwargs.get('minimal_axes', True)
        show_triad = kwargs.get('show_triad', True)  # << new
        triad_len_frac = kwargs.get('triad_len_frac', 0.12)  # << new (fraction of axis span)
        triad_off_frac = kwargs.get('triad_off_frac', 0.04)  # << new (margin from min corner)

        if kwargs.get('plot', None) is None:
            fig, axes = plt.subplots(
                ncols=kwargs.get('n_cols', 1),
                subplot_kw={"projection": "3d"},
                figsize=(10, 10)
            )
        else:
            fig, axes = kwargs.get('plot')

        if isinstance(axes, plt.Axes):
            axes = [axes]

        # --- data prep ---
        smoothing = kwargs.get('smoothing', 3)
        proj_ts = {
            e: {pc: self.projected_pca_ts_by_cond[prop][e][pc] for pc in pca_comps_2plot}
            for e in self.projected_pca_ts_by_cond[prop].keys()
        }
        proj_ts = {
            e: {pc: gaussian_filter1d(proj_ts[e][pc], smoothing) for pc in pca_comps_2plot}
            for e in self.projected_pca_ts_by_cond[prop].keys()
        }
        proj_ts_arrs = {e: np.vstack(list(pip_projs.values())) for e, pip_projs in proj_ts.items()}

        t0_time_s = kwargs.get('t0_time', 0)
        t_end_s = kwargs.get('t_end', event_window[1])
        x_ser = kwargs.get('x_ser', None)
        if x_ser is None:
            x_ser = np.round(
                np.linspace(event_window[0], event_window[1], list(proj_ts.values())[0][pca_comps_2plot[0]].shape[-1]),
                2
            )
        t0_time_idx = np.where(x_ser == t0_time_s)[0][0]
        t_end_idx = np.where(x_ser == t_end_s)[0][0]

        if kwargs.get('align_trajs'):
            align_window = kwargs.get('align_window', [t0_time_s, t_end_s])
            align_window_mask = np.array([t >= align_window[0] or t <= align_window[1] for t in x_ser])
            ref_proj = list(proj_ts_arrs.values())[0]
            proj_ts_arrs = {
                e: (cca_align_pca_timeseries(ref_proj, projs, align_window_mask)[1]) if ei >= 0 else projs
                for ei, (e, projs) in enumerate(list(proj_ts_arrs.items()))
            }

        # --- plotting ---
        for ei, (e, proj_pcas) in enumerate(proj_ts_arrs.items()):
            t0_points = [proj_pcas[pc][t0_time_idx] for pc, _ in enumerate(pca_comps_2plot)]
            t_end_points = [proj_pcas[pc][t_end_idx] for pc, _ in enumerate(pca_comps_2plot)]
            for ax in axes:
                idxs_in_event = np.logical_and(x_ser >= t0_time_s, x_ser <= t_end_s)
                idxs_out_event = np.logical_or(x_ser < t0_time_s, x_ser > t_end_s)
                for ii, (in_event, in_event_ls) in enumerate(zip([idxs_in_event, idxs_out_event],
                                                                 kwargs.get('in_event_ls', ['-', '--']))):
                    if not kwargs.get('plot_out_event', True) and ii == 1:
                        continue
                    masked_ts = [proj_pcas[pc].copy() for pc, _ in enumerate(pca_comps_2plot)]
                    for pc, _ in enumerate(masked_ts): masked_ts[pc][~in_event] = np.nan
                    ax.plot(*masked_ts, c=f'C{ei}', label=e if ii == 0 else None, ls=in_event_ls)

                scatter_kwargs = kwargs.get('scatter_kwargs', {})
                markers = scatter_kwargs.get('markers', ['v', 's'])
                size = scatter_kwargs.get('size', 20)
                ax.scatter(*t0_points, c=f'C{ei}', marker=markers[0], s=size)
                ax.scatter(*t_end_points, c=f'C{ei}', marker=markers[1], s=size)

                if kwargs.get('scatter_times'):
                    t_pnts = kwargs.get('scatter_times')
                    if not isinstance(t_pnts, list): t_pnts = [t_pnts]
                    for t_pnt in t_pnts:
                        t_pnt_idx = np.where(x_ser == t_pnt)[0][0]
                        t_pnts_pca = [proj_pcas[pc][t_pnt_idx] for pc in range(len(proj_pcas))]
                        ax.scatter(*t_pnts_pca, **scatter_kwargs)

                # keep axis names for tooltips, but we'll draw a triad instead
                ax.set_xlabel(f'PC{pca_comps_2plot[0]}')
                ax.set_ylabel(f'PC{pca_comps_2plot[1]}')
                ax.set_zlabel(f'PC{pca_comps_2plot[2]}')

        # --- view / legend ---
        axes[0].view_init(elev=22, azim=30)
        axes[0].legend()

        # --- styling ---
        fig.patch.set_facecolor('white')
        for ax in axes:
            ax.set_facecolor('white')
            ax.grid(False)

            # Remove panes/bounding box/axis lines (3D "spines")
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                try:
                    axis.pane.fill = False
                    axis.pane.set_edgecolor('white')
                    axis.line.set_color((1, 1, 1, 0))
                except Exception:
                    pass
            for attr in ('w_xaxis', 'w_yaxis', 'w_zaxis'):
                if hasattr(ax, attr):
                    getattr(ax, attr).line.set_visible(False)

            # --- ticks: remove X ticks only ---
            ax.set_xticks([])
            ax.set_xticklabels([])
            if minimal_axes:
                ax.set_yticks([])
                ax.set_zticks([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])

            # --- rotating axis triad in lower-left data corner ---
            if show_triad:
                xl, yl, zl = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
                dx, dy, dz = (xl[1] - xl[0]), (yl[1] - yl[0]), (zl[1] - zl[0])
                # offset from min corner
                x0 = xl[0] + triad_off_frac * dx
                y0 = yl[0] + triad_off_frac * dy
                z0 = zl[0] + triad_off_frac * dz
                Lx, Ly, Lz = triad_len_frac * dx, triad_len_frac * dy, triad_len_frac * dz

                # arrows (these rotate with the 3D view)
                ax.quiver(x0, y0, z0, Lx, 0, 0, arrow_length_ratio=0.2, color='k')
                ax.quiver(x0, y0, z0, 0, Ly, 0, arrow_length_ratio=0.2, color='k')
                ax.quiver(x0, y0, z0, 0, 0, Lz, arrow_length_ratio=0.2, color='k')

                # labels at arrow tips
                ax.text(x0 + Lx, y0, z0, 'PCA 1', ha='left', va='center')
                ax.text(x0, y0 + Ly, z0, 'PCA 2', ha='left', va='center')
                ax.text(x0, y0, z0 + Lz, 'PCA 3', ha='left', va='bottom')

        self.proj_3d_plot = fig,axes
        fig.tight_layout()
        fig.show()
        return fig, axes

    def plot_2d_pca_ts(self, prop, event_window, **kwargs):
        pca_comps_2plot = kwargs.get('pca_comps_2plot', [0, 1])
        fig, ax = plt.subplots(figsize=(4, 3))

        # Get projections
        smoothing = kwargs.get('smoothing', 3)
        proj_ts = {e: {pca_comp: self.projected_pca_ts_by_cond[prop][e][pca_comp] for pca_comp in pca_comps_2plot}
                   for e in self.projected_pca_ts_by_cond[prop].keys()}

        # Smooth projections
        proj_ts = {e: {pca_comp: gaussian_filter1d(proj_ts[e][pca_comp], smoothing) for pca_comp in pca_comps_2plot}
                   for e in self.projected_pca_ts_by_cond[prop].keys()}

        # Get time series and initial points
        t0_time_s = kwargs.get('t0_time', 0)
        t_end_s = kwargs.get('t_end', event_window[1])
        x_ser = kwargs.get('x_ser', None)
        if x_ser is None:
            x_ser = np.round(np.linspace(event_window[0], event_window[1], list(proj_ts.values())[0].shape[-1]), 2)

        t0_time_idx = np.where(x_ser == t0_time_s)[0][0]
        t_end_idx = np.where(x_ser == t_end_s)[0][0]

        proj_ts_arrs = {e: np.vstack(list(pip_projs.values())) for e, pip_projs in proj_ts.items()}

        if kwargs.get('align_trajs'):
            align_window = kwargs.get('align_window', [t0_time_s, t_end_s])
            align_window_mask = np.array([t >= align_window[0] or t <= align_window[1] for t in x_ser])
            ref_proj = list(proj_ts_arrs.values())[0]
            proj_ts_arrs = {
                e: (cca_align_pca_timeseries(ref_proj, projs, align_window_mask)[1]) if ei >= 0 else projs
                for ei, (e, projs) in enumerate(list(proj_ts_arrs.items()))
            }

        for ei, (e, proj_pcas) in enumerate(proj_ts_arrs.items()):
            t0_points = [proj_pcas[pca_comp][t0_time_idx] for pca_comp, _ in enumerate(pca_comps_2plot)]
            t_end_points = [proj_pcas[pca_comp][t_end_idx] for pca_comp, _ in enumerate(pca_comps_2plot)]

            idxs_in_event = np.logical_and(x_ser >= t0_time_s, x_ser <= t_end_s)
            idxs_out_event = np.logical_or(x_ser < t0_time_s, x_ser > t_end_s)

            for ii, (in_event, in_event_ls) in enumerate(zip([idxs_in_event, idxs_out_event],
                                                             kwargs.get('in_event_ls', ['-', '--']))):
                if not kwargs.get('plot_out_event', True) and ii == 1:
                    continue

                masked_ts = [proj_pcas[pca_comp].copy() for pca_comp, _ in enumerate(pca_comps_2plot)]
                for pca_comp, _ in enumerate(masked_ts):
                    masked_ts[pca_comp][np.invert(in_event)] = np.nan

                ax.plot(*masked_ts, c=f'C{ei}', label=e if ii == 0 else None, ls=in_event_ls)

            scatter_kwargs = kwargs.get('scatter_kwargs', {})
            markers = scatter_kwargs.get('markers', ['v', 's'])
            size = scatter_kwargs.get('size', 20)
            ax.scatter(*t0_points, c=f'C{ei}', marker=markers[0], s=size)
            ax.scatter(*t_end_points, c=f'C{ei}', marker=markers[1], s=size)

            if kwargs.get('scatter_times'):
                t_pnts = kwargs.get('scatter_times')

                if not isinstance(t_pnts, list):
                    t_pnts = [t_pnts]
                for t_pnt in t_pnts:
                    t_pnt_idx = np.where(x_ser == t_pnt)[0][0]
                    t_pnts_pca = [pca_comp[t_pnt_idx] for pca_comp in proj_pcas.values()]
                    ax.scatter(*t_pnts_pca, **scatter_kwargs)

        ax.set_xlabel(f'PC{pca_comps_2plot[0]}')
        ax.set_ylabel(f'PC{pca_comps_2plot[1]}')
        ax.legend()
        fig.show()

        self.proj_2d_plot = fig,ax


def get_event_response(event_response_dict, event):
    event_responses = np.array_split(event_response_dict[event], event_response_dict[event].shape[0], axis=0)

    event_responses = [np.squeeze(e) for e in event_responses]

    return event_responses


def format_tractories(trajectories, pc_idxs, subset=None, mean_axis=None):
    trajectories = np.transpose(np.array(trajectories), (1, 0, 2))
    by_pc_traj = [trajectories[idx] for idx in pc_idxs]
    if subset is not None:
        by_pc_traj = [traj[subset] for traj in by_pc_traj]

    if mean_axis is not None:
        by_pc_traj = [traj.mean(axis=mean_axis) for traj in by_pc_traj]

    return by_pc_traj


def compute_eig_vals(X,plot_flag=False):
    c = np.cov(X, rowvar=True)  # covariance matrix
    eig_vals, eig_vecs = np.linalg.eig(c)
    srt = np.argsort(eig_vals)[::-1]
    print(srt)
    eig_vals = eig_vals[srt]
    eig_vecs = eig_vecs[:, srt]
    fig, ax = plt.subplots()
    if plot_flag:
        ax.plot(np.cumsum(eig_vals / eig_vals.sum()), label='cumulative % variance explained')
        ax.plot(eig_vals / eig_vals.sum(), label='% variance explained')
        ax.set_ylim([0, 1])
        n_comp_to_thresh = np.argwhere(np.cumsum(eig_vals / eig_vals.sum()) > 0.9)[0][0]
        ax.plot([n_comp_to_thresh] * 2, [0, 0.9], color='k', ls='--', )
        ax.plot([0, n_comp_to_thresh], [0.9, 0.9], color='k', ls='--', )
        ax.legend()
        # fig.show()

    return eig_vals, eig_vecs,(fig,ax)


def compute_trial_averaged_pca(X_trial_averaged,n_components=15,standardise=False):
    # Xa = z_score(X_trial_averaged)
    if standardise:
        X_std = StandardScaler().fit_transform(X_trial_averaged)
    else:
        X_std = X_trial_averaged
    pca = PCA(n_components=n_components)
    pca.fit(X_std.T)

    return pca


def project_pca(X_trial,pca,standardise=False):
    # ss = StandardScaler(with_mean=True, with_std=True)
    if standardise:
        trial_sc = StandardScaler().fit_transform(X_trial)
    else:
        trial_sc = X_trial
    proj_trial = pca.transform(trial_sc.T).T
    return proj_trial


def plot_pca_ts(X_proj_by_event, events, window, plot=None, n_components=3,plot_kwargs=None):
    if not plot:
        fig, axes = plt.subplots(1, n_components, figsize=[20, 4],)
    else:
        fig,axes = plot

    x_ser = np.linspace(window[0], window[1], X_proj_by_event[0][0].shape[-1])
    # smooth
    if plot_kwargs.get('smoothing', None):
        smoothing = plot_kwargs.get('smoothing')
        for ei, event in enumerate(events):
            for comp in range(n_components):
                X_proj_by_event[ei][comp] = gaussian_filter1d(X_proj_by_event[ei][comp], smoothing)

    for comp in range(n_components):
        ax = axes[comp]
        for ei, event in enumerate(events):
            projected_trials = np.array(X_proj_by_event[ei])
            if projected_trials.ndim == 3:
                projected_trials_comp = projected_trials[:, comp, :]
                pc_mean_ts =  projected_trials.mean(axis=0)
            else:
                projected_trials_comp = projected_trials[comp]
                pc_mean_ts = projected_trials_comp
            kwargs2use = plot_kwargs if plot_kwargs is not None else {}
            if plot_kwargs.get('ls',{}):
                if isinstance(plot_kwargs.get('ls'),list):
                    kwargs2use['ls']=plot_kwargs.get('ls',{})[ei]
                else:
                    kwargs2use['ls']=plot_kwargs.get('ls',{})
            if plot_kwargs.get('c'):
                if isinstance(plot_kwargs.get('c'),list):
                    kwargs2use['c']=plot_kwargs.get('c')[ei]
                else:
                    kwargs2use['c']=plot_kwargs.get('c')

            ax.plot(x_ser,pc_mean_ts,**kwargs2use)
            if projected_trials.ndim == 3:
                plot_ts_var(x_ser,projected_trials_comp,kwargs2use.get('c',f'C{ei}'),ax)

        ax.set_ylabel(f'PC {comp+1}')
        ax.set_xlabel('Time (s)')
        ax.axvline(0, color='k', ls='--')
        # ax.legend(ncol=len(events))

    # axes[-1].legend(ncol=len(events))
    return fig, axes



def cca_align_pca_timeseries(
    X_ref, X_tgt, time_mask, n_components=None, standardise=True, fix_signs=True
):
    """
    Align two PCA×time trajectories via CCA trained on a subset of timepoints,
    then apply the learned transforms to the full trajectories.

    Parameters
    ----------
    X_ref : (n_pcs, n_times) array
        Reference subject PCA time series.
    X_tgt : (n_pcs, n_times) array
        Target subject PCA time series to align to the reference.
    time_mask : (n_times,) boolean array
        True for the timepoints used to FIT CCA (alignment window).
    n_components : int or None
        Number of canonical components to extract. If None, uses
        min(n_pcs_ref, n_pcs_tgt).
    standardise : bool
        If True, z-score features using stats computed on the subset; then
        apply the same scaling to the full series.
    fix_signs : bool
        If True, flips canonical components so correlations on the subset are positive.

    Returns
    -------
    X_ref_c_full : (n_components, n_times) array
        Reference canonical time series for the full time axis.
    X_tgt_c_full : (n_components, n_times) array
        Target canonical time series, aligned to reference space, full time axis.
    cca : fitted sklearn.cross_decomposition.CCA
        The fitted CCA model (you can reuse it to transform other events).
    meta : dict
        Useful info: scalers, applied sign flips, feature indices, etc.
    """
    # --- Validate & harmonize feature dims ---
    n_pcs_ref, n_times_ref = X_ref.shape
    n_pcs_tgt, n_times_tgt = X_tgt.shape
    if n_times_ref != n_times_tgt:
        raise ValueError(f"Time dimension mismatch: {n_times_ref} vs {n_times_tgt}")
    if time_mask.dtype != bool or time_mask.shape[0] != n_times_ref:
        raise ValueError("time_mask must be boolean of length n_times")

    # Use common feature count if they differ
    n_feat = min(n_pcs_ref, n_pcs_tgt)
    if n_pcs_ref != n_pcs_tgt:
        # Keep leading PCs; adjust here if you prefer another selection
        X_ref = X_ref[:n_feat, :]
        X_tgt = X_tgt[:n_feat, :]

    # Transpose to samples×features for CCA
    Xr = X_ref.T          # (n_times, n_feat)
    Xt = X_tgt.T          # (n_times, n_feat)
    Xr_sub = Xr[time_mask]
    Xt_sub = Xt[time_mask]

    # Standardize using subset stats, apply to full series
    if standardise:
        scaler_r = StandardScaler().fit(Xr_sub)
        scaler_t = StandardScaler().fit(Xt_sub)
        Xr_full_std = scaler_r.transform(Xr)
        Xt_full_std = scaler_t.transform(Xt)
        Xr_sub_std  = Xr_full_std[time_mask]
        Xt_sub_std  = Xt_full_std[time_mask]
    else:
        scaler_r = scaler_t = None
        Xr_full_std, Xt_full_std = Xr, Xt
        Xr_sub_std,  Xt_sub_std  = Xr_sub, Xt_sub

    # Number of canonical components
    if n_components is None:
        n_components = n_feat
    n_components = min(n_components, n_feat)

    # Fit CCA on the subset only
    cca = CCA(n_components=n_components)
    Zr_sub, Zt_sub = cca.fit(Xr_sub_std, Xt_sub_std).transform(Xr_sub_std, Xt_sub_std)

    # Transform FULL time series with the fitted model
    Zr_full, Zt_full = cca.transform(Xr_full_std, Xt_full_std)  # (n_times, n_components)

    # Optional: fix signs to make subset correlations positive component-wise
    flips = np.ones(n_components)
    if fix_signs:
        # compute Pearson sign on subset and flip Zt to match Zr
        for k in range(n_components):
            r = np.corrcoef(Zr_sub[:, k], Zt_sub[:, k])[0, 1]
            if np.isnan(r) or r < 0:
                Zt_full[:, k] *= -1
                Zt_sub[:, k]  *= -1
                flips[k] = -1.0

    # Return in (components × time) like your input
    X_ref_c_full = Zr_full.T
    X_tgt_c_full = Zt_full.T

    meta = {
        "scaler_ref": scaler_r,
        "scaler_tgt": scaler_t,
        "n_components": n_components,
        "n_features_used": n_feat,
        "sign_flips_tgt": flips,
        "time_mask": time_mask.copy(),
    }
    return X_ref_c_full, X_tgt_c_full, cca, meta

def procrustes_align_pca_timeseries(
    X_ref, X_tgt, time_mask, n_components=None, standardise=False, fix_signs=False
):
    """
    Align two PCA×time trajectories via Orthogonal Procrustes (rotation-only)
    trained on a subset of timepoints, then apply to the full trajectories.

    Parameters
    ----------
    X_ref : (n_pcs, n_times) array
        Reference PCA time series.
    X_tgt : (n_pcs, n_times) array
        Target PCA time series to align to the reference.
    time_mask : (n_times,) boolean array
        True for the timepoints used to FIT the rotation (alignment window).
    n_components : int or None
        Number of PCs to keep after alignment (from the reference ordering).
        If None, uses min(n_pcs_ref, n_pcs_tgt).
    standardise : bool
        If True, z-score each series using stats computed on the subset, then
        apply the same scaling to the full series. Returned results are mapped
        back to the reference's ORIGINAL units.
    fix_signs : bool
        If True, flips individual aligned target axes so correlations with the
        reference on the subset are positive (after rotation).

    Returns
    -------
    X_ref_full : (n_components, n_times)
        Reference time series (possibly standardised then *inverse* transformed
        back to original units), truncated to n_components.
    X_tgt_aligned_full : (n_components, n_times)
        Target time series rotated into the reference space (and mapped back to
        the reference's units), full time axis.
    R : (n_feat, n_feat) ndarray
        The orthogonal rotation matrix learned on the subset (acts on features).
    meta : dict
        {"scaler_ref","scaler_tgt","n_components","n_features_used",
         "sign_flips_tgt","time_mask","op_scale"}
    """
    # --- validate shapes ---
    X_ref = np.asarray(X_ref)
    X_tgt = np.asarray(X_tgt)
    time_mask = np.asarray(time_mask, dtype=bool)

    n_pcs_ref, n_times_ref = X_ref.shape
    n_pcs_tgt, n_times_tgt = X_tgt.shape
    if n_times_ref != n_times_tgt:
        raise ValueError(f"Time dimension mismatch: {n_times_ref} vs {n_times_tgt}")
    if time_mask.shape[0] != n_times_ref:
        raise ValueError("time_mask must be boolean of length n_times")

    # --- harmonize feature count ---
    n_feat = min(n_pcs_ref, n_pcs_tgt)
    X_ref = X_ref[:n_feat, :]
    X_tgt = X_tgt[:n_feat, :]
    if n_components is None:
        n_components = n_feat
    n_components = min(n_components, n_feat)

    # Work in samples×features form
    Xr = X_ref.T            # (n_times, n_feat)
    Xt = X_tgt.T            # (n_times, n_feat)
    Xr_sub = Xr[time_mask]  # (n_sub, n_feat)
    Xt_sub = Xt[time_mask]

    # Standardize using subset stats; keep objects for inverse-transform
    if standardise:
        scaler_r = StandardScaler().fit(Xr_sub)
        scaler_t = StandardScaler().fit(Xt_sub)
        Xr_full_std = scaler_r.transform(Xr)
        Xt_full_std = scaler_t.transform(Xt)
        Xr_sub_std  = Xr_full_std[time_mask]
        Xt_sub_std  = Xt_full_std[time_mask]
    else:
        scaler_r = scaler_t = None
        Xr_full_std, Xt_full_std = Xr, Xt
        Xr_sub_std,  Xt_sub_std  = Xr_sub, Xt_sub

    # --- learn orthogonal rotation on the subset ---
    # Find R such that Xt_sub_std @ R ≈ Xr_sub_std
    R, op_scale = orthogonal_procrustes(Xt_sub_std, Xr_sub_std)  # R: (n_feat, n_feat)

    # Apply rotation to FULL time axis (still in standardized units if used)
    Xt_full_rot = Xt_full_std @ R
    Xt_sub_rot  = Xt_sub_std  @ R

    # Optional: make each aligned component positively correlated with reference on the subset
    flips = np.ones(n_feat)
    if fix_signs:
        for k in range(n_feat):
            r = np.corrcoef(Xr_sub_std[:, k], Xt_sub_rot[:, k])[0, 1]
            if np.isnan(r) or r < 0:
                Xt_full_rot[:, k] *= -1
                Xt_sub_rot[:, k]  *= -1
                flips[k] = -1.0

    # Map to reference's ORIGINAL units if we standardised
    if standardise:
        X_ref_full_units = scaler_r.inverse_transform(Xr_full_std)     # reference back to its units
        X_tgt_full_units = scaler_r.inverse_transform(Xt_full_rot)     # target in ref units
    else:
        X_ref_full_units = Xr_full_std
        X_tgt_full_units = Xt_full_rot

    # Keep requested number of components (use reference ordering)
    X_ref_full = X_ref_full_units[:, :n_components].T        # (n_components, n_times)
    X_tgt_aligned_full = X_tgt_full_units[:, :n_components].T

    meta = {
        "scaler_ref": scaler_r,
        "scaler_tgt": scaler_t,
        "n_components": n_components,
        "n_features_used": n_feat,
        "sign_flips_tgt": flips[:n_components],
        "time_mask": time_mask.copy(),
        "op_scale": op_scale,          # sum of singular values (diagnostic)
        "R": R,                        # rotation (acts on features/PCs)
    }
    return X_ref_full, X_tgt_aligned_full, R, meta