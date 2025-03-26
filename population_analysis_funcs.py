import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
import neo
import numpy as np
import pandas as pd
from matplotlib import animation
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import KernelDensity

from ephys_analysis_funcs import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import euclidean
from scipy.stats import norm
from scipy.stats import bootstrap, entropy
import argparse
import yaml
import platform
from elephant.gpfa import GPFA
from elephant.conversion import BinnedSpikeTrain
import neo
from sklearn.model_selection import cross_val_score
from os import cpu_count
from itertools import combinations

from behviour_analysis_funcs import get_sess_name_date_idx
from sythesise_spikes import integrated_oscillator, random_projection,generate_spiketrains
import quantities as pq
from IPython.display import HTML
from IPython import display
from neural_similarity_funcs import compare_pip_sims_2way,plot_similarity_mat
# from npeet import entropy_estimators as ee


def get_event_response(event_response_dict, event):
    event_responses = np.array_split(event_response_dict[event], event_response_dict[event].shape[0], axis=0)

    event_responses = [np.squeeze(e) for e in event_responses]

    return event_responses


def get_event_trajs(event_response_dict, event, pc_idxs,X_pca, subset=None, mean_axis=None):
    event_trials = get_event_response(event_response_dict, event)
    projected_trials_by_event = [project_pca(trial, X_pca, standardise=False)
                                 for trial in event_trials]
    by_pc_traj = format_tractories(projected_trials_by_event, pc_idxs, subset=subset, mean_axis=mean_axis)
    return by_pc_traj


def format_tractories(trajectories, pc_idxs, subset=None, mean_axis=None):
    trajectories = np.transpose(np.array(trajectories), (1, 0, 2))
    by_pc_traj = [trajectories[idx] for idx in pc_idxs]
    if subset is not None:
        by_pc_traj = [traj[subset] for traj in by_pc_traj]

    if mean_axis is not None:
        by_pc_traj = [traj.mean(axis=mean_axis) for traj in by_pc_traj]

    return by_pc_traj

def get_population_pca(rate_arr:np.ndarray):

    assert rate_arr.ndim == 3


def z_score(X):
    # X: ndarray, shape (n_features, n_samples)
    ss = StandardScaler(with_mean=False, with_std=False)
    Xz = ss.fit_transform(X.T).T
    return Xz


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


def run_gpfa(spike_trains:list[list[neo.SpikeTrain]],bin_dur=0.02,latent_dimensionality=2):

    gpfa_ndim = GPFA(bin_size=bin_dur*s, x_dim=latent_dimensionality)
    trajectories = gpfa_ndim.fit_transform(spike_trains)
    # trajectories = gpfa_ndim.transform(spike_trains)
    return gpfa_ndim, trajectories


def cross_val_gpfa(spike_trains, x_dims):
    log_likelihoods = []
    for x_dim in x_dims:
        gpfa_cv = GPFA(x_dim=x_dim)
        # estimate the log-likelihood for the given dimensionality as the mean of the log-likelihoods from 3 cross-vailidation folds
        cv_log_likelihoods = cross_val_score(gpfa_cv, spike_trains, cv=3, n_jobs=cpu_count()-1, verbose=True,
                                             error_score='raise')
        log_likelihoods.append(np.mean(cv_log_likelihoods))

    return log_likelihoods


def plot_pca_traj(by_pc_traj, pc_idxs, mean_axis=None, plot=None, subset=None, plt_kwargs=None):

    if plot:
        fig: plt.Figure = plot[0]
        ax:plt.Axes = plot[1]
        assert not isinstance(ax,np.ndarray), 'ax must be a single axis'
    else:
        plot = plt.subplots()
        fig: plt.Figure = plot[0]
        ax: plt.Axes = plot[1]
    to_plot = [e.mean(axis=mean_axis) for e in by_pc_traj] if mean_axis is not None else by_pc_traj
    if len(pc_idxs) == 2:
        ax.plot(to_plot[pc_idxs[0]], to_plot[pc_idxs[1]], **plt_kwargs)
    elif len(pc_idxs) > 2:
        ax.plot(to_plot[pc_idxs[0]], to_plot[pc_idxs[1]],to_plot[pc_idxs[2]], **plt_kwargs)
    if mean_axis is not None:
        if len(pc_idxs) == 2:
            ax.scatter(to_plot[pc_idxs[0]][0], to_plot[pc_idxs[1]][0], c=plt_kwargs.get('c', 'k'))
        elif len(pc_idxs) > 2:
            ax.scatter(to_plot[pc_idxs[0]][0], to_plot[pc_idxs[1]][0], to_plot[pc_idxs[2]][0],
                       c=plt_kwargs.get('c', 'k'))

    set_label_funcs = (ax.set_xlabel,ax.set_ylabel)
    [label_ax(f'PC {pc_idx}') for label_ax,pc_idx in zip(set_label_funcs,pc_idxs)]
    ax.set_xlabel(f'PC {pc_idxs[0]}')
    ax.set_ylabel(f'PC {pc_idxs[1]}')
    # return fig,ax

def style_3d_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')


def compute_mi(x,cond_responses_by_prop: list | dict |np.ndarray,
               bins:np.ndarray):

    p_x, _ = np.histogram(x, density=True, bins=bins)
    p_x = p_x[p_x > 0]
    p_x = p_x / p_x.sum()

    # entropy_x = entropy(x)
    entropy_x = -(p_x * np.log2(p_x)).sum()
    # entropy_x = -np.sum(p_x*np.log2(p_x +1e-5))
    # print(entropy_x)
    assert entropy_x >= 0, f'{entropy_x = } {p_x = }'
    entropy_x_given_y_vals = []
    p_y = np.array([len(cur_list) for cur_list in cond_responses_by_prop[0]])
    p_y  = p_y / np.sum(p_y)
    print(f'{p_y = }')
    for cond_responses in cond_responses_by_prop:
        x_given_y = np.hstack(cond_responses)
        # get kde of x_given_y
        # model = KernelDensity()
        # model.fit(x_given_y.reshape(-1, 1))
        # probabilities = model.score_samples(np.linspace(bins.min(), bins.max(), len(bins)).reshape(-1, 1))
        # probabilities = np.exp(probabilities)
        # p_x_given_y = probabilities[probabilities > 0.0]
        p_x_given_y, _ = np.histogram(x_given_y, density=True, bins=bins)
        p_x_given_y = p_x_given_y[p_x_given_y > 0]
        p_x_given_y = p_x_given_y/p_x_given_y.sum()
        p_x_given_y = p_x_given_y / p_x_given_y.sum()
        print(p_x_given_y)
        # p_x_given_y = p_x_given_y[p_x_given_y > 0]
        # [print(e.max()) for e in cond_responses]
        # p_x_given_y, _ = np.histogram(x_given_y[x_given_y>0], density=True, bins=bins)
        # p_x_given_y = p_x_given_y[p_x_given_y > 0]
        # entropy_x_given_y_vals += [entropy(np.hstack(cond_responses))]
        entropy_x_given_y_vals += [-np.sum(p_x_given_y* np.log(p_x_given_y))]
    assert ((entropy_x - np.array(entropy_x_given_y_vals) @ p_y) < 0).sum() == 0, f'{entropy_x = } {entropy_x_given_y_vals = }'
    return entropy_x - np.array(entropy_x_given_y_vals) @ p_y
    # return np.mean(mis_by_prop)
    #
    # for j in range(len_abs):  # loop over different rules
    #     # p_x_given_y is the distribution of responses to stimuli following rule y
    #     p_x_given_y, _ = np.histogram(y_h_trials_one[:, j, :, t, i], density=True, bins=bins)
    #     entropy_x_given_y_vals += [
    #         - np.sum(dx * p_x_given_y * np.log2(p_x_given_y + 1e-9))]  # entropy for a single rule


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('pkldir')
    parser.add_argument('--sessname',default=None)
    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])

    if sys_os == 'windows':
        import pathlib
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath

    sessions = {}
    sess_topology_path = ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology.csv')
    session_topology = pd.read_csv(sess_topology_path)
    # sessname = 'DO81_240529b' if args.sessname is None else args.sessname
    sessnames = args.sessname.split('-')
    pkldir = ceph_dir / posix_from_win(args.pkldir)
    standardise_flag = False
    figdir = ceph_dir/'Dammy'/'figures'/f'popualation_analysis_trial_concat'
    if standardise_flag and 'standardised' not in figdir.stem:
        figdir = figdir.with_stem(f'{figdir.stem}_standardised')

    if not figdir.is_dir():
        figdir.mkdir(parents=True)
    for sessname in sessnames:
        name,date,sess_idx,sess_info = get_sess_name_date_idx(sessname,session_topology)
        assert pkldir.is_dir()
        sess_pkls = list(pkldir.glob(f'{sessname}.pkl'))
        with open(sess_pkls[0], 'rb') as f:
            sessions[sessname]: Session = pickle.load(f)
        # sessions[sessname]: Session = pd.read_pickle(sess_pkls[0])

        window = [-0.0,0.25]
        plot_dist_metrics = False
        # window = [0,2]
        for e in list(sessions[sessname].sound_event_dict.keys()):
            sessions[sessname].spike_obj.get_event_spikes(sessions[sessname].sound_event_dict[e].times,e,window,
                                                          get_spike_matrix=False)

        # sessions[sessname].init_spike_obj(spike_times_path, spike_cluster_path, start_time, parent_dir=spike_dir)



        all_events = [e for e in sessions[sessname].sound_event_dict.keys() if any(pip in e for pip in ['A','B','C','D'])]
        event_psth_dict = {e: get_predictor_from_psth(sessions[sessname], e, [-2,3], window,mean=None,
                                                      use_unit_zscore=True, use_iti_zscore=False,baseline=0)
                           for e in all_events}
        full_event_psth_dict = {e: get_predictor_from_psth(sessions[sessname], e, [-2, 3], [0,1], mean=None,
                                                           use_unit_zscore=True, use_iti_zscore=False, baseline=0)
                                for e in [f'A-{i}' for i in range(4)]}
        _all_trials = [np.array_split(event_psth_dict[e], event_psth_dict[e].shape[0], axis=0)
                       for e in all_events]
        event_psth_dict['full_normal'] = get_predictor_from_psth(sessions[sessname], 'A-0',[-2,3], [-0.1,1], mean=None,
                                                                  use_unit_zscore=True, use_iti_zscore=False)
        if 4 in sessions[sessname].td_df['Stage'].values:
            event_psth_dict['full_deviant'] = get_predictor_from_psth(sessions[sessname], 'A-1', [-2, 3], [-0.1, 1],
                                                                      mean=None,
                                                                      use_unit_zscore=True, use_iti_zscore=False)
        # _all_trials = np.split(np.array(trial_oscillator),6)
        x_ser = np.linspace(window[0], window[1], _all_trials[0][0].shape[-1])
        try:
            timepoints = [np.argwhere(x_ser==t)[0][0] for t in np.arange(0.25,1.25,0.25)]
        except IndexError:
            timepoints=[np.argwhere(x_ser==x_ser[-1])[0][0]]
        _all_trials = [[np.squeeze(ee) for ee in e] for e in _all_trials]
        Xa_trial_concatenated = np.hstack(sum(_all_trials, []))
        Xa_trial_averaged = [np.mean(e,axis=0) for e in _all_trials]
        Xa_trial_concatenated_pca = compute_trial_averaged_pca(Xa_trial_concatenated, n_components=15, standardise=standardise_flag)
        Xa_trial_averaged_pca = compute_trial_averaged_pca(np.hstack([e for e in Xa_trial_averaged]),
                                                           n_components=15, standardise=standardise_flag)
        eig_vals = compute_eig_vals(Xa_trial_concatenated,plot_flag=True)
        eig_vals_averaged = compute_eig_vals(np.hstack(Xa_trial_averaged),plot_flag=True)
        eig_vals_averaged[2][0].show()
        n_comp_toplot = 8
        pca_ts_plot = plt.subplots(4,n_comp_toplot,figsize=[100,15],squeeze=False)
        rules = ['ABCD 0','ABBA 0','ABCD 1','ABBA 1','ABCD 2','ABBA 2']
        pca_scatter_plot = plt.subplots(ncols=3,figsize=[15,5])
        # for ii,pip in enumerate(['A','B','C','D']):
        #     events = [e for e in all_events if e[0] == pip]
        #     # for event in events:
        #     #
        #     #     event_psth_dict[event] = get_predictor_from_psth(sessions[sessname], event, [-2,3], window,mean=None,
        #     #                                                      use_unit_zscore=False, use_iti_zscore=False)
        #     # Xa_trial_averaged = np.hstack([event_psth_dict[e].mean(axis=0) for e in events])
        #     # x_shape = event_psth_dict[events[0]].shape
        #     # eig_vals = compute_eig_vals(Xa_trial_averaged,plot_flag=True)
        #     # Xa_trial_averaged_pca = compute_trial_averaged_pca(Xa_trial_averaged,n_components=15,standardise=standardise_flag)
        #     # projected_trials_by_event = [[project_pca(trial,Xa_trial_averaged_pca,standardise=standardise_flag)
        #     #                               for trial in trials_by_event] for trials_by_event in event_psth_dict.values()]
        #     # plot_pca_ts(projected_trials_by_event,events,window,n_components=8,plot=[pca_ts_plot[0], pca_ts_plot[1][ii]])
        #
        #     # _all_trials = [np.array_split(event_psth_dict[e],event_psth_dict[e].shape[0],axis=0) for e in events]
        #     # _all_trials = [[np.squeeze(ee) for ee in e] for e in _all_trials]
        #     # Xa_trial_concatenated = np.hstack(sum(_all_trials,[]))
        #
        #     # eig_vals = compute_eig_vals(Xa_trial_concatenated,plot_flag=True)
        #     # Xa_trial_concatenated_pca = compute_trial_averaged_pca(Xa_trial_concatenated,n_components=15,standardise=standardise_flag)
        #
        #     # event_trials = [np.array_split(event_psth_dict[e], event_psth_dict[e].shape[0], axis=0)
        #     #                 for e in events]
        #     event_trials = [get_event_response(event_psth_dict,event) for event in events]
        #
        #     # event_trials = _all_trials
        #     projected_trials_by_event = [[project_pca(trial,Xa_trial_concatenated_pca,standardise=standardise_flag)
        #                                   for trial in trials_by_event] for trials_by_event in event_trials]
        #     # projected_trials_by_event = [project_pca(trial, Xa_trial_averaged_pca, standardise=standardise_flag)
        #     #                               for trial in Xa_trial_averaged]
        #     plot_pca_ts(projected_trials_by_event,events,window,n_components=n_comp_toplot,plot=[pca_ts_plot[0], pca_ts_plot[1][ii]],ls={})
        #
        #     [[ax.axvline(t, ls='--', c='k') for t in np.arange(0,min([1,window[1]]),0.25)]
        #      for ax in pca_ts_plot[1][ii]]
        #
        #     [[pca_scatter_plot[1][ci].scatter(np.array(proj)[:,comps[0],timepoints[-1]],
        #                                       np.array(proj)[:,comps[1],timepoints[-1]],
        #                                  c=f'C{ii}',label=pip,alpha=0.25,marker='o')
        #      for ti,proj in enumerate(projected_trials_by_event)] for ci, comps in enumerate([[0,1],[0,2],[1,2]])]
        #     [(pca_scatter_plot[1][ci].set_xlabel(f'PC{comps[0]}'), pca_scatter_plot[1][ci].set_ylabel(f'PC{comps[1]}'))
        #      for ci, comps in enumerate([[0,1],[0,2],[1,2]])]
        #     # pca_ts_plot[1][ii]
        # # pca_ts_plot[0].set_layout_engine('constrained')
        # pca_ts_plot[0].show()
        # pca_ts_plot[0].savefig(figdir/f'pca_ts_{sessname}.pdf')
        # unique_legend(pca_scatter_plot[1],)
        # pca_scatter_plot[0].savefig(figdir/f'pca_scatter_{sessname}.pdf')
        # # pca_scatter_plot[1][-1].legend(ncols=len(rules))
        # pca_scatter_plot[0].show()

        # prepare trial averages
        # pick the components corresponding to the x, y, and z axes
        component_x = 0
        component_y = 1
        component_z = 2

        # create a boolean mask so we can plot activity during stimulus as
        # solid line, and pre and post stimulus as a dashed line
        stim_mask = np.logical_and(x_ser>=0,x_ser<=0.25)


        # utility function to clean up and label the axes


        sigma = 3  # smoothing amount

        # set up a figure with two 3d subplots, so we can have two different views

        fig,axes = plt.subplots(ncols=2, figsize=[9, 4],subplot_kw={'projection': '3d'})
        pca_point_plot = plt.subplots(ncols=1, figsize=[9, 4],subplot_kw={'projection': '3d'})
        pca_point_plot_slice_plot = plt.subplots(ncols=3)
        # for ii,pip in enumerate(['A','B','C','D','base'][:1]):
        #     events = [e for e in event_psth_dict if pip in e]
        for ii, pip in enumerate([f'A-{i}' for i in range(4)]):
            events = [e for e in event_psth_dict if pip in e]

            # event_trials = [np.array_split(event_psth_dict[e], event_psth_dict[e].shape[0], axis=0)
            #                 for e in events]
            event_trials = [get_event_response(full_event_psth_dict, event) for event in events]
            projected_trials_by_event = [[project_pca(trial, Xa_trial_averaged_pca, standardise=standardise_flag)
                                          for trial in trials_by_event] for trials_by_event in event_trials]
            project_trials_arr = np.array(projected_trials_by_event)[0]
            # pca_point_plot[1].scatter(project_trials_arr[:,0,0],project_trials_arr[:,1,0],project_trials_arr[:,2,0],
            #                           c=f'C{ii}',alpha=0.25,marker='o')
            pca_point_plot[1].scatter(project_trials_arr[:, 0, -1], project_trials_arr[:, 1, -1],
                                      project_trials_arr[:, 2, -1],
                                      c=f'C{ii}', alpha=0.5, marker='x')
            [ax.scatter(project_trials_arr[:, pc[0], -1],project_trials_arr[:, pc[1], -1],
                        c=f'C{ii}',alpha=0.5,marker='x') for
             pc,ax in zip(list(combinations(range(3),2)),pca_point_plot_slice_plot[1])]
            # [ax.scatter(project_trials_arr[:, pc[0], 0],project_trials_arr[:, pc[1], -1],
            #             c=f'C{ii}',alpha=0.5,marker='o') for
            #  pc,ax in zip(list(combinations(range(3),2)),pca_point_plot_slice_plot[1])]
            for ax in axes:
                for t, t_type in enumerate(projected_trials_by_event):
                    proj_arr = np.array(t_type).mean(axis=0)
                    # for every trial type, select the part of the component
                    # which corresponds to that trial type:
                    x = proj_arr[component_x]
                    y = proj_arr[component_y]
                    z = proj_arr[component_z]


                    # # apply some smoothing to the trajectories
                    x = gaussian_filter1d(x, sigma=sigma)
                    y = gaussian_filter1d(y, sigma=sigma)
                    z = gaussian_filter1d(z, sigma=sigma)

                    # use the mask to plot stimulus and pre/post stimulus separately
                    z_stim = z.copy()
                    # z_stim[~stim_mask] = np.nan
                    z_prepost = z.copy()
                    # z_prepost[stim_mask] = np.nan

                    ax.plot(x, y, z_stim, c=f'C{ii}',ls='-' if 'A' in pip else '--')
                    ax.plot(x, y, z_prepost, c=f'C{ii}', ls=':')

                    # plot dots at initial point
                    ax.scatter(x[0], y[0], z[0], c=f'C{ii}', s=14,label=pip)
                    ax.scatter(x[-1], y[-1], z[-1], c=f'C{ii}', s=14,label=pip,marker='x')

                    # make the axes a bit cleaner
                    style_3d_ax(ax)
                    style_3d_ax(pca_point_plot[1])

            # specify the orientation of the 3d plot
            axes[0].view_init(elev=22, azim=30)
            pca_point_plot[1].view_init(elev=22, azim=30)
            # axes[0].set_title(f'{pip} 3D view')
            # axes[1].view_init(elev=22, azim=110)
        axes[-1].legend()
        plt.tight_layout()

        fig.show()
        pca_point_plot[0].show()
        pca_point_plot_slice_plot[0].show()
        fig.savefig(figdir/f'pca_3d_{sessname}.pdf')

        # for ii, pip in enumerate(['A', 'B', 'C', 'D']):
        pip = 'A'
        fig, axes = plt.subplots(ncols=2, figsize=[9, 4], subplot_kw={'projection': '3d'})
        events = [e for e in event_psth_dict if e.startswith(pip)]


        projected_trials_by_event = [[project_pca(trial, Xa_trial_averaged_pca, standardise=standardise_flag)
                                      for trial in trials_by_event] for trials_by_event in event_trials]

        by_cond_by_event_trajs = {}
        n_pcs = 10
        if 3 in sessions[sessname].td_df['Stage'].values:
            # group by rare or freq
            sessions[sessname].td_df['cum_patts'] = np.cumsum(sessions[sessname].td_df['Tone_Position'] == 0)
            rare_filt = 'Tone_Position==0 and local_rate>=0.8'
            freq_filt = 'Tone_Position==0 and local_rate<=0.1'

            patt_trial_idxs = sessions[sessname].sound_event_dict['A-0'].trial_nums - 1

            by_cond_trial_nums = [sessions[sessname].td_df.query(cond_filt).index.values
                                  for cond_filt in [rare_filt, freq_filt]]
            by_cond_trial_idxs = [[np.argwhere(idx == patt_trial_idxs)[0] for idx in cond_idxs if idx in patt_trial_idxs]
                                  for cond_idxs in by_cond_trial_nums]
            by_cond_trial_idxs = [np.hstack(e) for e in by_cond_trial_idxs]

            cond_lbls = ['rare', 'freq']
            by_cond_ls = ['-', '--']
            [[by_cond_by_event_trajs.update({f'{e[0]}_{e_lbl}': get_event_trajs(event_psth_dict, e, list(range(n_pcs)),
                                                                                Xa_trial_concatenated_pca,
                                                                                subset=e_subset)})
             for e in [f'{ee}-0' for ee in 'ABCD']]
             for e_subset,e_lbl in zip(by_cond_trial_idxs,cond_lbls)]

        if 4 in sessions[sessname].td_df['Stage'].values:
            cond_lbls = ['normal', 'deviant']
            by_cond_ls = ['-', '--']
            normal_filt = 'Tone_Position==0 and Pattern_Type == 0 and Session_Block==2'
            dev0_idx = sessions[sessname].td_df.query('Tone_Position==0 and Pattern_Type == 1').index.values[0]
            normal_patt_trial_idxs = sessions[sessname].sound_event_dict['A-0'].trial_nums - 1

            by_cond_trial_nums = [sessions[sessname].td_df.query(cond_filt).index.values
                                  for cond_filt in [normal_filt]]
            by_cond_trial_idxs = [
                [np.argwhere(idx == normal_patt_trial_idxs)[0] for idx in cond_idxs if idx in normal_patt_trial_idxs]
                for cond_idxs in by_cond_trial_nums]
            by_cond_trial_idxs = [np.hstack(e) for e in by_cond_trial_idxs]

            normal_subset = by_cond_trial_idxs[0] > dev0_idx
            [[by_cond_by_event_trajs.update({f'{e[0]}_{e_lbl}': get_event_trajs(event_psth_dict, e, list(range(n_pcs)),
                                                                                Xa_trial_concatenated_pca,
                                                                                subset=cond_idxs)})
              for e in [f'{ee}-{cond_i}' for ee in 'ABCD']]
             for cond_i, (cond_idxs,e_lbl) in enumerate(zip([normal_subset, None], cond_lbls))]

        pc_combs = list(combinations(range(3), 2))
        pca_traj_plot = plt.subplots(ncols=len(pc_combs), figsize=[9, 4], sharey=True)
        pca_traj_plot_3d = plt.subplots(figsize=(12,9),subplot_kw={'projection': '3d'})

        for pi, pip in enumerate('ABCD'):
            by_pc_traj = [by_cond_by_event_trajs[f'{pip}_{cond}'] for cond in cond_lbls]
            # plot pc trajectories
            [[[plot_pca_traj(trajs2plot, pc_comb, mean_axis=mean_type,
                             plt_kwargs={'c':f'C{pi}', 'ls':cond_ls, 'lw':mean_lw, 'label': lbl},
                             plot=(pca_traj_plot[0],pca_traj_plot[1][comb_i]))
               for comb_i,pc_comb in enumerate(pc_combs)]
             for trajs2plot,cond_ls,lbl in zip(by_pc_traj,by_cond_ls,cond_lbls)]
             for mean_type, mean_lw in zip([0],[2,0.1])]
            [[plot_pca_traj(trajs2plot, [0,1,2], mean_axis=mean_type,
                            plt_kwargs={'c': f'C{pi}', 'ls': cond_ls, 'lw': mean_lw, 'label': lbl},
                            plot=(pca_traj_plot_3d[0], pca_traj_plot_3d[1]))
              for trajs2plot, cond_ls, lbl in zip(by_pc_traj, by_cond_ls, cond_lbls)]
             for mean_type, mean_lw in zip([0], [2, 0.1])]
        pca_traj_plot[0].legend(loc='upper right')
        pca_traj_plot[0].tight_layout(rect=[0, 0, 0.9, 1])

        pca_traj_plot[0].show()
        pca_traj_plot[0].savefig(figdir/f'pca_trajs_{sessname}.pdf')

        style_3d_ax(pca_traj_plot_3d[1])
        pca_traj_plot_3d[0].legend(loc='upper right')
        pca_traj_plot_3d[0].tight_layout(rect=[0, 0, 0.9, 1])

        pca_traj_plot_3d[0].show()
        pca_traj_plot_3d[0].savefig(figdir/f'pca_trajs_3d_{sessname}.pdf')

        if plot_dist_metrics:
            if 3 in sessions[sessname].td_df['Stage'].values:
                [by_cond_by_event_trajs.update({f'full_{e_lbl}': get_event_trajs(event_psth_dict, e, list(range(n_pcs)),
                                                                                 subset=e_subset)})
                 for e_subset, e_lbl in zip(by_cond_trial_idxs, cond_lbls)]

            if 4 in sessions[sessname].td_df['Stage'].values:
                [by_cond_by_event_trajs.update({e: get_event_trajs(event_psth_dict, e, list(range(n_pcs)))})
                 for e in ['full_normal','full_deviant']]
            projected_pattern_pc123 = [by_cond_by_event_trajs[f'full_{cond}'] for cond in cond_lbls]
            projection_sim_ts = [compare_pip_sims_2way([np.array(projected_pattern_pc123[0]).transpose((1,0,2)),
                                                        np.array(projected_pattern_pc123[1]).transpose((1,0,2))],
                                                       t=t,mean_flag=True)
                                 for t in range(projected_pattern_pc123[0][0].shape[-1])]  # len = n time points
            mean_comped_sims = np.array([[np.squeeze(pip_sims)[:, 0, 1] for pip_sims in
                                          np.array_split(t_sim[0], (len(cond_lbls)))]
                                         for t_sim in projection_sim_ts])
            mean_cross_comped_sims = np.array([np.squeeze(t_sim[1][:, 0, 1]) for t_sim in projection_sim_ts])

            full_pattern_xser = np.linspace(-0.1,1, projected_pattern_pc123[0][0].shape[1])
            by_cond_sim_ts_plot = plt.subplots()
            [by_cond_sim_ts_plot[1].plot(full_pattern_xser, mean_comped_sims[:, cond_i].mean(axis=1), label=cond_lbl)
             for cond_i, cond_lbl in enumerate(cond_lbls)]
            by_cond_sim_ts_plot[1].plot(full_pattern_xser, mean_cross_comped_sims.mean(axis=1), label=" vs ".join(cond_lbls),
                                        c='k')
            by_cond_sim_ts_plot[1].set_ylabel('cosine similarity')
            by_cond_sim_ts_plot[1].set_xlabel('Time (s)')
            by_cond_sim_ts_plot[1].set_title(f'{"vs ".join(cond_lbls)} response similarity')
            by_cond_sim_ts_plot[0].legend()
            by_cond_sim_ts_plot[0].show()
            by_cond_sim_ts_plot[0].savefig(figdir/f'{"vs".join(cond_lbls)}_sim_ts_{sessname}.pdf')

            projected_pattern_pc_means = [[ee.mean(axis=0) for ee in e] for e in projected_pattern_pc123]  # need to take mean of each event by cond
            projection_distance_ts = [euclidean(np.array(projected_pattern_pc_means[0])[:, t],
                                                np.array(projected_pattern_pc_means[1])[:, t])
                                      for t in range(projected_pattern_pc_means[0][0].shape[0])]

            pca_euc_dist_plot = plt.subplots()
            pca_euc_dist_plot[1].plot(full_pattern_xser,projection_distance_ts)
            pca_euc_dist_plot[1].set_ylabel('Euclidean distance')
            pca_euc_dist_plot[1].set_xlabel('Time (s)')
            pca_euc_dist_plot[1].set_title('Pattern response')
            pca_euc_dist_plot[0].show()
            pca_euc_dist_plot[0].savefig(figdir/f'pca_euc_dist_{sessname}.pdf')



#
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# from matplotlib import pyplot as plt
# import matplotlib.animation as animation
# import pandas as pd
#
# a = np.random.rand(2000, 3)*10
# t = np.array([np.ones(100)*i for i in range(20)]).flatten()
# df = pd.DataFrame({"time": t ,"x" : a[:,0], "y" : a[:,1], "z" : a[:,2]})
#
# def update_graph(num):
#     data=df[df['time']==num]
#     graph._offsets3d = (data.x, data.y, data.z)
#     title.set_text('3D Test, time={}'.format(num))
#     return graph, title
#
# fig = plt.figure()
#
# ax = fig.add_subplot(221, projection='3d')
# ax2 = fig.add_subplot(222, projection='3d')
# title = ax.set_title('3D Test')
# ax.scatter(df.x, df.y, df.z)
#
# data=df[df['time']==0]
# graph = ax2.scatter(data.x, data.y, data.z)
#
# ani = animation.FuncAnimation(fig, update_graph, 20,
#                                    interval=50, blit=True)
# ani.save('motion.gif', writer="imagemagick")
# plt.show()
