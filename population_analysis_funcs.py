import matplotlib.pyplot as plt
import neo
import numpy as np
import pandas as pd
from matplotlib import animation
from ephys_analysis_funcs import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import euclidean
from scipy.stats import bootstrap
import argparse
import yaml
import platform
from elephant.gpfa import GPFA
from elephant.conversion import BinnedSpikeTrain
import neo
from sklearn.model_selection import cross_val_score
from os import cpu_count
from itertools import combinations
from sythesise_spikes import integrated_oscillator, random_projection,generate_spiketrains
import quantities as pq
from IPython.display import HTML
from IPython import display
from neural_similarity_funcs import compare_pip_sims_2way,plot_similarity_mat


def get_event_response(event_response_dict, events):
    event_responses = [np.array_split(event_response_dict[e], event_response_dict[e].shape[0], axis=0)
                    for e in events]
    event_responses = [[np.squeeze(ee) for ee in e] for e in event_responses]

    return event_responses


def get_event_trajs(event_response_dict, events, pc_idxs, subset=None,mean_axis=None):
    event_trials = get_event_response(event_response_dict,events)
    projected_trials_by_event = [[project_pca(trial, Xa_trial_averaged_pca, standardise=False)
                                  for trial in trials_by_event] for trials_by_event in event_trials]
    by_pc_traj = [format_tractories(trajectories, pc_idxs, subset=subset, mean_axis=mean_axis)
                  for trajectories in projected_trials_by_event]
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


def compute_trial_averaged_pca(X_trial_averaged,n_components=15,standardise=True):
    # Xa = z_score(X_trial_averaged)
    if standardise:
        X_std = StandardScaler().fit_transform(X_trial_averaged)
    else:
        X_std = X_trial_averaged
    pca = PCA(n_components=n_components)
    pca.fit(X_std.T)

    return pca


def project_pca(X_trial,pca,standardise=True):
    # ss = StandardScaler(with_mean=True, with_std=True)
    if standardise:
        trial_sc = StandardScaler().fit_transform(X_trial)
    else:
        trial_sc = X_trial
    proj_trial = pca.transform(trial_sc.T).T
    return proj_trial


def plot_pca_ts(X_proj_by_event, events, window, plot=None, n_components=3):
    if not plot:
        fig, axes = plt.subplots(1, n_components, figsize=[20, 4],)
    else:
        fig,axes = plot

    x_ser = np.linspace(window[0], window[1], X_proj_by_event[0][0].shape[-1])

    for comp in range(n_components):
        ax = axes[comp]
        for ei, event in enumerate(events):
            projected_trials = np.array(X_proj_by_event[ei])
            projected_trials_comp = projected_trials[:,comp,:]
            ax.plot(x_ser, projected_trials_comp.mean(axis=0), label=event)
            plot_ts_var(x_ser,projected_trials_comp,f'C{ei}',ax)

        ax.set_ylabel(f'PC {comp+1}')
        ax.set_xlabel('Time (s)')
        ax.axvline(0, color='k', ls='--')
        ax.legend(ncol=len(events))

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
    if len(by_pc_traj) == 2:
        ax.plot(by_pc_traj[0], by_pc_traj[1], **plt_kwargs)
    elif len(by_pc_traj) == 3:
        ax.plot(by_pc_traj[0], by_pc_traj[1], by_pc_traj[2], **plt_kwargs)
    if mean_axis is not None:
        if len(by_pc_traj) == 2:
            ax.scatter(by_pc_traj[0][0], by_pc_traj[1][0], c=plt_kwargs.get('c', 'k'))
        elif len(by_pc_traj) > 2:
            ax.scatter(by_pc_traj[0][0], by_pc_traj[1][0], by_pc_traj[2][0], c=plt_kwargs.get('c', 'k'))

    set_label_funcs = (ax.set_xlabel,ax.set_ylabel)
    [label_ax(f'PC {pc_idx}') for label_ax,pc_idx in zip(set_label_funcs,pc_idxs)]
    # ax.set_xlabel(f'PC {pc_idxs[0]}')
    # ax.set_ylabel(f'PC {pc_idxs[1]}')
    return fig,ax

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
    sessname = 'DO81_240529b' if args.sessname is None else args.sessname
    name,date,suffix = sessname.split('_')[0],int(sessname.split('_')[1][:-1]),sessname[-1]
    sessions_on_date = session_topology.query('name==@name and date==@date').reset_index()
    sess_suffixes = [Path(e).stem[-1]for e in sessions_on_date['sound_bin']]
    sess_idx = [ei for ei,e in enumerate(sess_suffixes) if e == suffix][0]
    sess_info = sessions_on_date.loc[sess_idx]

    pkldir = ceph_dir / posix_from_win(args.pkldir)
    assert pkldir.is_dir()
    sess_pkls = list(pkldir.glob(f'{sessname}.pkl'))
    with open(sess_pkls[0], 'rb') as f:
        sessions[sessname]: Session = pickle.load(f)
    # sessions[sessname]: Session = pd.read_pickle(sess_pkls[0])

    window = [-0.0,0.25]
    # window = [0,2]
    for e in list(sessions[sessname].sound_event_dict.keys()):
        sessions[sessname].spike_obj.get_event_spikes(sessions[sessname].sound_event_dict[e].times,e,window,
                                                      get_spike_matrix=False)

    # sessions[sessname].init_spike_obj(spike_times_path, spike_cluster_path, start_time, parent_dir=spike_dir)



    all_events = [e for e in sessions[sessname].sound_event_dict.keys() if any(pip in e for pip in ['A','B','C','D'])]
    event_psth_dict = {e: get_predictor_from_psth(sessions[sessname], e, [-2,3], window,mean=None,
                                                  use_unit_zscore=True, use_iti_zscore=False)
                       for e in all_events}
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
    Xa_trial_concatenated_pca = compute_trial_averaged_pca(Xa_trial_concatenated, n_components=15, standardise=False)
    Xa_trial_averaged_pca = compute_trial_averaged_pca(np.hstack([e for e in Xa_trial_averaged]),
                                                       n_components=15, standardise=False)
    eig_vals = compute_eig_vals(Xa_trial_concatenated,plot_flag=True)
    eig_vals_averaged = compute_eig_vals(np.hstack(Xa_trial_averaged),plot_flag=True)
    eig_vals_averaged[2][0].show()
    n_comp_toplot = 8
    pca_ts_plot = plt.subplots(4,n_comp_toplot,figsize=[100,15],squeeze=False)
    rules = ['ABCD 0','ABBA 0','ABCD 1','ABBA 1','ABCD 2','ABBA 2']
    pca_scatter_plot = plt.subplots(ncols=3,figsize=[15,5])
    for ii,pip in enumerate(['A','B','C','D']):
        events = [e for e in all_events if e[0] == pip]
        # for event in events:
        #
        #     event_psth_dict[event] = get_predictor_from_psth(sessions[sessname], event, [-2,3], window,mean=None,
        #                                                      use_unit_zscore=False, use_iti_zscore=False)
        # Xa_trial_averaged = np.hstack([event_psth_dict[e].mean(axis=0) for e in events])
        # x_shape = event_psth_dict[events[0]].shape
        # eig_vals = compute_eig_vals(Xa_trial_averaged,plot_flag=True)
        # Xa_trial_averaged_pca = compute_trial_averaged_pca(Xa_trial_averaged,n_components=15,standardise=True)
        # projected_trials_by_event = [[project_pca(trial,Xa_trial_averaged_pca,standardise=True)
        #                               for trial in trials_by_event] for trials_by_event in event_psth_dict.values()]
        # plot_pca_ts(projected_trials_by_event,events,window,n_components=8,plot=[pca_ts_plot[0], pca_ts_plot[1][ii]])

        # _all_trials = [np.array_split(event_psth_dict[e],event_psth_dict[e].shape[0],axis=0) for e in events]
        # _all_trials = [[np.squeeze(ee) for ee in e] for e in _all_trials]
        # Xa_trial_concatenated = np.hstack(sum(_all_trials,[]))

        # eig_vals = compute_eig_vals(Xa_trial_concatenated,plot_flag=True)
        # Xa_trial_concatenated_pca = compute_trial_averaged_pca(Xa_trial_concatenated,n_components=15,standardise=True)

        # event_trials = [np.array_split(event_psth_dict[e], event_psth_dict[e].shape[0], axis=0)
        #                 for e in events]
        event_trials = get_event_response(event_psth_dict,events)

        # event_trials = _all_trials
        projected_trials_by_event = [[project_pca(trial,Xa_trial_concatenated_pca,standardise=False)
                                      for trial in trials_by_event] for trials_by_event in event_trials]
        # projected_trials_by_event = [project_pca(trial, Xa_trial_averaged_pca, standardise=False)
        #                               for trial in Xa_trial_averaged]
        plot_pca_ts(projected_trials_by_event,events,window,n_components=n_comp_toplot,plot=[pca_ts_plot[0], pca_ts_plot[1][ii]])

        [[ax.axvline(t, ls='--', c='k') for t in np.arange(0,min([1,window[1]]),0.25)]
         for ax in pca_ts_plot[1][ii]]

        [[pca_scatter_plot[1][ci].scatter(np.array(proj)[:,comps[0],timepoints[-1]],
                                          np.array(proj)[:,comps[1],timepoints[-1]],
                                     c=f'C{ii}',label=pip,alpha=0.25,marker='o')
         for ti,proj in enumerate(projected_trials_by_event)] for ci, comps in enumerate([[0,1],[0,2],[1,2]])]
        [(pca_scatter_plot[1][ci].set_xlabel(f'PC{comps[0]}'), pca_scatter_plot[1][ci].set_ylabel(f'PC{comps[1]}'))
         for ci, comps in enumerate([[0,1],[0,2],[1,2]])]
        # pca_ts_plot[1][ii]
    # pca_ts_plot[0].set_layout_engine('constrained')
    pca_ts_plot[0].show()
    pca_ts_plot[0].savefig('pca_ts.svg')
    pca_scatter_plot[1][-1].legend()
    pca_scatter_plot[0].savefig('pca_scatter.svg')
    # pca_scatter_plot[1][-1].legend(ncols=len(rules))
    pca_scatter_plot[0].show()

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
    for ii,pip in enumerate(['A','B','C','D','base']):
        events = [e for e in event_psth_dict if pip in e]

        # event_trials = [np.array_split(event_psth_dict[e], event_psth_dict[e].shape[0], axis=0)
        #                 for e in events]
        event_trials = get_event_response(event_psth_dict, events)
        projected_trials_by_event = [[project_pca(trial, Xa_trial_averaged_pca, standardise=False)
                                      for trial in trials_by_event] for trials_by_event in event_trials]
        for ax in axes:
            for t, t_type in enumerate(projected_trials_by_event):
                proj_arr = np.array(t_type).mean(axis=0)
                # for every trial type, select the part of the component
                # which corresponds to that trial type:
                x = proj_arr[component_x]
                y = proj_arr[component_y]
                z = proj_arr[component_z]

                # apply some smoothing to the trajectories
                x = gaussian_filter1d(x, sigma=sigma)
                y = gaussian_filter1d(y, sigma=sigma)
                z = gaussian_filter1d(z, sigma=sigma)

                # use the mask to plot stimulus and pre/post stimulus separately
                z_stim = z.copy()
                z_stim[~stim_mask] = np.nan
                z_prepost = z.copy()
                z_prepost[stim_mask] = np.nan

                ax.plot(x, y, z_stim, c=f'C{ii}')
                ax.plot(x, y, z_prepost, c=f'C{ii}', ls=':')

                # plot dots at initial point
                ax.scatter(x[0], y[0], z[0], c=f'C{ii}', s=14,label=pip)

                # make the axes a bit cleaner
                style_3d_ax(ax)

        # specify the orientation of the 3d plot
        axes[0].view_init(elev=22, azim=30)
        # axes[0].set_title(f'{pip} 3D view')
        axes[1].view_init(elev=22, azim=110)
    axes[-1].legend()
    plt.tight_layout()

    fig.show()
    fig.savefig(f'{sessname}_pca_3d.svg')

    # for ii, pip in enumerate(['A', 'B', 'C', 'D']):
    pip = 'A'
    fig, axes = plt.subplots(ncols=2, figsize=[9, 4], subplot_kw={'projection': '3d'})
    events = [e for e in event_psth_dict if e.startswith(pip)]


    projected_trials_by_event = [[project_pca(trial, Xa_trial_averaged_pca, standardise=False)
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
        [[by_cond_by_event_trajs.update({f'{e}_{e_lbl}': get_event_trajs(event_psth_dict, e, list(range(n_pcs)),
                                                                         subset=e_subset)})
         for e, in [f'{ee}-0' for ee in 'ABCD']]
         for e_subset,e_lbl in zip(by_cond_trial_idxs,cond_lbls)]

    if 4 in sessions[sessname].td_df['Stage'].values:
        cond_lbls = ['normal', 'deviant']
        [[by_cond_by_event_trajs.update({f'{e}_{e_lbl}': get_event_trajs(event_psth_dict, e, list(range(n_pcs)))})
          for e, in [f'{ee}-{cond_i}' for ee in 'ABCD']]
         for cond_i, e_lbl in enumerate(cond_lbls)]

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
    pca_traj_plot[1][-1].legend()
    pca_traj_plot[0].show()
    pca_traj_plot[0].savefig(f'{sessname}_pca_trajs.svg')

    style_3d_ax(pca_traj_plot_3d[1])
    pca_traj_plot_3d[1].legend()
    pca_traj_plot_3d[0].show()
    pca_traj_plot_3d[0].savefig(f'{sessname}_pca_trajs_3d.svg')

    if 3 in sessions[sessname].td_df['Stage'].values:
        [by_cond_by_event_trajs.update({f'full_{e_lbl}': get_event_trajs(event_psth_dict, e, list(range(n_pcs)),
                                                                         subset=e_subset)})
         for e_subset, e_lbl in zip(by_cond_trial_idxs, cond_lbls)]

    if 4 in sessions[sessname].td_df['Stage'].values:
        [by_cond_by_event_trajs.update({e: get_event_trajs(event_psth_dict, e, list(range(n_pcs)))})
          for e in ['full_normal','full_deviant']]

        

    projected_pattern_response = [project_pca(trial, Xa_trial_averaged_pca, standardise=False)
                                  for trial in pattern_response]
    projected_pattern_pc123 = [format_tractories(projected_pattern_response, list(range(10)), mean_axis=None,
                                                subset=cond_subset-1)
                               for cond_subset in by_cond_trial_idxs]
    projection_sim_ts = [compare_pip_sims_2way([np.array(projected_pattern_pc123[0]).transpose((1,0,2)),
                                                np.array(projected_pattern_pc123[1]).transpose((1,0,2))],
                                               t=t,mean_flag=True)
                         for t in range(projected_pattern_pc123[0][0].shape[-1])]  # len = n time points
    mean_comped_sims = np.array([[np.squeeze(pip_sims)[:, 0, 1] for pip_sims in
                        np.array_split(t_sim[0], (len(cond_lbls)))]
                        for t_sim in projection_sim_ts])
    mean_cross_comped_sims = np.array([np.squeeze(t_sim[1][:, 0, 1]) for t_sim in projection_sim_ts])

    full_pattern_xser = np.linspace(-0.1,1, projected_pattern_pc123[0][0].shape[1])
    rare_vs_freq_sim_ts_plot = plt.subplots()
    [rare_vs_freq_sim_ts_plot[1].plot(full_pattern_xser,mean_comped_sims[:,cond_i].mean(axis=1),label=cond_lbl)
     for cond_i, cond_lbl in enumerate(cond_lbls)]
    rare_vs_freq_sim_ts_plot[1].plot(full_pattern_xser,mean_cross_comped_sims.mean(axis=1),label='rare vs freq',c='k')
    rare_vs_freq_sim_ts_plot[1].set_ylabel('cosine similarity')
    rare_vs_freq_sim_ts_plot[1].set_xlabel('Time (s)')
    rare_vs_freq_sim_ts_plot[1].set_title('Rare vs Frequent response similarity')
    rare_vs_freq_sim_ts_plot[1].legend()
    rare_vs_freq_sim_ts_plot[0].show()
    rare_vs_freq_sim_ts_plot[0].savefig(f'{sessname}_rare_vs_freq_sim_ts.svg')

    projected_pattern_pc123 = [format_tractories(projected_pattern_response, list(range(10)), mean_axis=0,
                                                subset=cond_subset-1)
                               for cond_subset in by_cond_trial_idxs]
    projection_distance_ts = [euclidean(np.array(projected_pattern_pc123[0])[:, t],
                                        np.array(projected_pattern_pc123[1])[:, t])
                              for t in range(projected_pattern_pc123[0][0].shape[0])]

    pca_euc_dist_plot = plt.subplots()
    pca_euc_dist_plot[1].plot(full_pattern_xser,projection_distance_ts)
    pca_euc_dist_plot[1].set_ylabel('Euclidean distance')
    pca_euc_dist_plot[1].set_xlabel('Time (s)')
    pca_euc_dist_plot[1].set_title('Pattern response')
    pca_euc_dist_plot[0].show()
    pca_euc_dist_plot[0].savefig(f'{sessname}_pca_euc_dist.svg')



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