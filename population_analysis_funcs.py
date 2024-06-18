import matplotlib.pyplot as plt
import neo
import numpy as np
import pandas as pd
from matplotlib import animation
from ephys_analysis_funcs import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage import gaussian_filter1d
from scipy.stats import bootstrap
import argparse
import yaml
import platform
from elephant.gpfa import GPFA
from elephant.conversion import BinnedSpikeTrain
import neo
from sklearn.model_selection import cross_val_score
from os import cpu_count
from sythesise_spikes import integrated_oscillator, random_projection,generate_spiketrains
import quantities as pq
from IPython.display import HTML
from IPython import display
from neural_similarity_funcs import compare_pip_sims_2way,plot_similarity_mat


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
            # plot_ts_var(x_ser,projected_trials_comp,f'C{ei}',ax)

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
    sess_order_dict = {e: ee for e, ee in zip('abc',['pre','main','post'])}
    sess_topology_path = ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology.csv')
    session_topology = pd.read_csv(sess_topology_path)
    sessname = 'DO81_240529b' if args.sessname is None else args.sessname
    name,date,order = sessname.split('_')[0],int(sessname.split('_')[1][:-1]),sess_order_dict[sessname[-1]]
    sess_info = session_topology.query('name==@name and date==@date and sess_order==@order').iloc[0]

    pkldir = ceph_dir / posix_from_win(args.pkldir)
    assert pkldir.is_dir()
    sess_pkls = list(pkldir.glob(f'{sessname}.pkl'))
    with open(sess_pkls[0], 'rb') as f:
        sessions[sessname]: Session = pickle.load(f)
    # sessions[sessname]: Session = pd.read_pickle(sess_pkls[0])

    window = [-0.25,0.25]
    # window = [0,2]
    for e in list(sessions[sessname].sound_event_dict.keys()):
        sessions[sessname].spike_obj.get_event_spikes(sessions[sessname].sound_event_dict[e].times,e,window,
                                                      get_spike_matrix=False)

    # sessions[sessname].init_spike_obj(spike_times_path, spike_cluster_path, start_time, parent_dir=spike_dir)

    event_psth_dict = {}
    event_spikes_dict = {}
    # events = ['X','A','B','C','D','base']
    events = [f'D-{i}' for i in range(6)]
    # get event spike trains all trials
    for event in events:
        event_spikes = [sessions[sessname].spike_obj.event_cluster_spike_times[e]
                        for e in sessions[sessname].spike_obj.event_cluster_spike_times.keys()
                        if event in e]
        event_spikes_dict[event] = [[SpikeTrain(ee,t_start=window[0],t_stop=window[1],units=s) for ee in e.values()]
                                    for e in event_spikes]

    # cv_results = cross_val_gpfa(event_spikes_dict['A-3'],x_dims=np.arange(1,30).astype(int))
    # cv_plot = plt.subplots()
    # cv_plot[1].plot(np.arange(1,30),cv_results)
    # cv_plot[1].set_xlabel('latent dimensionality')
    # cv_plot[1].set_ylabel('log likelihood')
    # cv_plot[1].set_title('cross validation')
    # cv_plot[0].show()
    # results = run_gpfa(event_spikes_dict['A-0'],latent_dimensionality=14)
    #
    # fig, ax = plt.subplots()
    # [ax.plot(e[0],e[1],c='k',lw=2) for e in results[1]]
    # average_trajectory = np.mean(results[1],axis=0)
    # ax.plot(average_trajectory[0], average_trajectory[1], c='r', lw=5)
    # fig.show()
    # fig,ax = plt.subplots(14,figsize=[10,15])
    # x_ser = np.linspace(window[0], window[1], average_trajectory.shape[-1])
    # [ax[i].plot(x_ser,e,c='k',lw=2) for i,e in enumerate(average_trajectory)]
    # fig.show()


    # events = events[1:]

    # set parameters for the integration of the harmonic oscillator
    timestep = 1 * pq.ms
    trial_duration = 2 * pq.s
    num_steps = int((trial_duration.rescale('ms') / timestep).magnitude)

    # set parameters for spike train generation
    max_rate = 70 * pq.Hz
    np.random.seed(42)  # for visualization purposes, we want to get identical spike trains at any run

    # specify data size
    num_trials = 24
    num_spiketrains = 50

    # generate a low-dimensional trajectory
    times_oscillator, oscillator_trajectory_2dim = integrated_oscillator(
        timestep.magnitude, num_steps=num_steps, x0=0, y0=1)
    times_oscillator = (times_oscillator * timestep.units).rescale('s')

    # random projection to high-dimensional space
    oscillator_trajectory_Ndim = random_projection(
        oscillator_trajectory_2dim, embedding_dimension=num_spiketrains)

    # convert to instantaneous rate for Poisson process
    normed_traj = oscillator_trajectory_Ndim / oscillator_trajectory_Ndim.max()
    instantaneous_rates_oscillator = np.power(max_rate.magnitude, normed_traj)

    # generate spike trains
    spiketrains_oscillator = generate_spiketrains(
        instantaneous_rates_oscillator, num_trials, timestep)

    trial_oscillator = [instantaneous_rate(trial, sampling_period=10*ms,kernel=GaussianKernel(40*ms)).as_array().T
                        for trial in spiketrains_oscillator]

    all_events = [e for e in sessions[sessname].sound_event_dict.keys() if e[0] in ['A','B','C','D']]
    event_psth_dict = {e: get_predictor_from_psth(sessions[sessname], e, [-2,3], window,mean=None,
                                                  use_unit_zscore=True, use_iti_zscore=False)
                       for e in all_events}
    _all_trials = [np.array_split(event_psth_dict[e], event_psth_dict[e].shape[0], axis=0)
                   for e in all_events]
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
    for ii,pip in enumerate(['A','B','C','D']):
        pca_scatter_plot = plt.subplots(ncols=3,figsize=[15,5])
        events = [f'{pip}-{i}' for i in range(6)]
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

        event_trials = [np.array_split(event_psth_dict[e], event_psth_dict[e].shape[0], axis=0)
                        for e in events]
        event_trials = [[np.squeeze(ee) for ee in e] for e in event_trials]

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
                                     c=f'C{ti}',label=rules[ti],alpha=0.25,marker='o')
         for ti,proj in enumerate(projected_trials_by_event)] for ci, comps in enumerate([[0,1],[0,2],[1,2]])]
        [(pca_scatter_plot[1][ci].set_xlabel(f'PC{comps[0]}'), pca_scatter_plot[1][ci].set_ylabel(f'PC{comps[1]}'))
         for ci, comps in enumerate([[0,1],[0,2],[1,2]])]
        # pca_ts_plot[1][ii]
    # pca_ts_plot[0].set_layout_engine('constrained')
    pca_ts_plot[0].show()
    pca_ts_plot[0].savefig('pca_ts.svg')
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


    sigma = 3  # smoothing amount

    # set up a figure with two 3d subplots, so we can have two different views

    for ii,pip in enumerate(['A','B','C','D']):
        fig,axes = plt.subplots(ncols=2, figsize=[9, 4],subplot_kw={'projection': '3d'})
        events = [f'{pip}-{i}' for i in range(6)]

        event_trials = [np.array_split(event_psth_dict[e], event_psth_dict[e].shape[0], axis=0)
                        for e in events]
        event_trials = [[np.squeeze(ee) for ee in e] for e in event_trials]
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

                ax.plot(x, y, z_stim, c=f'C{t}')
                ax.plot(x, y, z_prepost, c=f'C{t}', ls=':')

                # plot dots at initial point
                ax.scatter(x[0], y[0], z[0], c=f'C{t}', s=14)

                # make the axes a bit cleaner
                style_3d_ax(ax)

        # specify the orientation of the 3d plot
        axes[0].view_init(elev=22, azim=30)
        axes[0].set_title(f'{pip} 3D view')
        axes[1].view_init(elev=22, azim=110)
        plt.tight_layout()

        fig.show()


def animate(i):
    ax.clear()
    style_3d_ax(ax)
    ax.view_init(elev=22, azim=30)
    print(i)
    for t, t_type in enumerate(projected_trials_by_event):
        proj_arr = np.array(t_type).mean(axis=0)
        # for every trial type, select the part of the component
        # which corresponds to that trial type:
        x = proj_arr[component_x][0:i]
        y = proj_arr[component_y][0:i]
        z = proj_arr[component_z][0:i]
        print(x.shape)
        # apply some smoothing to the trajectories
        # x = gaussian_filter1d(x, sigma=sigma)
        # y = gaussian_filter1d(y, sigma=sigma)
        # z = gaussian_filter1d(z, sigma=sigma)

        stim_mask = np.logical_and(x_ser >= 0, x_ser <= 0.25)
        z_stim = z.copy()
        # z_stim[~stim_mask] = np.nan
        # z_prepost = z.copy()
        # z_prepost[stim_mask] = np.nan

        ax.plot(x, y, z, c=f'C{t}')
        # ax.plot(x, y, z_prepost, c=f'C{t}', ls=':')

    # ax.set_xlim((-12, 12))
    # ax.set_ylim((-12, 12))
    # ax.set_zlim((-13, 13))
    ax.view_init(elev=22, azim=30)

    return []


# for ii, pip in enumerate(['A', 'B', 'C', 'D']):
pip = 'A'
fig, axes = plt.subplots(ncols=2, figsize=[9, 4], subplot_kw={'projection': '3d'})
events = [f'{pip}-{i}' for i in range(6)]

event_trials = [np.array_split(event_psth_dict[e], event_psth_dict[e].shape[0], axis=0)
                for e in events]
event_trials = [[np.squeeze(ee) for ee in e] for e in event_trials]
projected_trials_by_event = [[project_pca(trial, Xa_trial_averaged_pca, standardise=False)
                              for trial in trials_by_event] for trials_by_event in event_trials]

anim = animation.FuncAnimation(fig, animate, frames=x_ser.shape[0], interval=50,blit=True)
video = anim.to_html5_video()
html = display.HTML(video)
display.display(html)


for pip in ['A', 'B', 'C', 'D']:
    compared_pips = compare_pip_sims_2way([event_psth_dict[f'{pip}-0'], event_psth_dict[f'{pip}-1']])

    compared_pips_plot = plt.subplots()
    mean_comped_sims = [np.squeeze(pip_sims)[:,0,1] for pip_sims in np.array_split(compared_pips[0], 2)]
    mean_comped_sims.append(np.squeeze(compared_pips[1][:,0,1]))
    compared_pips_plot[1].boxplot(mean_comped_sims, labels=[f'{pip}-0 self', f'{pip}-1 self', f'{pip}-0 vs {pip}-1'], )
    compared_pips_plot[1].set_ylim([0, 1])
    compared_pips_plot[0].show()
    compared_pips_plot[0].savefig(f'{pip}_compared_{sessname}.svg')
