import numpy as np
from mne.stats import permutation_cluster_test
import matplotlib.pyplot as plt

from plot_funcs import choose_hist_rule


def cluster_analysis(X1, X2, n_permutations=1000,p_alpha=0.05, **kwargs):
    """
    Perform cluster-based permutation test on two conditions.

    Returns
    -------
    clusters : list of tuples
        All clusters (tuple of slices).
    cluster_p_values : list of float
        p-value of each cluster from permutation test.
    cluster_mass : list of float
        Sum of t-values per cluster (raw cluster-mass).
    normalized_mass : list of float
        Mean t-value per cluster (length-normalized).
    """
    # return None
    X = [X2, X1]
    tfce_params = dict(
    E=1,    # extent exponent (sensitivity to cluster width)
    H=1.0,      # height exponent (sensitivity to effect size)
    start=0.2,   # start threshold (no threshold)
    step=0.1,)   # step size (smaller = more accurate)

    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        X, n_permutations=n_permutations, tail=1, n_jobs=1, seed=42,threshold=3,
        verbose=False,
    )
    # print(f'--- H0: {H0} ---')

    # T_obs, clusters, cluster_p_values, H0 = split_clusters(T_obs, clusters, cluster_p_values, H0)
    # clusters = [c[0] for c in clusters]

    cluster_mass = [T_obs[c].sum() for c in clusters]
    cluster_mass = [e if e != np.nan else 0 for e in cluster_mass]
    normalized_mass = [T_obs[c].mean() for c in clusters]

    return clusters, cluster_p_values, cluster_mass, normalized_mass


def plot_clusters(x_ser, clusters, cluster_p_values,plot,p_alpha=0.05,plot_y=0,plot_kwargs=None):
    """
    Plot mean responses with significant clusters shaded.
    """

    fig, ax = plot

    if plot_kwargs is None:
        plot_kwargs = {}

    for i, c in enumerate(clusters):
        if cluster_p_values[i] < p_alpha:
            ax.plot([x_ser[c[0]], x_ser[c[-1]]],[plot_y,plot_y], **plot_kwargs)

    return fig, ax


def plot_cluster_stats(cluster_mass, shuff_cluster_mass,plot=None, plot_kwargs=None,
                       plot_raw=True, plot_normed=True):

    if plot is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot

    if plot_kwargs is None:
        plot_kwargs = {}

    """
        Plot total cluster mass and total normalized cluster mass (sums across all clusters).
        """
    total_mass = np.sum(cluster_mass)
    total_shuff_mass = [np.sum(shuff) for shuff in shuff_cluster_mass]
    # plot 1% to 99% range
    total_shuff_mass = [m for m in total_shuff_mass if m >= np.percentile(total_shuff_mass,1) 
                        and m <= np.percentile(total_shuff_mass,99)]

    ax.hist(total_shuff_mass,bins=choose_hist_rule(total_shuff_mass),fc='gray')
    ax.axvline(total_mass,color='goldenrod',ls='-')
    ax.set_ylabel("Frequency",fontdict={'size':5})
    ax.set_xlabel("Total mass",fontdict={'size':5})
    ax.yaxis.labelpad = 1
    ax.xaxis.labelpad = 2
    ax.tick_params(axis='both', which='both', pad=2, labelsize=5,)
    # set symlog scale if needed
    if abs(total_mass) / np.min(total_shuff_mass) > 20:
        ax.set_xscale('symlog', linthresh=1)
    else:
        ax.set_xscale('linear')
    ax.set_xscale('symlog', linthresh=abs(np.max(total_shuff_mass)-np.min(total_shuff_mass))*0.1+0.1)
    # ax.set_xscale('linear',)

    ax.set_title("",fontdict={'size':5})
    ax.locator_params(axis='y', nbins=3)
    # ax.locator_params(axis='x', nbins=4)
    fig.tight_layout()
    fig.show()

    return fig, ax


def split_clusters(T_obs, clusters, cluster_p_values, H0=None,
                   depth_fraction=0.5, min_gap=1):
    """
    Split clusters (arrays of indices) if valleys appear inside them.
    Returns results in the exact same format as the input:
        T_obs, clusters, cluster_p_values, H0

    Parameters
    ----------
    T_obs : ndarray, shape (n_timepoints,)
        Observed t-values.
    clusters : list of np.ndarray
        Each cluster is an array of indices (sorted, contiguous).
    cluster_p_values : ndarray
        P-values corresponding to clusters.
    H0 : ndarray | None
        Permutation null distribution (unchanged).
    depth_fraction : float
        Split when |t| dips below (peak * depth_fraction) inside a cluster.
    min_gap : int
        Minimum valley length (in samples) to split on.

    Returns
    -------
    T_obs : ndarray
        Same as input.
    new_clusters : list of np.ndarray
        Same format as input, but with splits applied.
    new_cluster_p_values : ndarray
        Same length as new_clusters, children inherit parent's p-value.
    H0 : ndarray | None
        Same as input.
    """
    new_clusters = []
    new_pvals = []

    for idx, pval in zip(clusters, cluster_p_values):
        # ensure indices are sorted
        idx = np.sort(np.unique(idx))
        if idx.size == 0:
            continue

        tvals = T_obs[idx]
        peak = np.nanmax(np.abs(tvals))
        if peak == 0:  # nothing to split
            new_clusters.append(idx)
            new_pvals.append(pval)
            continue

        valley_threshold = peak * depth_fraction
        below = np.where(np.abs(tvals) <= valley_threshold)[0]

        if below.size == 0:
            # no valley → keep cluster as is
            new_clusters.append(idx)
            new_pvals.append(pval)
            continue

        # contiguous runs of below-threshold points
        runs = np.split(below, np.where(np.diff(below) > 1)[0] + 1)

        cut_points = []
        for run in runs:
            if run.size >= min_gap:
                cut_rel = run[-1] + 1
                if cut_rel < idx.size:  # stay inside cluster
                    cut_points.append(idx[cut_rel])

        if not cut_points:
            new_clusters.append(idx)
            new_pvals.append(pval)
            continue

        # split cluster at cut_points
        split_idx = [idx[0]] + cut_points + [idx[-1] + 1]
        for a, b in zip(split_idx[:-1], split_idx[1:]):
            sub = idx[(idx >= a) & (idx < b)]
            if sub.size > 0:
                new_clusters.append(sub)
                new_pvals.append(pval)

    return T_obs, new_clusters, np.array(new_pvals), H0
