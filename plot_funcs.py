import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from tqdm import tqdm
import scipy


def plot_spike_time_raster(spike_time_dict: dict, ax=None, **pltkwargs):
    if not ax:
        fig, ax = plt.subplots()
    assert isinstance(ax, plt.Axes)
    for cluster_id in tqdm(spike_time_dict, desc='plotting spike times for event', total=len(spike_time_dict),
                           disable=True):
        ax.scatter(spike_time_dict[cluster_id], [cluster_id] * len(spike_time_dict[cluster_id]), **pltkwargs)
        ax.invert_xaxis()


def unique_legend(plotfig:(plt.figure().figure,list,tuple),**leg_kwargs):
    if isinstance(plotfig,(tuple,list)):
        if isinstance(plotfig[1],np.ndarray):
            plotaxes2use = plotfig[1].flatten()
        elif isinstance(plotfig[1], dict):
            plotaxes2use = plotfig[1].values()
        elif isinstance(plotfig[1], plt.Axes):
            plotaxes2use = [plotfig[1]]
        else:
            print('wrong figure used, returning none')
            plotaxes2use = None
    elif isinstance(plotfig,np.ndarray):
        plotaxes2use = plotfig.flatten()
    elif isinstance(plotfig[1],dict):
        plotaxes2use = plotfig[1].values()
    elif isinstance(plotfig,plt.Axes):
        plotaxes2use = [plotfig]
    else:
        plotaxes2use = None
        print('wrong figure used, returning none')
    for axis in plotaxes2use:
        handle, label = axis.get_legend_handles_labels()
        axis.legend(pd.Series(handle).unique(), pd.Series(label).unique(),**leg_kwargs)


def plot_2d_array_with_subplots(array_2d: np.ndarray, cmap='viridis', cbar_width=0.1, cbar_height=0.8,
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
    plot_cbar = im_kwargs.get('plot_cbar', True)
    im_kwargs.pop('plot_cbar', None)

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
        cax = divider.append_axes('right', size='7.5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, fraction=cbar_width, aspect=cbar_height, )
    else:
        cbar = None
    # Show the plot
    # plt.show()
    return fig, ax, cbar


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


def plot_psth(psth_rate_mat, event_lbl, window, title='', cbar_label=None, cmap='Reds', **im_kwargs):
    if not cbar_label:
        cbar_label = 'firing rate (Hz)'
    if not im_kwargs.get('aspect'):
        im_kwargs['aspect'] = 0.1
    fig, ax, cbar = plot_2d_array_with_subplots(psth_rate_mat, cmap=cmap,
                                                extent=[window[0], window[1], psth_rate_mat.shape[0], 0],
                                                cbar_height=50, **im_kwargs)
    ax.set_ylabel('unit number', fontsize=14)
    ax.set_xlabel(f'time from {event_lbl} onset (s)', fontsize=14)
    ax.set_title(title)
    if cbar:
        cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=15)
    for t in np.arange(0, 1, 1):
        ax.axvline(t, ls='--', c='white', lw=1)

    return fig, ax, cbar


def plot_sorted_psth(responses_by_sess, pip, sort_pip, window, sort_window, plot=None, sessname_filter=None,
                     im_kwargs=None, plot_ts=True, plot_cv=True, plot_window=None):
    if isinstance(sessname_filter, str):
        sessname_filter = [sessname_filter]
    # [print(sess) for sess in responses_by_sess.keys() if sessname_filter in sess]
    # print('new sorting')
    print(f'{sessname_filter = }')
    responses_4_sorting = {
        e: np.concatenate([responses_by_sess[sessname][e][0::2].mean(axis=0) if plot_cv else
                           responses_by_sess[sessname][e][:].mean(axis=0)
                           for sessname in responses_by_sess
                           if (any([e in sessname for e in sessname_filter]) if sessname_filter else True)])
        for e in [pip, sort_pip]}

    responses_4_plotting = {
        e: np.concatenate([responses_by_sess[sessname][e][1::2].mean(axis=0) if plot_cv else
                           responses_by_sess[sessname][e][:].mean(axis=0)
                           for sessname in responses_by_sess
                           if (any([e in sessname for e in sessname_filter]) if sessname_filter else True)])
        for e in [pip, sort_pip]}

    # print(responses_4_plotting.keys())
    if plot is None:
        psth_plot = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [1, 9], 'hspace': 0.1})
    else:
        psth_plot = plot

    resp_mat = responses_4_plotting[pip]
    if resp_mat.ndim == 3:
        resp_mat = resp_mat.mean(axis=0)
    x_ser = np.round(np.linspace(*window, resp_mat.shape[-1]), 2)
    sort_idxs = [np.where(x_ser == t)[0][0] for t in sort_window]

    if plot_window is not None:
        plot_t_idxs = [np.where(x_ser == t)[0][0] for t in plot_window]
    else:
        plot_t_idxs = [0, x_ser.shape[0]]
        plot_window = [x_ser[plot_t_idxs[0]], x_ser[plot_t_idxs[1]]]

    max_by_row = np.argmax(responses_4_sorting[sort_pip][:, sort_idxs[0]:sort_idxs[1]], axis=1)
    resp_mat_sorted = resp_mat[max_by_row.argsort()][:, plot_t_idxs[0]:plot_t_idxs[1]]
    _,_,cbar = plot_psth(resp_mat_sorted, pip,
                         plot_window, aspect=0.1, plot=(psth_plot[0], psth_plot[1][1] if plot_ts else psth_plot[1]),
                         **im_kwargs if im_kwargs else {})
    psth_plot[1][1].set_yticks([resp_mat_sorted.shape[0]])
    psth_plot[1][1].set_yticklabels([resp_mat_sorted.shape[0]])

    if plot_ts:
        resp_mat_4_ts = resp_mat[:, plot_t_idxs[0]:plot_t_idxs[1]]
        assert len(psth_plot[1]) > 1, 'need psth plot with 2 axes'
        psth_plot[1][0].plot(x_ser[plot_t_idxs[0]:plot_t_idxs[1]], resp_mat_4_ts.mean(axis=0), color='k')
        psth_plot[1][0].fill_between(x_ser[plot_t_idxs[0]:plot_t_idxs[1]],
                                     resp_mat_4_ts.mean(axis=0) - resp_mat_4_ts.std(axis=0) / np.sqrt(resp_mat_4_ts.shape[0]),
                                     resp_mat_4_ts.mean(axis=0) + resp_mat_4_ts.std(axis=0) / np.sqrt(resp_mat_4_ts.shape[0]),
                                     color='k', alpha=0.1)
        # fix positions
        axs_bbox = [ax.get_position() for ax in psth_plot[1]]
        og_ts_bbox = [axs_bbox[0].x0, axs_bbox[0].y0, axs_bbox[0].width, axs_bbox[0].height]
        og_ts_bbox[2] = axs_bbox[0].width * 0.8
        psth_plot[1][0].set_position(og_ts_bbox)

    print(f'n units plotted for {pip}: {resp_mat.shape[0]}')
    if plot is None:
        return psth_plot[0],psth_plot[1],[cbar]


def get_sorted_psth_matrix(responses_by_sess, pip, sort_pip, window, sort_window,
                           sessname_filter=None, plot_cv=True, **kwargs):
    if isinstance(sessname_filter, str):
        sessname_filter = [sessname_filter]

    # Collect responses for sorting
    responses_4_sorting = {
        e: np.concatenate([
            responses_by_sess[sessname][e][0::2].mean(axis=0) if plot_cv else
            responses_by_sess[sessname][e][:].mean(axis=0)
            for sessname in responses_by_sess
            if (any(sub in sessname for sub in sessname_filter) if sessname_filter else True)
        ])
        for e in [pip, sort_pip]
    }

    # Collect responses for plotting
    responses_4_plotting = {
        e: np.concatenate([
            responses_by_sess[sessname][e][1::2].mean(axis=0) if plot_cv else
            responses_by_sess[sessname][e][:].mean(axis=0)
            for sessname in responses_by_sess
            if (any(sub in sessname for sub in sessname_filter) if sessname_filter else True)
        ])
        for e in [pip, sort_pip]
    }

    resp_mat = responses_4_plotting[pip]
    if resp_mat.ndim == 3:
        resp_mat = resp_mat.mean(axis=0)

    x_ser = np.round(np.linspace(*window, resp_mat.shape[-1]), 2)
    sort_idxs = [np.where(x_ser == t)[0][0] for t in sort_window]

    max_by_row = np.argmax(responses_4_sorting[sort_pip][:, sort_idxs[0]:sort_idxs[1]], axis=1)
    sorted_matrix = resp_mat[max_by_row.argsort()]

    return sorted_matrix, resp_mat, max_by_row, x_ser


def plot_sorted_psth_matrix(sorted_matrix, x_ser, pip, plot_window=None,
                            im_kwargs=None, plot_ts=True, existing_plot=None, **kwargs):

    if existing_plot is None:
        fig, axes = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [1, 9], 'hspace': 0.1})
    else:
        fig, axes = existing_plot

    if plot_window is not None:
        plot_t_idxs = [np.where(x_ser == t)[0][0] for t in plot_window]
    else:
        plot_t_idxs = [0, x_ser.shape[0]]
        plot_window = [x_ser[plot_t_idxs[0]], x_ser[plot_t_idxs[1]]]

    resp_to_plot = sorted_matrix[:, plot_t_idxs[0]:plot_t_idxs[1]]
    _, _, cbar = plot_psth(resp_to_plot, pip, plot_window, aspect=0.1,
                           plot=(fig, axes[1] if plot_ts else axes),
                           **(im_kwargs if im_kwargs else {}))

    axes[1].set_yticks([resp_to_plot.shape[0]])
    axes[1].set_yticklabels([resp_to_plot.shape[0]])

    if plot_ts:
        mean_ts = resp_to_plot.mean(axis=0)
        std_ts = resp_to_plot.std(axis=0) / np.sqrt(resp_to_plot.shape[0])
        axes[0].plot(x_ser[plot_t_idxs[0]:plot_t_idxs[1]], mean_ts, color='k')
        axes[0].fill_between(x_ser[plot_t_idxs[0]:plot_t_idxs[1]],
                             mean_ts - std_ts, mean_ts + std_ts,
                             color='k', alpha=0.1)

        # Adjust top axis width
        axs_bbox = [ax.get_position() for ax in axes]
        new_ts_bbox = [axs_bbox[0].x0, axs_bbox[0].y0, axs_bbox[0].width * 0.8, axs_bbox[0].height]
        axes[0].set_position(new_ts_bbox)

    print(f'n units plotted for {pip}: {sorted_matrix.shape[0]}')
    return fig, axes, cbar


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
    def mean_confidence_interval(data, confidence=0.95, var_func=np.std):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.nanmean(a), var_func(a)  # a.std()
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return m, m - h, m + h

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


def format_axis(ax, **kwargs):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_color('k')
    ax.spines['left'].set_color('k')
    ax.spines['bottom'].set_zorder(1)

    ax.locator_params(axis='both', nbins=kwargs.get('nbins', 4))
    ax.set_xlabel(kwargs.get('xlabel', ''))
    ax.set_ylabel(kwargs.get('ylabel', ''))

    if kwargs.get('ylim', None):
        ax.set_ylim(kwargs.get('ylim'))
    if kwargs.get('xlim', None):
        ax.set_xlim(kwargs.get('xlim'))

    [ax.axhline(e, color='k', ls='--',lw=kwargs.get('lw',0.5)) for e in kwargs.get('hlines', []) if kwargs.get('hlines', None)]
    [ax.axvline(e, color='k', ls='--',lw=kwargs.get('lw',0.5)) for e in kwargs.get('vlines', []) if kwargs.get('vlines', None)]

    [ax.axvspan(*e, color='k', alpha=0.1) for e in kwargs.get('vspan', [[]]) if kwargs.get('vspan', None)]
    [ax.axhspan(*e, color='k', alpha=0.1) for e in kwargs.get('hspan', [[]]) if kwargs.get('hspan', None)]


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


def add_x_scale_bar(ax, size, label=None, location='lower right', **kwargs):
    import matplotlib.font_manager as fm
    """
    Add an x-axis scale bar to a matplotlib Axes.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The Axes object to which the scale bar will be added.
    size : float
        Length of the scale bar in data units.
    label : str, optional
        Label text shown below the scale bar. Defaults to f"{size}".
    location : str, optional
        Location of the scale bar (default: 'lower right').
    **kwargs :
        Additional keyword arguments passed to AnchoredSizeBar, except
        for 'transform', 'size', and 'label' which are set internally.
    """

    # Prevent accidental overwriting of required internal arguments
    disallowed_keys = {'transform', 'size', 'label'}
    clean_kwargs = {k: v for k, v in kwargs.items() if k not in disallowed_keys}

    # Use default font if none provided
    if 'fontproperties' not in clean_kwargs:
        clean_kwargs['fontproperties'] = fm.FontProperties(size=10)

    scalebar = AnchoredSizeBar(
        transform=ax.transData,
        size=size,
        label=label if label is not None else f"{size}",
        loc=location,
        **clean_kwargs
    )

    ax.add_artist(scalebar)


def plot_shaded_error_ts(ax,x_ser,mean_ts,sem_ts,**kwargs):
    x_ser, mean_ts, sem_ts = [np.array(e).astype(float) for e in [x_ser, mean_ts, sem_ts]]
    upper, lower = mean_ts+sem_ts, mean_ts-sem_ts
    ax.fill_between(x_ser,lower,upper, **kwargs)

def choose_hist_rule(x, *, discrete_hint=None):
    """
    Decide a histogram bin rule: 'fd' or another ('sturges', 'scott', 'doane', 'sqrt').

    Heuristics:
      - very small n  -> 'sturges'
      - many repeats / likely discrete -> 'sqrt'
      - skewed or heavy-tailed -> 'fd'
      - otherwise -> 'scott' (near-normal, continuous)
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return 'sturges'

    # basic stats
    mu = x.mean()
    sd = x.std()
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    # sample skewness (population version; stable enough for this decision)
    sk = ((x - mu) ** 3).mean() / ((sd + 1e-12) ** 3)

    # heuristics
    uniq_ratio = np.unique(x).size / n
    if n < 50:
        return 'sturges'

    # user can hint discrete data (e.g., integer counts)
    if discrete_hint is True or uniq_ratio < 0.2:
        return 'sqrt'

    # heavy tails / skew: compare σ to expected σ from IQR for Normal (IQR ≈ 1.349σ)
    sigma_from_iqr = (iqr / 1.349) if iqr > 0 else 0.0
    heavy_tails = (sigma_from_iqr > 0 and sd > 1.6 * sigma_from_iqr)
    skewed = abs(sk) > 0.5

    if (skewed or heavy_tails) and iqr > 0:
        return 'fd'

    # mild skew but not extreme: Doane copes better than Sturges
    if abs(sk) > 0.25:
        return 'doane'

    # near-normal, continuous
    return 'scott'