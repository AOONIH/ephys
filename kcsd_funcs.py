import numpy as np
from kcsd import KCSD2D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import filtfilt, butter
import argparse
import yaml
import platform
from pathlib import Path

from datetime import datetime

from postprocessing_utils import get_sorting_dirs, get_sorting_objs
from plot_funcs import plot_2d_array_with_subplots

plt.close('all')


def set_axis(ax, x, y, letter=None):
    ax.text(
        x,
        y,
        letter,
        fontsize=25,
        weight='bold',
        transform=ax.transAxes)
    return ax

plt.rcParams.update({
    'xtick.labelsize': 15,
    'xtick.major.size': 10,
    'ytick.labelsize': 15,
    'ytick.major.size': 10,
    'font.size': 12,
    'axes.labelsize': 15,
    'axes.titlesize': 20,
    'axes.titlepad' : 30,
    'legend.fontsize': 15,
    # 'figure.subplot.wspace': 0.4,
    # 'figure.subplot.hspace': 0.4,
    # 'figure.subplot.left': 0.1,
})


# %%
def make_plot_spacetime(ax, xx, yy, zz, Fs, title='True CSD', cmap=cm.bwr_r, ymin=0, ymax=10000, ylabel=True):
    im = ax.imshow(zz, extent=[0, zz.shape[1] / Fs * 1000, -0, 200], aspect='auto',
                   vmax=1 * zz.max(), vmin=-1 * zz.max(), cmap=cmap)
    ax.set_xlabel('Time (ms)')
    if ylabel:
        ax.set_ylabel('Y ($\mu$m)')
    if 'Pot' in title: ax.set_ylabel('Y ($\mu$m)')
    ax.set_title(title)
    ticks = np.linspace(-zz.max(), zz.max(), 3, endpoint=True)
    if 'CSD' in title:
        plt.colorbar(im, orientation='horizontal', format='%.2f', ticks=ticks)
    else:
        plt.colorbar(im, orientation='horizontal', format='%.1f', ticks=ticks)
        # plt.gca().invert_yaxis()


def make_plot(ax,fig, xx, yy, zz, title='True CSD', cmap=cm.bwr):
    ax.set_aspect('auto')
    levels = np.linspace(zz.min(), -zz.min(), 61)
    im = ax.contourf(xx, yy, zz, levels=levels, cmap=cmap)
    ax.set_xlabel('X ($\mu$m)')
    ax.set_ylabel('Y ($\mu$m)')
    ax.set_title(title)
    divider = make_axes_locatable(ax)

    cax = divider.append_axes('right', size='7.5%', pad=0.1)
    cbar = fig.colorbar(im, cax=cax, fraction=0.1, aspect=1, )
    # if 'CSD' in title:
    #     plt.colorbar(im, orientation='vertical', format='%.2f', ticks=[-0.02, 0, 0.02])
    # else:
    #     plt.colorbar(im, orientation='vertical', format='%.1f', ticks=[-0.6, 0, 0.6])
    # ax.scatter(ele_pos[:, 0],
    #             (ele_pos[:, 1]),
    #             s=0.8, color='black')
    # plt.gca().invert_yaxis()
    return ax


def eles_to_ycoord(eles):
    y_coords = []
    for ii in range(192):
        y_coords.append(ii * 20)
        y_coords.append(ii * 20)
    return y_coords[::-1]


def eles_to_xcoord(eles):
    x_coords = []
    for ele in eles:
        off = ele % 4
        if off == 1:
            x_coords.append(-24)
        elif off == 2:
            x_coords.append(8)
        elif off == 3:
            x_coords.append(-8)
        elif off == 0:
            x_coords.append(24)
    return x_coords


def eles_to_coords(eles):
    xs = eles_to_xcoord(eles)
    ys = eles_to_ycoord(eles)
    return np.array((xs, ys)).T


def plot_1D_pics(k, est_csd, est_pots, tp, Fs, cut=9):
    plt.figure(figsize=(12, 8))
    # plt.suptitle('plane: '+str(k.estm_x[cut,0])+' $\mu$m '+' $\lambda$ : '+str(k.lambd)+
    # '  R: '+ str(k.R))
    ax1 = plt.subplot(122)
    set_axis(ax1, -0.05, 1.05, letter='D')
    make_plot_spacetime(ax1, k.estm_x, k.estm_y, est_csd[cut, :, :], Fs,
                        title='Estimated CSD', cmap='bwr')

    # plt.xlim(250, 400)
    # plt.xticks([250, 300, 350, 400], [-50, 0, 50, 100])
    ax2 = plt.subplot(121)
    set_axis(ax2, -0.05, 1.05, letter='C')
    make_plot_spacetime(ax2, k.estm_x, k.estm_y, est_pots[cut, :, :],
                        title='Estimated LFP', cmap='PRGn')
    plt.axvline(tp / Fs * 1000, ls='--', color='grey', lw=2)
    # plt.xlim(250, 400)
    # plt.xticks([250, 300, 350, 400], [-50, 0, 50, 100])
    # plt.tight_layout()
    plt.savefig('figure_1D_pics', dpi=300)


def plot_2D_pics(k, est_csd, est_pots, tp, Fs, cut, save=0) -> (plt.Figure, plt.Axes):
    fig, axes = plt.subplots(1,2, figsize=(15, 8))
    # set_axis(axes[0], -0.05, 1.05, letter='B')
    make_plot(axes[0],fig, k.estm_x, k.estm_y, est_csd[:, :, tp],
              title='Estimated CSD', cmap='bwr')
    # for i in range(383): plt.text(ele_pos_for_csd[i,0], ele_pos_for_csd[i,1]+8, str(i+1))
    axes[0].axvline(k.estm_x[cut][0], ls='--', color='grey', lw=2)
    # set_axis(axes[1], -0.05, 1.05, letter='A')
    make_plot(axes[1],fig, k.estm_x, k.estm_y, est_pots[:, :, tp],
              title='Estimated LFP', cmap='PRGn')
    # plt.suptitle(' $\lambda$ : '+str(k.lambd)+ '  R: '+ str(k.R))
    # plt.savefig('figure_2D_pics', dpi=300)
    return fig,axes


def do_kcsd(ele_pos_for_csd, pots_for_csd, ele_limit):
    ele_position = ele_pos_for_csd  # [:ele_limit[1]][0::1]
    csd_pots = pots_for_csd #  [:ele_limit[1]][0::1]
    k = KCSD2D(ele_position, csd_pots,
               h=1, sigma=1, R_init=32, lambd=1e-9,
               xmin=-42, xmax=42, gdx=4,
               ymin=0, ymax=200, gdy=4)
    # k.L_curve(Rs=np.linspace(16, 48, 3), lambdas=np.logspace(-9, -3, 20))
    return k, k.values('CSD'), k.values('POT'), ele_position


# %%
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('sess_date')
    args = parser.parse_args()
    with open(args.config_file,'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])
    date_str = datetime.strptime(args.sess_date.split('_')[-1],'%y%m%d').strftime('%Y-%m-%d')

    ephys_dir = ceph_dir / 'Dammy' / 'ephys'
    assert ephys_dir.is_dir()
    sess = f'{args.sess_date.split("_")[0]}_{date_str}'
    sorting_dirs = get_sorting_dirs(ephys_dir, sess, 'sorting_no_si_drift', 'kilosort2_5_ks_drift',output_name='from_concat')
    sorter_outputs, recordings = get_sorting_objs(sorting_dirs)

    # start of kcsd script

    lowpass = 0.5
    highpass = 300
    Fs = 30000
    resamp = 12
    tp = 760

    Fs = int(Fs / resamp)

    recording = recordings[0]
    shank_i = 0
    probes_df = recording.get_probegroup().to_dataframe(complete=True)
    [print(probes_df.query(f'shank_ids=="{str(i)}"')['x'].min()) for i in sorted(probes_df['shank_ids'].unique())]
    for shank_i in sorted(probes_df['shank_ids'].unique()):
        ids2use = probes_df.query(f'shank_ids=="{str(shank_i)}"').index
        xs = probes_df.query(f'shank_ids=="{str(shank_i)}"')['x'].values
        xs = xs-xs.min()
        ys = probes_df.query(f'shank_ids=="{str(shank_i)}"')['y'].values
        ys = ys-ys.min()

        pots_for_csd = recording.get_traces(end_frame=10*Fs).T[ids2use]
        ele_pos_for_csd = np.array((xs, ys)).T

        k, est_csd, est_pots, ele_pos = do_kcsd(ele_pos_for_csd, pots_for_csd, ele_limit=(0, 16))

        # plot_1D_pics(k, est_csd, est_pots, tp, Fs, cut=15)
        plots = plot_2D_pics(k, est_csd, est_pots, tp, Fs, cut=15)
        plots[0].set_layout_engine('tight')
        plots[0].show()