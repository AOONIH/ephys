import argparse
import pickle
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
from aggregate_ephys_funcs import load_aggregate_sessions, decode_responses, aggregate_event_responses, \
    load_aggregate_td_df
from io_utils import posix_from_win
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt

def run_all_vs_all_decoding(event_responses, pips, decode_fn, n_runs=100):
    """Run all-vs-all decoding for each session."""
    pips_as_ints = {pip: i for i, pip in enumerate(pips)}
    decoders_dict = {}
    bad_dec_sess = set()
    for sessname in tqdm(list(event_responses.keys()), total=len(event_responses), desc='Decoding across sessions'):
        try:
            xys = [(event_responses[sessname][pip], np.full_like(event_responses[sessname][pip][:, 0, 0], pips_as_ints[pip]))
                   for pip in pips]
            xs = np.vstack([xy[0][:, :, -15:-5].mean(axis=-1) for xy in xys])
            ys = np.hstack([xy[1] for xy in xys])
            decoders_dict[f'{sessname}-allvall'] = decode_fn(xs, ys, n_runs=n_runs)
        except Exception as e:
            print(f'{sessname}-allvall failed: {e}')
            bad_dec_sess.add(sessname)
    return decoders_dict, bad_dec_sess

def plot_mean_confusion_matrix(all_cms, labels):
    """Plot mean confusion matrix."""
    mean_cm = np.mean(all_cms, axis=0)
    cmap_norm = TwoSlopeNorm(vcenter=1 / mean_cm.shape[0], vmin=0, vmax=0.5)
    cm_plot = ConfusionMatrixDisplay(mean_cm, display_labels=labels)
    cm_plot.plot(cmap='bwr', include_values=False, colorbar=False, im_kw=dict(norm=cmap_norm))
    cm_plot.ax_.invert_yaxis()
    cm_plot.ax_.set_xlabel('')
    cm_plot.ax_.set_ylabel('')
    cm_plot.figure_.set_size_inches(2, 2)
    cm_plot.figure_.set_layout_engine('tight')
    cm_plot.figure_.show()
    return cm_plot

def plot_decoding_accuracy_boxplot(accuracy_df, chance_level):
    """Plot boxplot of decoding accuracy for each pip."""
    fig, ax = plt.subplots()
    ax.boxplot([accuracy_df[col] for col in accuracy_df.columns], labels=accuracy_df.columns)
    ax.set_ylabel('Decoding Accuracy')
    ax.axhline(chance_level, color='k', linestyle='--')
    fig.set_size_inches(3.5, 2.2)
    fig.set_layout_engine('tight')
    fig.show()
    return fig, ax

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('pkldir')
    parser.add_argument('session_topology_csv', help="Path to session topology CSV file")
    parser.add_argument('--outdir', default='.', help="Directory to save decoding results")
    args = parser.parse_args()

    sys_os = platform.system().lower()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    home_dir = Path(config[f'home_dir_{sys_os}'])
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])

    session_topology_path = ceph_dir / posix_from_win(args.session_topology_csv)
    session_topology = pd.read_csv(session_topology_path)

    session_filter = f'sess_order in ["pre","post"] & date >= 240219'
    all_sess_info = session_topology.query(session_filter)
    sessions2use = [Path(row['sound_bin']).stem.replace("_SoundData","") for idx,row in all_sess_info.iterrows()]

    pkldir = ceph_dir / posix_from_win(args.pkldir)
    pkls = list(pkldir.glob('*pkl'))
    pkls2load = [pkl for pkl in pkls if Path(pkl).stem in sessions2use]
    sessions = load_aggregate_sessions(pkls2load)

    # Aggregate event responses
    window = (-0.1, 0.25)
    event_responses = aggregate_event_responses(
        sessions, events=None, window=window,
        pred_from_psth_kwargs={'use_unit_zscore': True, 'use_iti_zscore': False, 'baseline': 0, 'mean': None, 'mean_axis': 0}
    )

    # Perform all-vs-all decoding
    pips = [f'{pip}-0' for pip in 'ABCD']
    decoders_dict, bad_dec_sess = run_all_vs_all_decoding(event_responses, pips, decode_responses, n_runs=100)

    # Save decoding results
    outdir = ceph_dir / posix_from_win(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    decoding_results_path = outdir / 'decoding_results_pre_post_sessions.pkl'
    with open(decoding_results_path, 'wb') as f:
        pickle.dump(decoders_dict, f)
    print(f"Decoding results saved to {decoding_results_path}")

    # Visualize confusion matrix across sessions
    all_cms = [decoders_dict[dec_name]['data'].cm for dec_name in decoders_dict.keys()]
    plot_mean_confusion_matrix(all_cms, labels=pips)

    # Calculate and visualize mean decoding accuracy for each pip across sessions
    accuracy_across_sessions = [np.diagonal(cm) for cm in all_cms]
    accuracy_df = pd.DataFrame(accuracy_across_sessions, columns=pips)
    plot_decoding_accuracy_boxplot(accuracy_df, chance_level=1 / len(pips))

if __name__ == '__main__':
    main()
