import matplotlib
from matplotlib.cm import ScalarMappable

from aggregate_ephys_funcs import *
from aggregate_ephys_funcs import load_or_generate_event_responses
from io_utils import posix_from_win
from plot_funcs import format_axis


def run_multiclass_decoding(event_responses, animals, x_ser, decoding_window, pips, cache_path=None, overwrite=False,
                            sess_list=None):
    """
    Multiclass decoding for given pips (e.g. ['A-0','B-0','C-0','D-0']) over specified window.
    Returns decoder_df with accuracy and confusion matrix for each session.
    """

    if cache_path is not None and Path(cache_path).is_file() and not overwrite:
        with open(cache_path, 'rb') as f:
            decoder_df = pickle.load(f)
        return decoder_df

    records = []
    conf_mats = []

    if sess_list is None:
        sess_list = list(event_responses.keys())
    for sessname in tqdm(event_responses.keys(), desc='multiclass decoding', total=len(event_responses)):
        if sessname not in sess_list:
            continue
        session_events = event_responses[sessname]
        if not all(p in session_events for p in pips):
            continue
        xs_list = [session_events[pip] for pip in pips]
        idx_4_decoding = [np.where(x_ser == t)[0][0] for t in decoding_window]
        xs = np.vstack([x[:, :, idx_4_decoding[0]:idx_4_decoding[1]].mean(axis=-1) for x in xs_list])
        ys = np.hstack([np.full(x.shape[0], ci) for ci, x in enumerate(xs_list)])
        decode_result = decode_responses(xs, ys, n_runs=1, dec_kwargs={'cv_folds': 10})
        decode_result['data'].plot_confusion_matrix(labels=pips)
        
        acc = np.mean(decode_result['data'].accuracy)
        conf_mat = decode_result['data'].cm
        record = {
            'sessname': sessname,
            'animal': next((a for a in animals if a in sessname), None),
            'accuracy': acc,
            # 'confusion_matrix': conf_mat
        }
        records.append(record)
        conf_mats.append(conf_mat)

    decoder_df = pd.DataFrame(records).set_index('sessname')
    if cache_path is not None:
        with open(cache_path, 'wb') as f:
            pickle.dump(decoder_df, f)
    return decoder_df, conf_mats
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('pkldir')
    parser.add_argument('event_responses_pkl')
    parser.add_argument('--plot_config_path', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    # Load plot config if provided
    plot_config = {}
    if args.plot_config_path is not None:
        plot_config_path = Path(args.plot_config_path)
        if plot_config_path.is_file():
            with open(plot_config_path, 'r') as f:
                plot_config = yaml.safe_load(f)

    event_responses, event_features = load_or_generate_event_responses(args, plot_config)
    animals = plot_config.get('animals', list({sess.split('_')[0] for sess in event_responses.keys()}))
    sess2use = [sess for sess in event_responses.keys() if any(a in sess for a in animals)]
    window = plot_config.get('window', [-0.25, 1])
    x_ser = np.round(np.arange(window[0], window[1]+0.01, 0.01),2)
    decoding_window = plot_config.get('decoding_window', [0.1, 0.25])
    pips = plot_config.get('pips2decode',['A-0', 'B-0', 'C-0', 'D-0'])
    ceph_dir = Path(yaml.safe_load(open(args.config_file))['ceph_dir_' + platform.system().lower()])
    decoding_figdir = ceph_dir / posix_from_win(
        plot_config.get('psth_figdir', r'X:\Dammy\figures\aggregate_decoding'))

    decoding_figdir.mkdir(parents=True, exist_ok=True)
    decode_cache_path = decoding_figdir / 'multiclass_decode_results_cache.pkl'

    decoder_df, conf_mats = run_multiclass_decoding(
        event_responses, animals, x_ser, decoding_window, pips,
        cache_path=decode_cache_path, overwrite=True, sess_list=sess2use
    )

    cm_plot_kwargs = plot_config.get('cm_plot_kwargs', {}).copy()

    # Set color map kwargs
    if all(e in list(cm_plot_kwargs.get('im_kwargs', {}).keys()) for e in ['vmin', 'vmax']):
        vmin, vmax = [cm_plot_kwargs['im_kwargs'][e] for e in ['vmin', 'vmax']]
        vc = 1/len(pips)
        cm_plot_kwargs['im_kwargs']['norm'] = matplotlib.colors.TwoSlopeNorm(vcenter=vc, vmin=vmin, vmax=vmax)
        # pop vcenter, vmin, vmax
        for e in [ 'vmin', 'vmax']:
            cm_plot_kwargs['im_kwargs'].pop(e)

    # Plot mean confusion matrix across sessions
    cm_plot = plot_aggr_cm(np.array(conf_mats), **cm_plot_kwargs,)
    cm_plot[0].show()
    cm_plot[0].savefig(decoding_figdir / f'cross_session_confusion_matrix.pdf')

    sm = ScalarMappable(cm_plot_kwargs['im_kwargs']['norm'], cmap=cm_plot_kwargs['im_kwargs']['cmap'])
    sm.set_array([])  # Required to satisfy colorbar API
    cbar = plt.subplots(figsize=(0.5, 0.25))  # Adjust size as needed
    cb = cbar[0].colorbar(sm, cax=cbar[1], orientation='horizontal',)
    cb.set_label('Decoding rate')  # Optional
    format_axis(cbar[1])
    cbar[0].set_layout_engine('tight')
    cbar[0].show()
    cbar[0].savefig(decoding_figdir / f'cross_session_confusion_matrix_colorbar.pdf')
    # Print mean accuracy
    print(f"Mean accuracy across sessions: {decoder_df['accuracy'].mean():.3f}")
