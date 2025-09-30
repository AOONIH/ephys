from decoding_funcs import get_decoder_accuracy_from_pkl, get_property_from_decoder_pkl
from ephys_analysis_funcs import *
import argparse
import yaml
import platform
import multiprocessing
from functools import partial

from io_utils import posix_from_win

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])

    pickles_dir = ceph_dir / posix_from_win(r'X:\Dammy\ephys_concat_pkls')
    # pickles_dir = ceph_dir / posix_from_win(r'X:\Dammy\ephys_pkls')
    assert pickles_dir.is_dir()
    pickle_files = list(pickles_dir.glob('*.pkl'))

    sess_topology_path = ceph_dir/posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_ephys_2401.csv')
    session_topology = pd.read_csv(sess_topology_path)
    sesstype = ['pre','post']
    animals = ['DO79','DO81']
    # animals = ['DO80', 'DO81', 'DO82']
    main_sessions = [Path(e).stem.replace('_SoundData','')
                     for e in session_topology.query('sess_order in @sesstype')['sound_bin'].values]

    session_decoder_accuracy = {}

    pickle_files = sorted([e for e in pickle_files if any([a in e.stem for a in animals]) and e.stem in main_sessions])
    # add rare freq from DO79
    rare_freq_DO79_pickles = [e for e in list((ceph_dir / posix_from_win(r'X:\Dammy\ephys_pkls')).glob('*.pkl'))
                              if any(d in e.stem for d in ['240219', '240220', '240221']) and e.stem in main_sessions]
    pickle_files.extend(rare_freq_DO79_pickles)

    with multiprocessing.Pool() as pool:
        results = pool.map(get_decoder_accuracy_from_pkl,pickle_files)
    for sess_i, ses_result in enumerate(results):
        session_decoder_accuracy[pickle_files[sess_i].stem] = ses_result
    # print(results)
    [session_decoder_accuracy.pop(sess) for sess in list(session_decoder_accuracy.keys()) if not session_decoder_accuracy[sess]
     or 'all_vs_all_no_base' not in session_decoder_accuracy[sess]]
    decoding_accuracies_tobase = [[session_decoder_accuracy[sess][f'all_vs_all_no_base'] for sess in session_decoder_accuracy]
                                  for pip in 'ABCD']
    cross_sess_plot = plt.subplots(figsize=(8,8))
    # cross_sess_plot[1].scatter([0]*len(decoding_accuracies[0]),[np.mean(e) for e in decoding_accuracies[0]])
    cross_sess_plot[1].boxplot([[np.mean(e) for e in ee[0]] for ee in decoding_accuracies_tobase],
                               labels=list('ABCD'),)
    cross_sess_plot[1].set_ylim([0,1])
    cross_sess_plot[1].tick_params(axis='both',labelsize=16)
    cross_sess_plot[1].axhline(0.25,linestyle='--',color='k')
    cross_sess_plot[1].set_title('decoding accuracy for pip vs base tone across sessions',fontsize=16)
    cross_sess_plot[0].show()
    cross_sess_plot[0].set_constrained_layout(True)
    cross_sess_plot[0].savefig(f'decoding_accuracies_across_{"_".join(animals)}_passive_{"_".join(sesstype)}_sessions.svg')

    # rare vs freq accuracy
    rare_vs_freq_plot = plt.subplots(figsize=(8,8))
    rare_vs_freq_decoding_accuracies = [[session_decoder_accuracy[sess][decoder_name] for sess in session_decoder_accuracy]
                                        for decoder_name in ['rec_dist_A','rec_dist_A_shuffle']]
    rare_vs_freq_plot[1].boxplot([[np.mean(e) for e in ee[0]] for ee in rare_vs_freq_decoding_accuracies],
                                 labels=['rare vs freq','rare vs freq shuffled'],)
    # rare_vs_freq_plot[1].set_ylim([0,1])
    rare_vs_freq_plot[0].set_constrained_layout(True)
    rare_vs_freq_plot[1].tick_params(axis='both',labelsize=16)
    rare_vs_freq_plot[1].axhline(0.5,linestyle='--',color='k')
    rare_vs_freq_plot[0].show()
    rare_vs_freq_plot[0].savefig(f'rare_vs_freq_accuracies_across_sessions.svg')
    # get confusion matrices

    with multiprocessing.Pool() as pool:
        cms = pool.map(partial(get_property_from_decoder_pkl,decoder_name='all_vs_all',property_name='cm'),
                       pickle_files)

    # remove nones from list
    cms = [e for e in cms if e is not None]
    all_sess_perf = np.array(cms).diagonal(axis1=1, axis2=2)
    all_vs_all_perf_plot = plt.subplots(figsize=(8,8))
    all_vs_all_perf_plot[1].boxplot(all_sess_perf,labels=list('ABCD')+['base'],bootstrap=10000,)
    all_vs_all_perf_plot[1].set_ylim([0,1])
    all_vs_all_perf_plot[1].tick_params(axis='both',labelsize=16)
    all_vs_all_perf_plot[1].axhline(0.2,linestyle='--',color='k')
    all_vs_all_perf_plot[1].set_title('decoding accuracy for pip (5 way) across sessions',fontsize=16)
    all_vs_all_perf_plot[0].show()
    all_vs_all_perf_plot[0].set_constrained_layout(True)

    mean_cm_plot_ = ConfusionMatrixDisplay(np.mean(cms,axis=0),display_labels=list('ABCD')+['base'],)
    mean_cm_plot = mean_cm_plot_.plot(cmap='copper',include_values=False)
    mean_cm_plot.ax_.invert_yaxis()
    mean_cm_plot.figure_.show()
    # mean_cm_plot.figure_.set_constrained_layout(True)
    mean_cm_plot.figure_.savefig(f'all_vs_all_cm_{"_".join(animals)}_passive_{"_".join(sesstype)}_sessions.svg')
    # all_vs_all_perf_plot[0].savefig('all_vs_all_across_passive_sessions.svg')
