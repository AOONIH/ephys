import json
import warnings
from copy import copy

import matplotlib
import numpy as np
import pandas as pd
import yaml
import logging
import platform
import pickle
from pathlib import Path

from matplotlib import pyplot as plt
from scipy.stats import sem, ttest_ind, ttest_1samp
from tqdm import tqdm
import joblib
from aggregate_ephys_funcs import get_responses_by_pip_and_condition, run_decoding, parse_args, plot_aggr_cm
from population_analysis_funcs import PopPCA
from behviour_analysis_funcs import get_all_cond_filts
from io_utils import posix_from_win
from plot_funcs import plot_shaded_error_ts, format_axis, add_x_scale_bar, get_sorted_psth_matrix, plot_sorted_psth_matrix
from reformat_dir_struct import extract_date
from save_utils import save_stats_to_tex
from spike_time_utils import zscore_by_trial
from unit_analysis import UnitAnalysis


def rolling_mean_convolve(arr, window):
    return np.convolve(arr, np.ones(window) / window, mode='valid')


def padded_rolling_mean(arr, window):

    _arr = arr.copy()
    _arr = arr.reshape(-1,arr.shape[-1])
    if arr.ndim == 1:
        mean_vals = rolling_mean_convolve(arr.flatten(), window)
        pad = np.full(window - 1, np.nan)  # or use another method like forward-fill
        return np.concatenate((pad, mean_vals))
    else:
        mean_vals = [rolling_mean_convolve(e,window) for e in _arr]
        _pad = [np.full(window - 1, np.nan) for e in _arr]
        _padded = [np.concatenate((e,ee)) for e,ee in zip(_pad, mean_vals)]
        return np.concatenate(_padded,axis=0).reshape(arr.shape)



def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
    logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

    # load config
    config_path = Path(args.config_file)
    if config_path.is_file():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded config from {config_path}")
    else:
        logging.warning(f"Config {config_path} not found. Continuing without.")

    ceph_dir = config['ceph_dir_' + platform.system().lower()]

    # Load plot config if provided
    plot_config = {}
    if args.plot_config_path:
        plot_path = Path(args.plot_config_path)
        if plot_path.is_file():
            with open(plot_path, 'r') as f:
                plot_config = yaml.safe_load(f)
            logging.info(f"Loaded plot config from {plot_path}")
        else:
            logging.warning(f"Plot config {plot_path} not found. Continuing without.")

    pkl_dir = Path(args.event_responses_pkl).parent

    # Set figure directories
    psth_figdir = ceph_dir / posix_from_win(plot_config.get('psth_figdir', r'X:\Dammy\figures\psth_analysis'))

    decoding_figdir = ceph_dir / posix_from_win(plot_config.get('decoding_figdir',
                                                                r'X:\Dammy\figures\rare_freq_decoding'))
    pca_fidir = ceph_dir / posix_from_win(plot_config.get('pca_figdir',
                                                          r'X:\Dammy\figures\pca_plots'))
    for fig_dir in [decoding_figdir, pca_fidir, psth_figdir]:
        if not fig_dir.is_dir():
            fig_dir.mkdir(parents=False)
    plt.subplots()

    plt.style.use('figure_stylesheet.mplstyle')

    # all_stim_resps = AggregateSession(pkl_dir,args, plot_config,['A-0','X','base'])
    all_stim_resps = AggregateSession(pkl_dir,args, plot_config,['A-0'])
    all_stim_resps.aggregate_mean_sess_responses()
    plot_config['psth_plot_kwargs']['plot_window'] = [-0.25,1]
    all_stim_resps.plot_sorted_psth_mat(plot_kwargs=plot_config['psth_plot_kwargs'])
    window = plot_config[f'plot_window']

    # for pip in ['A-0','X']:
    for pip in ['A-0']:
        add_x_scale_bar(all_stim_resps.plots[f'{pip}_sorted_psth'][1][1], size=0.2,
                        label='0.2 s', color='k', frameon=False, fontproperties={'size': 5})
        all_stim_resps.plots[f'{pip}_sorted_psth'][1][0].locator_params(axis='y', nbins=2)
        all_stim_resps.plots[f'{pip}_sorted_psth'][0].set_size_inches(1.2, 1.8)
        all_stim_resps.plots[f'{pip}_sorted_psth'][0].set_layout_engine('tight')

        save_name = psth_figdir / f"{pip}_all_animals_{window[0]}_{window[1]}.pdf"
        logging.info(f"Saving {save_name}")
        all_stim_resps.plots[f'{pip}_sorted_psth'][0].savefig(save_name, dpi=600)
        plt.close(all_stim_resps.plots[f'{pip}_sorted_psth'][0])
    #
    all_stim_resps.aggregate_sess_decoding('stim_decoding',
                                           df_save_path=decoding_figdir/'stim_decoding_df.h5')
    all_stim_resps.plot_decoder_boxplot(decoding_figdir)
    for pips2decode in plot_config['stim_decoding']['pips2decode']:
        all_stim_resps.decoding_ttest('_vs_'.join(pips2decode),'data','shuffled')
    for ttest_name, ttest_res in all_stim_resps.ttest_res.items():
        save_stats_to_tex(ttest_res,decoding_figdir / f'{ttest_name}.tex')


    logging.info("All done.")

class ConcatResponses:
    def __init__(self, batch_event_responses: dict, event_features: dict,pips, plot_config: dict,
                 zscore_flag=False, **psth_kwargs):

        self.concatenated_event_sessnames = None
        self.peak_ts_by_pips = None
        self.event_mats_4_sorting = None
        self.concatenated_event_responses_sem = None
        self.concatenated_event_responses = None
        self.smoothed_event_responses = None
        self.smoothed_event_responses_sem = None

        self.event_features = event_features

        self.get_concatenated_event_responses(batch_event_responses, zscore_flag=zscore_flag)

        self.get_sorted_resp_mat(batch_event_responses, pips, plot_config['window'], **psth_kwargs)
        self.get_smoothed_responses(batch_event_responses,psth_kwargs.get('smoothing_window', 25))

    def get_concatenated_event_responses(self, batch_event_responses: dict, zscore_flag=False):
        concatenated_event_responses = {}
        concatenated_event_responses_sem = {}
        concatenated_event_sessnames = {}

        if zscore_flag:
            batch_event_responses = zscore_by_trial(batch_event_responses)

        for pip in list(batch_event_responses.values())[0]:
            concatenated_event_sessnames[pip] = np.concatenate([[e]*batch_event_responses[e][pip].shape[1]
                                                                for e in batch_event_responses], axis=0)

            concatenated_event_responses[pip] = np.concatenate([batch_event_responses[e][pip].mean(axis=0)
                                                                for e in batch_event_responses], axis=0)

            concatenated_event_responses_sem[pip] = np.concatenate([sem(batch_event_responses[e][pip])
                                                                    for e in batch_event_responses], axis=0)
        self.concatenated_event_responses = concatenated_event_responses
        self.concatenated_event_responses_sem = concatenated_event_responses_sem
        self.concatenated_event_sessnames = concatenated_event_sessnames


    def get_sorted_resp_mat(self,batch_event_responses: dict, pips_2_plot: list, window: tuple, **psth_plot_kwargs):
        resp_mats_by_pips = {}
        peak_ts_by_pips = {}

        # filter out sessions with insufficient data
        for sessname in list(batch_event_responses.keys()):
            if not all([e in list(batch_event_responses[sessname].keys()) for e in pips_2_plot]):
                logging.warning(f'Skipping {sessname} due to missing data')
                batch_event_responses.pop(sessname)
                continue

        for sessname in list(batch_event_responses.keys()):
                if any([len(batch_event_responses[sessname][pip]) < 4  for pip in pips_2_plot]):
                    logging.warning(f'Skipping {sessname} due to insufficient data')
                    batch_event_responses.pop(sessname)
                    continue

        if len(batch_event_responses) == 0:
            return None
        for pip in pips_2_plot:
            # for animal in animal_list:
            kwargs = dict(window=window, sessname_filter=None, **psth_plot_kwargs)
            _, batch_resp_mat, peak_ts, _ = get_sorted_psth_matrix(batch_event_responses, pip, pip, **kwargs)

            resp_mats_by_pips[pip] = batch_resp_mat
            peak_ts_by_pips[pip] = peak_ts
        self.event_mats_4_sorting = resp_mats_by_pips
        self.peak_ts_by_pips = peak_ts_by_pips

    def get_smoothed_responses(self,batch_event_responses: dict,smoothing_window=25):
        concatenated_event_responses = {}
        concatenated_event_responses_sem = {}

        smoothed_responses = {}
        for sessname in list(batch_event_responses.keys()):
            smoothed_responses[sessname] = {}
            for pip in list(batch_event_responses[sessname].keys()):
                smoothed_responses[sessname][pip] = padded_rolling_mean(batch_event_responses[sessname][pip],
                                                                        window=smoothing_window)

        for pip in list(smoothed_responses.values())[0]:
            concatenated_event_responses[pip] = np.concatenate([smoothed_responses[e][pip].mean(axis=0)
                                                                for e in smoothed_responses], axis=0)
            concatenated_event_responses_sem[pip] = np.concatenate([sem(smoothed_responses[e][pip])
                                                                    for e in smoothed_responses], axis=0)
        self.smoothed_event_responses = concatenated_event_responses
        self.smoothed_event_responses_sem = concatenated_event_responses_sem



class AggregateSession:
    def __init__(self, batch_pkl_dir, args, plot_config,pips_2_plot=None):


        self.pca = {}
        self.ttest_res = {}
        self.plot_config = plot_config

        self.decoder_name = None
        self.aggregate_decoding_df = None
        self.cms = None
        if not batch_pkl_dir.exists():
            raise FileNotFoundError(f'Batch pkl dir {batch_pkl_dir} does not exist')
        self.batch_pkl_dir = batch_pkl_dir if batch_pkl_dir.is_dir() else batch_pkl_dir.parent
        self._get_batch_pkl_paths()
        self.event_features = {}
        self._get_event_features(plot_config['pkl_filts'])

        self.args = args

        self.pips_2_plot = pips_2_plot if pips_2_plot is not None else plot_config['pips_2_plot']
        self.window = plot_config['window']
        dt = plot_config.get('dt',0.01)
        self.x_ser = np.round(np.arange(self.window[0],self.window[1]+dt,dt),2)

        self.peak_ts_by_pips = None
        self.event_mats_4_sorting = None
        self.concatenated_event_responses_sem = None
        self.concatenated_event_responses = None
        self.concatenated_sessnames = None

        self.smoothed_concatenated_responses = None
        self.smoothed_concatenated_responses_sem = None
        self.plots = {}


    def _get_batch_pkl_paths(self):
        pkl_filts = self.plot_config.get('pkl_filts', [])
        batch_pkl_paths = [p for p in list(self.batch_pkl_dir.glob('*.pkl'))
                           if (any([pkl_filt in p.stem for pkl_filt in pkl_filts]) if pkl_filts else True)]
        batch_joblib_paths = [p for p in list(self.batch_pkl_dir.glob('*.joblib'))
                              if (any([pkl_filt in p.stem for pkl_filt in pkl_filts]) if pkl_filts else True)]

        # select joblib if it exists else use pkl
        all_paths = batch_joblib_paths + batch_pkl_paths
        paths_2_load = [(self.batch_pkl_dir / p).with_suffix('.joblib')
                        if (self.batch_pkl_dir / p).with_suffix('.joblib').exists()
                        else (self.batch_pkl_dir / p).with_suffix('.pkl')
                        for p in set([e.stem for e in all_paths])]
        assert len(paths_2_load) > 0
        self.batch_pkl_paths = paths_2_load

    def _get_event_features(self,pkl_filts):
        event_features_path_filt = [f'{"_".join(filt.split("_")[1:])}_features' for filt in pkl_filts]
        event_features_paths = [p for p in list(self.batch_pkl_dir.iterdir())
                                if any([pkl_filt in p.stem for pkl_filt in event_features_path_filt])
                                and p.suffix in ['.pkl', '.joblib']]

        event_features_dicts = {}
        # load and merge dict with event features
        for p in event_features_paths:
            event_features_dicts = {**event_features_dicts, **joblib.load(p)}

        self.event_features = event_features_dicts

    @staticmethod
    def _load_batch_events_from_disk(batch_path: Path):
        if batch_path.with_suffix('.joblib').exists():
            batch = joblib.load(batch_path.with_suffix('.joblib'))
        else:
            with open(batch_path.with_suffix('.pkl'), 'rb') as f:
                batch = pickle.load(f)

        assert isinstance(batch, dict), f"Batch {batch_path} should be a dict"
        assert isinstance(list(batch.values())[0]['A-0'], np.ndarray), f"Batch {batch_path} should be a dict of dicts of np.arrays"

        return batch

    def _save_aggr_means(self, concat_savename:Path):
        savename_stem = concat_savename.stem
        joblib.dump(self.concatenated_event_responses, concat_savename)
        joblib.dump(self.concatenated_event_responses_sem, concat_savename.with_stem(f'{savename_stem}_sem'))
        joblib.dump(self.smoothed_concatenated_responses,concat_savename.with_stem(f'{savename_stem}_smooth'))
        joblib.dump(self.smoothed_concatenated_responses_sem,concat_savename.with_stem(f'{savename_stem}_smooth_sem'))
        joblib.dump(self.concatenated_sessnames,concat_savename.with_stem(f'{savename_stem}_sessnames'))

    def _load_aggr_means(self, concat_savename:Path):
        savename_stem = concat_savename.stem
        if not concat_savename.exists():
            warnings.warn(f'Concatenation {concat_savename} does not exist')
            return False
        self.concatenated_event_responses = joblib.load(concat_savename)
        self.concatenated_event_responses_sem = joblib.load(concat_savename.with_stem(f'{savename_stem}_sem'))
        self.smoothed_concatenated_responses = joblib.load(concat_savename.with_stem(f'{savename_stem}_smooth'))
        self.smoothed_concatenated_responses_sem = joblib.load(concat_savename.with_stem(f'{savename_stem}_smooth_sem'))
        self.concatenated_sessnames =joblib.load(concat_savename.with_stem(f'{savename_stem}_sessnames'))

        return True

    def subset_responses(self, batch_responses, conds=None, cond_filts=None, filt_by_prc=False, sessname_filts=None,
                         **kwargs):
        pips_2_plot = kwargs.get('pips',self.pips_2_plot)
        if sessname_filts is not None:
            batch_responses = {sessname: {pip: batch_responses[sessname][pip] for pip in batch_responses[sessname]}
                               for sessname in batch_responses if any([e in sessname for e in sessname_filts])}
            if len(batch_responses) == 0:
                logging.info(f'No responses found for {sessname_filts}')
                return batch_responses

        # Get subset of responses based on event features
        if conds is not None:
            assert cond_filts is not None
            pips = [p.split('_')[0] for p in kwargs.get('pips',self.pips_2_plot)]
            batch_responses = get_responses_by_pip_and_condition(pips, batch_responses,
                                                                 self.event_features, conds, cond_filts,
                                                                 zip_pip_conds=True)
            pips_2_plot = kwargs.get('pips',[f'{pip}_{cond}' for pip,cond in zip(self.pips_2_plot,conds)])

            if len(batch_responses) == 0:
                logging.info(f'No responses found for {conds.keys()}')
                return batch_responses

        for sessname, sess_resps in list(batch_responses.items()):
            if any([len(pip_resps) == 0 for pip_resps in sess_resps.values()]):
                batch_responses.pop(sessname)

        if len(batch_responses) == 0:
            logging.info(f'No responses found for {conds}')
            return batch_responses

        # Get subset of response above participation rate threshold
        if filt_by_prc:
            if kwargs.get('prc_thr', None) is not None:
                unit_obj = UnitAnalysis(batch_responses, pips_2_plot, resp_window=self.window)
                unit_obj.get_participation_rate(self.window, 2, )

                batch_responses = unit_obj.filter_by_prc_rate(prc_threshold=kwargs.get('prc_thr',
                                                                                       self.plot_config.get('prc_thr')),
                                                              prc_pips=kwargs.get('prc_pips'),
                                                              prc_mutual=kwargs.get('prc_mutual'))

        if kwargs.get('filt_by_trial_num', False):
            pips_4_filt = kwargs.get('pip_4_filt', [pips_2_plot[0]])
            if len(pips_4_filt) != len(pips_2_plot):
                pips_4_filt = [pips_4_filt[0]]*len(pips_2_plot)
            pips_2_filt = kwargs.get('pip_2_filt', pips_2_plot)
            for sessname, sess_resps in batch_responses.items():
                for run_i, (pip, filt_pip) in enumerate(zip(pips_2_filt,pips_4_filt)):
                    if isinstance(filt_pip, str):
                        filt_pips = [filt_pip]
                    else:
                        filt_pips = filt_pip
                    _filt_pips = [filt_pip.split('_')[0] for filt_pip in filt_pips]
                    filt_trial_nums = np.hstack(
                        [np.array(self.event_features[sessname][_filt_pip].get('trial_nums', None))
                         for _filt_pip in _filt_pips]
                    )
                    _pip = pip.split('_')[0]
                    td_df: pd.DataFrame = self.event_features[sessname][_pip].get('td_df', None)
                    if len(pip.split('_')) == 2:
                        td_df = td_df.query(cond_filts[pip.split('_')[1]])
                    n_events = min(sess_resps[pip].shape[0], td_df.shape[0])
                    td_df = td_df.head(n_events)
                    trial_nums = td_df.index.unique().to_series()
                    filt_func = kwargs.get('filt_func', )
                    filt_trial_nums = filt_func(trial_nums, filt_trial_nums)
                    mask = trial_nums.isin(filt_trial_nums)
                    try:
                        batch_responses[sessname][pip] = batch_responses[sessname][pip][:n_events][mask]
                    except IndexError as e:
                        print(sessname,pip,e,run_i)


        return batch_responses

    def aggregate_mean_sess_responses(self,tag=None, conds=None,concat_savename=None, **kwargs):
        if tag is None:
            tag = 'all'

        concat_resps_by_batch = []
        batch_sessnames = []

        cond_filts = get_all_cond_filts()

        pips_2_plot = copy(self.pips_2_plot)
        if conds is not None:
            pips_2_plot = [f'{pip}_{cond}' for pip,cond in zip(pips_2_plot,conds)]

        if concat_savename is not None and kwargs.get('reload_save', True):
            reloaded: bool = self._load_aggr_means(concat_savename)
            if reloaded:
                return None

        for batch_path in tqdm(self.batch_pkl_paths, desc=f'Aggregating {tag}',total=len(self.batch_pkl_paths)):
            try:
                batch_responses = self._load_batch_events_from_disk(batch_path)
            except AssertionError:
                continue

            # Get subset of responses based on event features anf participation rate
            batch_responses = self.subset_responses(batch_responses, conds, cond_filts, **kwargs)
            to_pop = set()
            [to_pop.add(sess) for sess in self.plot_config.get('excluded_sessions',[])]
            for sessname in list(batch_responses.keys()):
                if not all([pip in list(batch_responses[sessname].keys()) for pip in pips_2_plot]):
                    to_pop.add(sessname)
                if not all([len(batch_responses[sessname][pip])>3 for pip in pips_2_plot]):
                    to_pop.add(sessname)
                if any([(batch_responses[sessname][pip]).size == 0 for pip in pips_2_plot]):
                    to_pop.add(sessname)

            [batch_responses.pop(sessname) for sessname in to_pop if sessname in batch_responses]

            if len(batch_responses) == 0:
                logging.info(f'No responses found for {cond_filts}')
                continue

            _resp_obj = ConcatResponses(batch_responses, self.event_features,
                                        pips_2_plot,self.plot_config,kwargs.get('zscore_flag',False),
                                        **self.plot_config['psth_plot_kwargs'])

            if _resp_obj is not None:
                concat_resps_by_batch.append(_resp_obj)
                batch_sessnames.extend(list(batch_responses.keys()))

        # Aggregate across sessions
        if not concat_resps_by_batch:
            return None

        concat_resps_by_batch = [e for e in concat_resps_by_batch if e is not None]
        self.peak_ts_by_pips = {pip: np.concatenate([e.peak_ts_by_pips[pip] for e in concat_resps_by_batch
                                                     if e.peak_ts_by_pips is not None], axis=0)
                                for pip in pips_2_plot}
        self.event_mats_4_sorting = {pip: np.concatenate([e.event_mats_4_sorting[pip] for e in concat_resps_by_batch
                                                          if e.event_mats_4_sorting is not None],
                                                         axis=0)
                                     for pip in pips_2_plot}
        self.concatenated_event_responses_sem = {pip: np.concatenate([e.concatenated_event_responses_sem[pip]
                                                                      for e in concat_resps_by_batch
                                                                      if e.concatenated_event_responses_sem is not None], axis=0)
                                                 for pip in pips_2_plot}
        self.concatenated_event_responses = {pip: np.concatenate([e.concatenated_event_responses[pip]
                                                                  for e in concat_resps_by_batch
                                                                  if e.concatenated_event_responses is not None], axis=0)
                                             for pip in pips_2_plot}
        self.smoothed_concatenated_responses = {pip: np.concatenate([e.smoothed_event_responses[pip]
                                                                      for e in concat_resps_by_batch
                                                                      if e.smoothed_event_responses is not None], axis=0)
                                                 for pip in pips_2_plot}
        self.smoothed_concatenated_responses_sem = {pip: np.concatenate([e.smoothed_event_responses_sem[pip]
                                                                     for e in concat_resps_by_batch
                                                                     if e.smoothed_event_responses_sem is not None],
                                                                    axis=0)
                                                for pip in pips_2_plot}
        self.concatenated_sessnames = np.concatenate([e.concatenated_event_sessnames[pips_2_plot[0]]
                                                      for e in concat_resps_by_batch
                                                      if e.smoothed_event_responses_sem is not None])
        logging.info(f'{pips_2_plot} shape: {[e.shape for e in self.concatenated_event_responses.values()]}')
        if concat_savename:
            self._save_aggr_means(concat_savename)
        return None

    def aggregate_sess_decoding(self,dec_tag, conds=None, **kwargs):
        cond_filts = get_all_cond_filts()

        pips2decode = self.plot_config[dec_tag]['pips2decode']
        decoding_window = self.plot_config[dec_tag]['decoding_window']

        df_loaded, cms_loaded = False, False

        if kwargs.get('df_save_path'):
            df_save_path = Path(kwargs.get('df_save_path'))
            cm_save_path = df_save_path.with_name(df_save_path.name.replace('df.h5', 'cm.npy'))
            if kwargs.get('reload_save', False) and df_save_path.is_file():
                self.aggregate_decoding_df = pd.read_hdf(df_save_path)
                df_loaded = True
            if kwargs.get('reload_save', False) and cm_save_path.is_file():
                self.cms = np.load(cm_save_path)
                cms_loaded = True
        if all([df_loaded, cms_loaded]):
            self.decoder_name = dec_tag
            logging.info(f'Decoding {dec_tag} reloaded')
            return

        assert decoding_window is not None, f"Decoding window must be specified in {self.plot_config} or in {kwargs}"

        batch_decoding_dfs = []
        batch_cms = []
        for batch_path in tqdm(self.batch_pkl_paths, total=len(self.batch_pkl_paths),
                               desc=f'Decoding batches',):
            batch_responses = self._load_batch_events_from_disk(batch_path)
            batch_responses = self.subset_responses(batch_responses, conds, cond_filts, **kwargs)
            if len(batch_responses) == 0:
                logging.info(f'No responses found for {conds}')
                continue

            decode_dfs,cms = run_decoding(
                batch_responses,  self.x_ser, decoding_window, pips2decode, overwrite=True,
                **self.plot_config[dec_tag].get('decoding_kwargs', {})
            )
            batch_decoding_dfs.append(decode_dfs)

            batch_cms.append(cms)

        self.aggregate_decoding_df = pd.concat(batch_decoding_dfs, axis=0)
        batch_cms = [cm for cm in batch_cms if cm.ndim==3]
        self.cms =np.concatenate(batch_cms, axis=0)
        self.decoder_name = dec_tag

        if kwargs.get('df_save_path'):
            df_save_path = Path(kwargs.get('df_save_path'))
            cm_save_path = df_save_path.with_name(df_save_path.name.replace('df.h5', 'cm.npy'))
            self.aggregate_decoding_df.to_hdf(df_save_path,'df')
            np.save(cm_save_path, self.cms)

    def decoding_ttest(self, decoder_name, key1, key2, **ttest_kwargs):

        data_acc = self.aggregate_decoding_df[f'{decoder_name}_{key1}_accuracy'].dropna().values
        if isinstance(key2, float):
            ttest_res = ttest_1samp(data_acc,key2,**ttest_kwargs)
        else:
            shuff_acc = self.aggregate_decoding_df[f'{decoder_name}_{key2}_accuracy'].dropna().values
            ttest_res = ttest_ind(data_acc, shuff_acc, **ttest_kwargs)
        self.ttest_res[f'{decoder_name}_{"_vs_".join([key1,key2])}'] = ttest_res
        logging.info(f'{decoder_name} ttest results: {ttest_res}')

    def plot_mean_ts(self,figdir:Path, pips=None, **kwargs):

        if pips is None:
            pips = list(self.concatenated_event_responses.keys())

        ts_plot = plt.subplots()
        if kwargs.get('ts_plot_window') is not None:
            plot_t_idxs = [np.where(self.x_ser==t)[0][0]
                           for t in kwargs.get('ts_plot_window')]
            plot_x_ser = self.x_ser[plot_t_idxs[0]:plot_t_idxs[1]]
        else:
            plot_t_idxs = [0,self.x_ser.shape[0]]
            plot_x_ser = self.x_ser

        if kwargs.get('plot_smoothed_ts', True):
            event_mean_dict = self.smoothed_concatenated_responses
        else:
            event_mean_dict = self.concatenated_event_responses
        print(kwargs)

        colours = kwargs.get('plot_cols',[f'C{i}' for i in range(len(pips))])
        name_date_df = pd.DataFrame([(sessname,
                                      sessname.split('_')[0],extract_date(sessname))
                                     for sessname in self.concatenated_sessnames],
                                    columns=['sess','name','date'])
        for pip,col in zip(pips,colours):
            event_dict_df = pd.DataFrame(event_mean_dict[pip],index=pd.MultiIndex.from_frame(name_date_df))
            mean_resp = event_dict_df.groupby('name').mean().mean(axis=0)[plot_t_idxs[0]:plot_t_idxs[1]]
            sem_resp = event_dict_df.groupby('name').mean().sem(axis=0)[plot_t_idxs[0]:plot_t_idxs[1]]
            # sem_resp = np.percentile(event_mean_dict[pip],[2.5,97.5],axis=0)[:,plot_t_idxs[0]:plot_t_idxs[1]]
            ts_plot[1].plot(plot_x_ser,mean_resp, label=pip, c=col,lw=1)
            # ts_plot[1].fill_b1etween(plot_x_ser, mean_resp-sem_resp,mean_resp+sem_resp, alpha=0.1)
            # ts_plot[1].fill_between(plot_x_ser, sem_resp[0],sem_resp[1], alpha=0.1)
            plot_shaded_error_ts(ts_plot[1],plot_x_ser,mean_resp,sem_resp,fc=col, alpha=0.1)

        if any(['A' in pip for pip in pips]):
            format_axis(ts_plot[1],vlines=[0],
                        vspan=[[t,t+0.15] for t in np.arange(0, min([1,plot_x_ser.max()]), 0.25)])
        else:
            format_axis(ts_plot[1],vlines=[0])

        ts_plot[1].legend()
        ts_plot[0].set_size_inches(kwargs.get('figsize',(2,1.5)))
        ts_plot[0].set_layout_engine('tight')
        ts_plot[0].show()
        ts_plot[0].savefig(figdir / f'{"_".join(pips)}_mean_ts.pdf')

        plot_info = {'n_units': [e.shape[0] for e in event_mean_dict.values()],
                     'sessnames': np.unique(self.concatenated_sessnames).tolist(),
                     'n_sessions': len(np.unique(self.concatenated_sessnames)),
                     'names': np.unique([e.split('_')[0] for e in self.concatenated_sessnames]).tolist(),
                     'n_names': np.unique([e.split('_')[0] for e in self.concatenated_sessnames]).shape[0]}
        with open(figdir / f'{"_".join(pips)}_mean_ts_info.json', 'w') as f:
            json.dump(plot_info, f)

        self.plots[f'{"_".join(pips)}_mean_ts'] = ts_plot

    def plot_sorted_psth_mat(self, pips=None, plot_kwargs=None):

        if pips is None:
            pips = list(self.concatenated_event_responses.keys())

        if plot_kwargs is None:
            plot_kwargs = {}

        for pip in pips:
            sorted_resp_mat = self.event_mats_4_sorting[pip][self.peak_ts_by_pips[pip].argsort()]
            psth_plot = plot_sorted_psth_matrix(sorted_resp_mat,self.x_ser, pip,  **plot_kwargs)
            t_max = min(1, plot_kwargs.get('plot_window',self.window)[1])
            format_axis(psth_plot[1][1], vlines=([0] if 'A' not in pip else np.arange(0,t_max,0.25).tolist()),
                        ylabel='', xlabel='')
            psth_plot[1][1].set_xticks([])
            format_axis(psth_plot[1][0], vlines=[0])
            try:
                format_axis(psth_plot[2].ax)
            except:
                pass

            self.plots[f'{pip}_sorted_psth'] = psth_plot


    def plot_decoder_boxplot(self,decoding_figdir:Path,decs2plot=None):

        # --- Boxplot plotting kwargs ---
        boxplot_kwargs = dict(
            widths=0.5,
            patch_artist=True,
            # showmeans=True,
            showfliers=False,
            medianprops=dict(lw=1),
            # meanline=True,
            meanprops=dict(mfc='k'),
            boxprops=dict(lw=0.5),
            whiskerprops=dict(lw=0.5),
            capprops=dict(lw=0.5),
            # whis=[5,95]
        )
        # Plot all decoding results on one figure
        stim_decoding_plot = plt.subplots()
        labels = []
        scatter_points = False  # Set to True to scatter individual points

        if decs2plot is None:
            all_dec_cols = self.aggregate_decoding_df.columns.tolist()
            all_data_dec_cols = [col for col in all_dec_cols if 'data_accuracy' in col]
        else:
            if isinstance(decs2plot, str):
                decs2plot = [decs2plot]
            all_data_dec_cols = [f'{dec}_data_accuracy' for dec in decs2plot]
        all_shuffle_dec_cols = [col.replace('data','shuffled') for col in all_data_dec_cols]

        for dec_i, (data_name, shuff_name) in enumerate(zip(all_data_dec_cols, all_shuffle_dec_cols)):
            data_acc = self.aggregate_decoding_df[data_name].dropna().values
            shuff_acc = self.aggregate_decoding_df[shuff_name].dropna().values
            lbls = [data_name.replace('_data_accuracy',f'\ndata: n {len(data_acc)}'),
                    shuff_name.replace('_shuffled_accuracy',f'\nshuffle: n {len(data_acc)}' )]
            box = stim_decoding_plot[1].boxplot(
                [data_acc, shuff_acc],
                labels=lbls,
                positions=np.array([-0.3,0.3])+dec_i*len(lbls),
                **boxplot_kwargs
            )
            # change box colors
            for patch in box['boxes']:
                patch.set_facecolor('white')

            labels.extend(lbls)

        stim_decoding_plot[1].set_ylabel('Decoding accuracy')
        format_axis(stim_decoding_plot[1], hlines=[0.5],)
        # stim_decoding_plot[1].set_xticks(np.arange(len(all_data_dec_cols)))
        stim_decoding_plot[1].set_xticks([])

        stim_decoding_plot[0].set_size_inches(0.9*len(all_data_dec_cols)+0.15,1.75)
        stim_decoding_plot[0].set_layout_engine('tight')
        stim_decoding_plot[0].show()
        stim_decoding_plot[0].savefig(decoding_figdir / f'{self.decoder_name}_decoding_accuracy.pdf')

    def plot_confusion_matrix(self,decoding_figdir:Path,cm_config):

        cm_plot = plot_aggr_cm(self.cms, **cm_config,)

        cm_plot[0].show()
        cm_plot[0].savefig(decoding_figdir / f'{self.decoder_name}_cm.pdf')

    def scatter_unit_means(self,pips,diff_window, figdir,**kwargs):

        assert len(pips)==2
        event_mean_dict = self.concatenated_event_responses

        plot_t_idxs = [np.where(self.x_ser==t)[0][0]
                       for t in diff_window]

        unit_resps = [event_mean_dict[pip][:,plot_t_idxs[0]:plot_t_idxs[1]].max(axis=1)
                     for pip in pips]

        # plot scatter
        cond_max_scatter = plt.subplots()
        cond_max_scatter[1].scatter(*unit_resps,alpha=0.02,fc='#1f76b2ff',ec='#1f76b2ff',lw=0.01)
        cond_max_scatter[1].set_xlim(*np.percentile(unit_resps,[1,99]))
        cond_max_scatter[1].set_ylim(*np.percentile(unit_resps,[1,99]))
        format_axis(cond_max_scatter[1])
        cond_max_scatter[1].locator_params(axis='both', nbins=6)
        cond_max_scatter[1].set_xlabel('frequent mean response')
        cond_max_scatter[1].set_ylabel('rare mean response')
        # plot unity line
        cond_max_scatter[1].plot(cond_max_scatter[1].get_xlim(), cond_max_scatter[1].get_ylim(), ls='--', c='k')

        cond_max_scatter[0].set_size_inches(2, 2)
        cond_max_scatter[0].show()
        cond_max_scatter[0].savefig(figdir / f'unit_resps_scatter{"_".join(pips)}.pdf')
        cond_max_scatter[0].savefig(figdir / f'unit_resps_scatter{"_".join(pips)}.svg')

        # plot hist
        cond_diff_by_unit_plot = plt.subplots()
        cond_diffs_by_unit = unit_resps[0]-unit_resps[1]
        bins2use = np.histogram(cond_diffs_by_unit, bins='fd', density=False)
        cond_diff_by_unit_plot[1].hist(cond_diffs_by_unit, bins=bins2use[1], density=False, alpha=0.9, fc='#a8cfe2ff',
                                       ec='k', lw=0.05)
        cond_diff_by_unit_plot[1].set_xlim(*np.percentile(cond_diffs_by_unit, [1,99]))

        format_axis(cond_diff_by_unit_plot[1], vlines=[0], lw=0.5,
                    ls='--')
        cond_diff_by_unit_plot[1].set_ylabel('Frequency')
        cond_diff_by_unit_plot[1].set_xlabel('\u0394firing rate (rare - frequent)')

        cond_diff_by_unit_plot[0].set_size_inches(2, 2)
        cond_diff_by_unit_plot[0].set_layout_engine('tight')
        cond_diff_by_unit_plot[0].show()
        cond_diff_by_unit_plot[0].savefig(figdir / f'unit_resps_hist_by_unit{"_".join(pips)}.pdf')

    def pca_pseudo_pop(self, pca_name: str, pips=None, standardise=True):
        if pips is None:
            pips = list(self.concatenated_event_responses.keys())

        dict_for_pca = {'by_class': {pip: self.concatenated_event_responses[pip] for pip in pips}}
        pca = PopPCA(dict_for_pca)

        pca.get_trial_averaged_pca(standardise=standardise)
        pca.get_projected_pca_ts(standardise=standardise)
        self.pca[pca_name] = pca

    def plot_3d_pca(self, pca_name, pca_comps_2plot, figdir, pca_kwargs):
        pca = self.pca[pca_name]
        pca.plot_3d_pca_ts('by_class', self.window, x_ser=self.x_ser,
                           pca_comps_2plot=pca_comps_2plot,
                           **pca_kwargs['plot_kwargs'])
        pca.proj_3d_plot[1][0].get_legend().remove()
        if pca_kwargs['fig_kwargs'].get('figsize', None) is not None:
            pca.proj_3d_plot[0].set_size_inches(*pca_kwargs['fig_kwargs']['figsize'])
        # save 3d plot
        pca.proj_3d_plot[0].savefig(figdir/f'{pca_name}_pca_{"_".join(list(map(str,pca_comps_2plot)))}.pdf')
        # pca.plot_3d_pca_ts_plotly('by_class', self.window,pca_comps_2plot=pca_comps_2plot, x_ser=self.x_ser,)

    def plot_2d_pca(self, pca_name, pca_comps_2plot, figdir, pca_kwargs):
        pca = self.pca[pca_name]
        pca.plot_2d_pca_ts('by_class', self.window, x_ser=self.x_ser,
                           pca_comps_2plot=pca_comps_2plot,
                           **pca_kwargs['plot_kwargs'])
        pca.proj_2d_plot[1].get_legend().remove()
        format_axis(pca.proj_2d_plot[1])
        if pca_kwargs['fig_kwargs'].get('figsize', None) is not None:
            pca.proj_2d_plot[0].set_size_inches(*pca_kwargs['fig_kwargs']['figsize'])
        # save 3d plot
        pca.proj_2d_plot[0].savefig(figdir / f'{pca_name}_2d_pca_{"_".join(list(map(str, pca_comps_2plot)))}.pdf')

    def scatter_pca(self, pca_name,t_s, pca_comps_2plot, figdir, pca_kwargs):
        pca = self.pca[pca_name]
        pca.scatter_pca_points('by_class',t_s, x_ser=self.x_ser,pca_comps_2plot=pca_comps_2plot,**pca_kwargs['scatter_kwargs'])
        pca.scatter_plot[0].show()
        pca.scatter_plot[0].savefig(figdir/f'{pca_name}_scatter_{"_".join(list(map(str,pca_comps_2plot)))}.pdf')


if __name__ == '__main__':
    main()
