import argparse
import pickle
import platform
from copy import copy
from multiprocessing import Pool
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import ttest_ind
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm

# For Qt


from behviour_analysis_funcs import get_all_cond_filts
from sess_dataclasses import get_predictor_from_psth
from decoding_funcs import Decoder
from io_utils import posix_from_win, load_sess_pkl
from pupil_analysis_funcs import process_pupil_td_data
from spike_time_utils import zscore_by_trial


def load_aggregate_td_df(session_topolgy: pd.DataFrame,home_dir:Path,td_df_query=None) -> pd.DataFrame:
    from behviour_analysis_funcs import get_main_sess_td_df
    # get main sess pattern
    td_path_pattern = 'data/Dammy/<name>/TrialData'
    if 'tdata_file' not in session_topolgy.columns:
        td_paths = [Path(sess_info['sound_bin'].replace('_SoundData', '_TrialData')).with_suffix('.csv').name
                    for _,sess_info in session_topolgy.iterrows()]
        abs_td_paths = [home_dir / td_path_pattern.replace('<name>', sess_info['name']) / td_path
                        for (td_path, (_, sess_info)) in zip(td_paths, session_topolgy.iterrows())]

    else:
        abs_td_paths = session_topolgy['tdata_file'].to_list()
    sessnames = [Path(sess_info['sound_bin'].replace('_SoundData', '')).stem
                 for _, sess_info in session_topolgy.iterrows()]
    abs_td_paths = [home_dir/posix_from_win(td_path,'/nfs/nhome/live/aonih') if isinstance(td_path,(str,Path)) else None for td_path in abs_td_paths]
    td_dfs = {sessname: get_main_sess_td_df(_main_sess_td_name=abs_td_path,_home_dir=home_dir)[0]
              for sessname, abs_td_path in zip(sessnames, abs_td_paths)
              if abs_td_path is not None
              }

    # td_dfs = {sessname:pd.read_csv(abs_td_path) for sessname, abs_td_path in zip(sessnames,abs_td_paths)
    #           if abs_td_path.is_file()}
    td_df = pd.concat(list(td_dfs.values()),keys=td_dfs.keys(),names=['sess'],axis=0)
    if td_df_query:
        td_df = td_df.query(td_df_query)
    return td_df


def load_aggregate_sessions(pkl_paths, td_df_query=None):
    sess_objs = []
    for pkl_path in tqdm(pkl_paths, total=len(pkl_paths), desc='loading sessions'):
        print(f'loading {pkl_path}')
        sess_obj = load_sess_pkl(pkl_path)
        if sess_obj is None:
            print(f'{pkl_path} error')
            continue
        if sess_obj.td_df is None:
            print(f'{pkl_path} no td_df')
            continue
        if td_df_query is not None:
            if sess_obj.td_df.query(td_df_query).empty:
                print(f'{pkl_path} no td_df query')
                continue
        sess_objs.append(sess_obj)
    return {e.sessname: e for e in sess_objs}


def aggregate_event_responses(sessions: dict, events=None, events2exclude=None, window=(0, 0.25),
                              pred_from_psth_kwargs=None,zscore_by_trial_flag=False ):

    events_by_session = [list(sess.sound_event_dict.keys()) for sess in sessions.values()]

    if events is None:
        raise NotImplementedError

    if events:
        sessions2use = [sess for sess, s_events in zip(sessions, events_by_session) if
                        all(e in s_events for e in events)]
    else:
        sessions2use = sessions

    if events:
        for sess in copy(sessions2use):
            if not all([e in list(sessions[sess].sound_event_dict.keys()) for e in events]):
                sessions.pop(sess)
                sessions2use.remove(sess)

    # if events2exclude:
    #     common_events = [e for e in common_events if e not in events2exclude]


    event_responses_across_sessions = {sessname: {e: get_predictor_from_psth(sessions[sessname], e, [-2, 3], window,
                                                                             **pred_from_psth_kwargs if pred_from_psth_kwargs else {})
                                                  for e in events}
                                       for sessname in
                                       tqdm(sessions2use, total=len(sessions2use), desc='Getting event responses')}
    if zscore_by_trial_flag:
        event_responses_across_sessions = zscore_by_trial(event_responses_across_sessions)
    return event_responses_across_sessions


def aggregate_event_features(sessions: dict, events=None, events2exclude=None):
    events_by_session = [list(sess.sound_event_dict.keys()) for sess in sessions.values()]
    # print(f'common events = {common_events}')

    if events is None:
        raise NotImplementedError

    if events:
        sessions2use = [sess for sess, s_events in zip(sessions, events_by_session) if all(e in s_events for e in events)]
    else:
        sessions2use = sessions

    if events:
        for sess in copy(sessions2use):
            if not all([e in list(sessions[sess].sound_event_dict.keys()) for e in events]):
                sessions.pop(sess)
                sessions2use.remove(sess)
    if len(sessions2use) == 0:
        return {}

    event_features_across_sessions = {sessname: {e: {'times': sessions[sessname].sound_event_dict[e].times.values,
                                                     'trial_nums': sessions[sessname].sound_event_dict[e].trial_nums, }
                                                 for e in events}
                                      for sessname in
                                      tqdm(sessions2use, total=len(sessions2use), desc='Getting event features')}
    for sessname in tqdm(event_features_across_sessions, total=len(event_features_across_sessions), desc='Adding td_df'):
        sess_td_df = sessions[sessname].td_df
        sess_td_df.index = sess_td_df.reset_index(drop=True).index+1
        for e in event_features_across_sessions[sessname]:
            trial_nums = event_features_across_sessions[sessname][e]['trial_nums']
            event_features_across_sessions[sessname][e]['td_df'] = sess_td_df.query('index in @trial_nums')
    return event_features_across_sessions


def concatenate_responses_by_td(responses_by_sess: dict, features_by_sess: dict, td_df_query=None):
    grouped_by_td = {}
    common_events = set.intersection(*[set(list(e.keys())) for e in responses_by_sess.values()])
    for event in tqdm(common_events, total=len(common_events), desc='Concatenating responses'):
        event_across_sess = [sess_responses[event][:len(sess_features[event]['td_df'])] for sess_responses,sess_features
                             in zip(responses_by_sess.values(), features_by_sess.values())]
        event_idxbool = [sess[event]['td_df'].eval(td_df_query).values for sess in features_by_sess.values()]
        try:
            grouped_by_td[event] = np.concatenate([np.nanmean(sess_e_responses[sess_idx_bool], axis=0)
                                                   for sess_e_responses, sess_idx_bool in
                                                   tqdm(zip(event_across_sess, event_idxbool), total=len(event_across_sess),
                                                        desc='Concatenating features')], axis=0)
        except IndexError:
            print(f'IndexError for {event}')
            grouped_by_td[event] = None

    return grouped_by_td


def group_ephys_resps(event_name:str,responses_by_sess: dict, features_by_sess: dict, td_df_query=None,
                      trial_nums2use=None,sess_list=None):
    resps_across_sess = {}
    trial_nums = {}
    for sess in sess_list if sess_list else responses_by_sess:
        if td_df_query:
            sess_resp_idxs = features_by_sess[sess][event_name]['td_df'].eval(td_df_query).values
        elif trial_nums2use is not None:
            sess_resp_idxs = np.isin(features_by_sess[sess][event_name]['trial_nums'], trial_nums2use)
        else:
            raise ValueError('must specify td_df_query or trial_nums2use')
        # print(sess_resp_idxs)
        if not np.any(sess_resp_idxs) or sess_resp_idxs.sum()<10:
            print(f'{sess} has not enough responses')
            # resps_across_sess[sess] = None
            continue
        sess_resps = responses_by_sess[sess][event_name]
        # print(f'{sess} has {len(sess_resps)} responses, {len(sess_resp_idxs)} indices')
        resps_across_sess[sess] = sess_resps[:len(sess_resp_idxs)][sess_resp_idxs]
        trial_nums[sess] = features_by_sess[sess][event_name]['trial_nums'][:len(sess_resp_idxs)][sess_resp_idxs]
    return resps_across_sess,trial_nums


def group_responses_by_pip_prop(responses_by_sess: dict, pip_desc:dict, properties=None,concatenate_flag=True):
    pip_groups_by_prop = {prop: {val: [pip for pip in pip_desc[prop].keys() if pip_desc[prop][pip] == val]
                                 for val in np.unique(list(pip_desc[prop].values()))} for prop in pip_desc}
    if not properties:
        properties = list(pip_groups_by_prop.keys())
    if concatenate_flag:
        responses_by_prop = {prop: {val: np.concatenate([responses_by_sess[pip] for pip in pip_groups_by_prop[prop][val]
                                                         if pip in responses_by_sess],
                                                        axis=0)
                                    for val in pip_groups_by_prop[prop]} for prop in
                             tqdm(properties, total=len(pip_groups_by_prop), desc='Concatenating responses')}
    else:
        responses_by_prop = {prop: {val: {pip:responses_by_sess[pip] for pip in pip_groups_by_prop[prop][val]
                                          if pip in responses_by_sess}
                                    for val in pip_groups_by_prop[prop]} for prop in
                             tqdm(properties, total=len(pip_groups_by_prop), desc='Concatenating responses')}

    return responses_by_prop


def get_responses_by_pip_and_condition(pip_names, event_responses, event_features, conds, cond_filters,
                                       zip_pip_conds=False):
    """
    Returns a dictionary with keys '{pip}_{cond}' and values as dicts: {sessname: response array}.
    Args:
        pip_names (list): List of pip event names (e.g. ['A-0', 'B-0']).
        event_responses (dict): Nested dict of session -> event -> response arrays.
        event_features (dict): Nested dict of session -> event -> features (e.g. 'td_df').
        conds (list): List of condition names (must be keys in cond_filters).
        cond_filters (dict): Dictionary mapping condition names to query strings.
    Returns:
        dict: {f'{pip}_{cond}': {sessname: response array}}
    """
    responses_by_pip_cond = {}

    if zip_pip_conds:
        pips2use = [f'{pip}_{cond}' for pip,cond in zip(pip_names, conds)]
    else:
        pips2use = [f'{pip}_{cond}' for pip in pip_names for cond in conds]
    for sessname in event_responses:
        responses_by_pip_cond[sessname] = {}
        for key in pips2use:
            pip = key.split('_')[0]
            responses_by_pip_cond[sessname][key] = np.array([])
            # Check if pip and cond exist for this session
            if pip not in event_responses[sessname]:
                continue
            if pip not in event_features[sessname]:
                continue

            cond = '_'.join(key.split('_')[1:])
            cond_query = cond_filters[cond]

            if sessname not in list(event_features):
                print(f'Session {sessname} not in event features. Skipping...')
                continue

            td_df = event_features[sessname][pip].get('td_df', None)
            if td_df is None:
                continue
            try:
                trial_mask = td_df.eval(cond_query)
            except AssertionError:
                continue
            # Only keep if enough trials
            if trial_mask.sum() < 1:
                continue
            n_events = min(len(event_responses[sessname][pip]), len(trial_mask))
            # responses_by_pip_cond[key][sessname] = event_responses[sessname][pip][:n_events][trial_mask.values[:n_events]]
            responses_by_pip_cond[sessname][key] = event_responses[sessname][pip][:n_events][trial_mask.values[:n_events]]

    # clean up dict
    for sessname in list(responses_by_pip_cond.keys()):
        if len(responses_by_pip_cond[sessname]) == 0:
            responses_by_pip_cond.pop(sessname)
    return responses_by_pip_cond


def decode_responses(predictors, features, model_name='logistic', dec_kwargs=None):
    decoder = {dec_lbl: Decoder(np.vstack(predictors), np.hstack(features), model_name=model_name)
               for dec_lbl in ['data', 'shuffled']}
    dec_kwargs = dec_kwargs or {}
    [decoder[dec_lbl].decode(dec_kwargs=dec_kwargs | {'shuffle': shuffle_flag}, parallel_flag=False,
                             )
     for dec_lbl, shuffle_flag in zip(decoder, [False, True])]

    return decoder


def plot_aggr_cm(pip_cms, im_kwargs=None, **cm_kwargs):
    cm_plot = ConfusionMatrixDisplay(np.mean(pip_cms,axis=0) if pip_cms.ndim==3 else pip_cms,
                                     display_labels=cm_kwargs.get('labels',list(range(pip_cms.shape[-1]))))
    cm_plot.plot(cmap=cm_kwargs.get('cmap','bwr'),
                 include_values=cm_kwargs.get('include_values',False),colorbar=cm_kwargs.get('colorbar',True),
                 im_kw=im_kwargs if im_kwargs is not None else {})

    cm_plot.ax_.invert_yaxis()
    cm_plot.ax_.set_xlabel('')
    cm_plot.ax_.set_ylabel('')
    cm_plot.figure_.set_size_inches(cm_kwargs.get('figsize',(2,2)))
    # cm_plot.figure_.set_layout_engine('constrained')

    return cm_plot.figure_, cm_plot.ax_


def decode_over_sliding_t(resps_dict: dict, window_s, resp_window, pips_as_ints_dict, pips2decode: list,
                          animals_to_use:list=None):
    resp_width = list(resps_dict.values())[0][pips2decode[0]].shape[-1]
    resp_x_ser = np.round(np.linspace(resp_window[0], resp_window[1], resp_width), 2)
    window_size = int(np.round(window_s / (resp_x_ser[1] - resp_x_ser[0])))

    dec_res_dict = {}
    bad_dec_sess = set()
    for sessname in tqdm(list(resps_dict.keys()), total=len(resps_dict),
                         desc='decoding across sessions'):
        # if not any(e in sessname for e in ['DO79', 'DO81']):
        if animals_to_use:
            if not any(e in sessname for e in animals_to_use):
                continue

        for t in tqdm(range(resp_width - window_size), total=resp_width - window_size, desc='decoding across time'):
            if 'A-0' in pips2decode:
                xys = [(
                    resps_dict[sessname][pip][150:200][::3] if pip.split('-')[1] == '0' else
                    resps_dict[sessname][pip],
                    np.full_like(resps_dict[sessname][pip][:, 0, 0], pips_as_ints_dict[pip]))
                    for pip in pips2decode]
            else:
                xys = [(resps_dict[sessname][pip],
                        np.full_like(resps_dict[sessname][pip][:, 0, 0], pips_as_ints_dict[pip]))
                       for pip in pips2decode]
            ys = [np.full(xy[0].shape[0], pips_as_ints_dict[pip]) for xy, pip in zip(xys, pips2decode)]
            xs = np.vstack([xy[0][:, :, t:t + window_size].mean(axis=-1) for xy in xys])
            # xs = [xy[0][:,:,15:].mean(axis=-1) for xy in xys]
            ys = np.hstack(ys)
            # if np.unique(ys).shape[0] < len(patt_is):
            #     continue
            try:
                dec_res_dict[f'{sessname}-{t}:{t + window_size}s'] = decode_responses(xs, ys, n_runs=50,
                                                                                      dec_kwargs={'cv_folds': 10})
            except ValueError:
                print(f'{sessname} failed')

                bad_dec_sess.add(sessname)
                continue
    # [decoders_dict[dec_name]['data'].plot_confusion_matrix([f'{pip}-{pip_i}' for pip in 'D' for pip_i in patt_is])
    #  for dec_name in decoders_dict.keys()]
    sess2use = [dec_name.split('-')[0] for dec_name in dec_res_dict.keys()]
    norm_dev_accs_ts_dict = {
        sessname: {t: np.mean(dec_res_dict[f'{sessname}-{t}:{t + window_size}s']['data'].accuracy)
                   for t in range(resp_width - window_size)}
        for sessname in sess2use if sessname not in bad_dec_sess}

    norm_dev_accs_ts_df = pd.DataFrame(norm_dev_accs_ts_dict).T
    norm_dev_accs_ts_df.columns = np.round(resp_x_ser[window_size:], 2)

    return norm_dev_accs_ts_df, norm_dev_accs_ts_dict


def predict_from_responses(dec_model,responses):
    return dec_model.predict(responses)


def predict_over_sliding_t(dec_model,resps_dict,pips2predict,window_s,resp_window):

    resp_width = list(resps_dict.values())[0][pips2predict[0]].shape[-1]
    resp_x_ser = np.round(np.linspace(resp_window[0], resp_window[1], resp_width), 2)
    window_size = int(np.round(window_s / (resp_x_ser[1] - resp_x_ser[0])))

    responses = []
    for pips2predict in pips2predict:
        pip_resp = resps_dict[pips2predict]
        if pip_resp.ndim == 3:
            pass

    return dec_model.predict(responses[:,:,window_size:])



def load_or_generate_event_responses(args, **kwargs):
    sys_os = platform.system().lower()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    home_dir = Path(config[f'home_dir_{sys_os}'])
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])

    # Load plot config if provided
    plot_config = {}
    if args.plot_config_path is not None:
        plot_config_path = Path(args.plot_config_path)
        if plot_config_path.is_file():
            with open(plot_config_path, 'r') as f:
                plot_config = yaml.safe_load(f)

    # Get config options
    session_topology_paths = plot_config.get('session_topology_paths',
                                             [r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_ephys_2401.csv'])
    session_topology_paths = [ceph_dir/ posix_from_win(sess_top_path) for sess_top_path in session_topology_paths]

    sess_top_query = plot_config.get('sess_top_query', 'sess_order=="main" & date >= 240219')
    td_df_query = plot_config.get('td_df_query', 'Stage==3')
    cond_filter = plot_config.get('cond_filter', None)
    window = plot_config.get('window', [-0.25, 1])
    by_animal = plot_config.get('by_animals', True)
    batch_size = plot_config.get('batch_size', 5)
    use_multiprocess = args.multiprocess
    skip_batches_if_exist = plot_config.get('skip_batches_if_exist', False)

    pkldir = ceph_dir / posix_from_win(args.pkldir)
    event_responses_pkl_path = Path(args.event_responses_pkl)
    event_features_path = event_responses_pkl_path.with_stem(event_responses_pkl_path.stem + '_features')

    # # Load full dataset if already cached
    # # if event_responses_pkl_path.exists() and not args.overwrite:
    # if event_responses_pkl_path.exists():
    #     # with open(event_responses_pkl_path, 'rb') as f:
    #     #     event_responses = pickle.load(f)
    #     event_responses = joblib.load(event_responses_pkl_path)
    #     # with open(event_features_path, 'rb') as f:
    #     #     event_features = pickle.load(f)
    #     event_features = joblib.load(event_features_path)
    #     return event_responses, event_features

    # Load session topology
    session_topology = pd.concat([pd.read_csv(sess_top_path) for sess_top_path in session_topology_paths])
    all_sess_info = session_topology.query(sess_top_query)

    # Determine td_df_query if cond_filter used
    cond_filters = get_all_cond_filts()
    all_sess_td_df = load_aggregate_td_df(all_sess_info, home_dir,)
    if td_df_query:
        all_sess_td_df = all_sess_td_df.query(td_df_query)
    if cond_filter:
        all_sess_td_df = all_sess_td_df.query(cond_filters[cond_filter])

    sessions2use = sorted(all_sess_td_df.index.get_level_values('sess').unique().tolist())
    print(sessions2use)

    # log missing pkls:
    sess_in_pkl_dir = [e.stem for e in list(pkldir.iterdir())]
    missing_sessions = [sess for sess in sessions2use if sess not in sess_in_pkl_dir]
    # save to txt file
    pd.Series(missing_sessions).to_csv(f'missing_sessions_{cond_filter}.csv')
    sessions2use = sorted([sess for sess in sessions2use if not any([e in sess for e in ['DO82','DO97_250530']])])

    # Group sessions
    if by_animal:
        from collections import defaultdict
        grouped_sess = defaultdict(list)
        for sess in sessions2use:
            animal = sess.split('_')[0]
            grouped_sess[animal].append(sess)
        batches = list(grouped_sess.values())
        batch_ids = list(grouped_sess.keys())
    else:
        batches = [sessions2use[i:i+batch_size] for i in range(0, len(sessions2use), batch_size)]
        batch_ids = [f'{event_responses_pkl_path.stem}_{i}' for i in range(len(batches))]

    # --- Replace local process_batch with top-level version for multiprocessing ---
    if use_multiprocess:
        with Pool(processes=min(len(batches), 12)) as pool:
            batch_outputs = list(tqdm(
                pool.starmap(
                    process_batch,
                    [(batch, batch_id, event_responses_pkl_path, pkldir, skip_batches_if_exist, window)
                     for batch, batch_id in zip(batches, batch_ids)]
                ),
                total=len(batches),
                desc="Processing Batches"
            ))
    else:
        batch_outputs = [
            process_batch(batch, batch_id, event_responses_pkl_path, pkldir, skip_batches_if_exist, window,
                          plot_config['pips_2_plot'])
            for batch, batch_id in tqdm(zip(batches, batch_ids), total=len(batches), desc="Processing Batches")
        ]

    # Remove empty batches
    batch_outputs = [b for b in batch_outputs if b is not None]
    batch_event_responses = [b[0] for b in batch_outputs]

    batch_event_features = [b[1] for b in batch_outputs]

    assert len(batch_event_responses) == len(batch_event_features)
    # Merge batch outputs
    all_keys = set()
    for b in batch_event_responses:
        all_keys.update(b.keys())

    merged_event_responses = {}
    for b in batch_event_responses:
        for k in b:
            merged_event_responses[k] = b[k]

    # Save merged results
    joblib.dump(merged_event_responses,event_responses_pkl_path.with_suffix('.joblib'))

    # Regenerate features from merged sessions
    merged_event_features = {}
    for b in batch_event_features:
        for k in b:
            merged_event_features[k] = b[k]

    with open(event_features_path, 'wb') as f:
        pickle.dump(merged_event_features, f)

    return merged_event_responses, merged_event_features


def process_batch(batch_sess: list, batch_id, event_responses_pkl_path, pkldir, skip_batches_if_exist, window,
                  events):

    batch_file = event_responses_pkl_path.parent / f"batch_{batch_id}.joblib"

    # if batch_file.exists() and skip_batches_if_exist:
    #     print(f'Skipping batch {batch_id}')
    #     with open(batch_file, 'rb') as f:
    #         return pickle.load(f)

    pkls = [p for p in pkldir.glob('*.pkl') if p.stem in batch_sess]
    if len(pkls) == 0:
        return None
    sessions = load_aggregate_sessions(pkls)
    [sessions.pop(k) for k in list(sessions.keys())
     if not all([_pip in sessions[k].sound_event_dict for _pip in events])]
    if len(sessions) == 0:
        return None
    for sessname, sess in sessions.items():
        process_pupil_td_data(sessions,sessname, {})


    event_responses = aggregate_event_responses(
        sessions, events=events,
        events2exclude=['trial_start'], window=window,zscore_by_trial_flag=False,
        pred_from_psth_kwargs={'use_unit_zscore': False, 'use_iti_zscore': False, 'baseline': 0,
                               'mean': None, 'mean_axis': 0}
    )
    # with open(batch_file, 'wb') as f:
    #     pickle.dump(event_responses, f)
    joblib.dump(event_responses, batch_file.with_suffix('.joblib'))

    event_features = aggregate_event_features(
        sessions, events=events,
        events2exclude=['trial_start']
    )

    return event_responses, event_features


def run_decoding(event_responses, x_ser, decoding_windows, pips2decode, cache_path=None, overwrite=False, **kwargs):
    """
    Run decoding analysis for specified pip pairs and return a DataFrame with one row per session,
    columns for each decoding's data/shuff accuracy.

    Args:
        event_responses (dict): Nested dict of session -> event -> response arrays.
        animals (list): List of animal names to include.
        x_ser (np.ndarray): Time axis for windowing.
        decoding_windows (list): List of [start, end] windows for decoding.
        pips2decode (list): List of [pip1, pip2] pairs to decode.
        cache_path (Path or str, optional): Path to pickle file for caching results.
        overwrite (bool): If True, recompute even if cache exists.

    Returns:
        pd.DataFrame: Index sessname, columns for each decoding's data/shuff accuracy.
    """


    # if cache_path is not None and Path(cache_path).is_file() and not overwrite:
    #     with open(cache_path, 'rb') as f:
    #         all_results_df = pickle.load(f)
    #     return all_results_df

    records = []
    cms = []
    for sessname in tqdm(event_responses.keys(), desc='decoding sessions', total=len(event_responses)):
        session_events = event_responses[sessname]
        record = {'sess': sessname, 'name': sessname.split('_')[0]}
        for pips, dec_wind in zip(pips2decode, decoding_windows):
            dec_sffx = "_vs_".join(pips)
            if not all(p in session_events for p in pips):
                record[f'{dec_sffx}_data_accuracy'] = np.nan
                record[f'{dec_sffx}_shuffled_accuracy'] = np.nan
                continue

            xs_list = [session_events[pip] for pip in pips]
            idx_4_decoding = [np.where(x_ser == t)[0][0] for t in dec_wind]
            xs = np.vstack([x[:, :, idx_4_decoding[0]:idx_4_decoding[1]].mean(axis=-1) for x in xs_list])
            ys = np.hstack([np.full(x.shape[0], ci) for ci, x in enumerate(xs_list)])

            dec_kwargs = kwargs.get('dec_kwargs', {})
            if kwargs.get('train_split_by_cond'):
                conds = list(set(['_'.join(pip.split('_')[1:]) for pip in pips]))
                n_conds1 = session_events[f'{pips[0]}'].shape[0]
                print(f'Debugging: n_conds1 = {n_conds1}, '
                      f'len all = {[session_events[f"{p}"].shape[0] for p in pips]}')
                dec_kwargs['pre_split'] = n_conds1*(len(pips)//len(conds))
                dec_kwargs['cv_folds'] = 0
                ys = ys % int(len(pips)/len(conds))


            try:
                decode_result = decode_responses(xs, ys, dec_kwargs=dec_kwargs)
                decode_result['data'].plot_confusion_matrix(labels=set(ys))

                cms.append(decode_result['data'].cm)

            except (AssertionError, ValueError) as e:
                print(e)
                print(f'WARNING: Could not decode session {sessname}')
                continue
            record[f'{dec_sffx}_data_accuracy'] = np.nanmean(decode_result['data'].accuracy)
            record[f'{dec_sffx}_shuffled_accuracy'] = np.nanmean(decode_result['shuffled'].accuracy)
        records.append(record)
    df = pd.DataFrame(records)
    # Remove duplicate sessname rows by grouping and keeping the first (should not happen, but just in case)
    df = df.groupby('sess', as_index=False).first().set_index('sess')
    if cache_path is not None:
        with open(cache_path, 'wb') as f:
            pickle.dump(df, f)

    return df, np.array(cms)


def ttest_decoding_results(decode_dfs, key, col1='data_accuracy', col2='shuff_accuracy'):
    """
    Perform an independent t-test between two columns in the decoding results DataFrame for a given key.

    Args:
        decode_dfs (dict): Output from run_decoding, mapping dec_sffx to DataFrame.
        key (str): Key for the decoding comparison (e.g., 'A-0_vs_base').
        col1 (str): First column for t-test (default 'data_accuracy').
        col2 (str): Second column for t-test (default 'shuff_accuracy').

    Returns:
        ttest_result: scipy.stats.ttest_ind result object.
    """
    df = decode_dfs[key]
    return ttest_ind(df[col1], df[col2], alternative='greater', equal_var=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('pkldir')
    parser.add_argument('event_responses_pkl')
    parser.add_argument('--plot_config_path', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--by_animal', action='store_true')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--multiprocess', action='store_true')
    parser.add_argument('--skip_batches_if_exist', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    pass


