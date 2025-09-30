import argparse
import pickle
from pathlib import Path

import pandas as pd

from behviour_analysis_funcs import get_all_cond_filts
from pupil_analysis_funcs import group_pupil_across_sessions


def get_analysed_sessions(pupil_by_cond:dict):

    mega_pupil_df = pd.concat(pupil_by_cond.values(), axis=0)
    if 'name' not in mega_pupil_df.index.names:
        mega_pupil_df['name'] = mega_pupil_df.index.get_level_values('sess').str.split('_').str[0]
        mega_pupil_df.set_index(['name'], append=True, inplace=True)
    if 'date' not in mega_pupil_df.index:
        mega_pupil_df['date'] = mega_pupil_df.index.get_level_values('sess').str.split('_').str[1]
        mega_pupil_df.set_index(['date'], append=True, inplace=True)
    # print(mega_pupil_df.index.droplevel('time').names)
    return mega_pupil_df.index.droplevel(['time','trial','sess']).unique().to_frame(index=False)[['name','date']]


if '__main__' == __name__:
    # add args for pupil_by_cond pickle
    parser = argparse.ArgumentParser()
    parser.add_argument('pupil_by_cond_path', type=str)
    args = parser.parse_args()

    with open(args.pupil_by_cond_path, 'rb') as f:
        pupil_by_cond = pd.read_pickle(Path(f))
    all_names_dates = get_analysed_sessions(pupil_by_cond)
    all_names_dates.to_csv('all_names_dates_no_abstraction.csv', index=False)

    # load sessions
    with open(Path(r"X:\Dammy\pupil_data\ephys_2401_musc_2401_cohort_sess_dicts_no_filt.pkl"), 'rb') as f:
        sessions = pickle.load(f)
    cond_filters = get_all_cond_filts()
    A_by_cond = {cond: group_pupil_across_sessions(sessions, list(sessions.keys()), 'A', cond, cond_filters, )
                 for cond in ['rare', 'frequent', 'deviant_C', 'alternating',]}

    # pickle A_by_cond
    with open(r'D:\A_by_cond_ephys_2401_musc_2401.pkl', 'wb') as f:
        pickle.dump(A_by_cond, f)