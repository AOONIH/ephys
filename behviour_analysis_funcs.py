from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import legend_handler
from matplotlib.lines import Line2D
from tqdm import tqdm

from ephys_analysis_funcs import SessionPupil, Session, get_main_sess_patterns

if '__main__' == __name__:
    pass


class MyHandlerLine2D(legend_handler.HandlerLine2D):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):

        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)

        ydata = ((height-ydescent)/2.)*np.ones_like(xdata, float)
        legline = Line2D(xdata, ydata)

        self.update_prop(legline, orig_handle, legend)
        #legline.update_from(orig_handle)
        #legend._set_artist_props(legline) # after update
        #legline.set_clip_box(None)
        #legline.set_clip_path(None)
        legline.set_drawstyle('default')
        legline.set_marker("")
        legline.set_linewidth(3)


        legline_marker = Line2D(xdata_marker, ydata[:len(xdata_marker)])
        self.update_prop(legline_marker, orig_handle, legend)
        #legline_marker.update_from(orig_handle)
        #legend._set_artist_props(legline_marker)
        #legline_marker.set_clip_box(None)
        #legline_marker.set_clip_path(None)
        legline_marker.set_linestyle('None')
        if legend.markerscale != 1:
            newsz = legline_marker.get_markersize()*legend.markerscale
            legline_marker.set_markersize(newsz)
        # we don't want to add this to the return list because
        # the texts and handles are assumed to be in one-to-one
        # correpondence.
        legline._legmarker = legline_marker

        return [legline, legline_marker]


def get_sess_name_date_idx(sessname,session_topology):
    name, date, suffix = sessname.split('_')[0], int(sessname.split('_')[1][:-1]), sessname[-1]
    sessions_on_date = session_topology.query('name==@name and date==@date').reset_index()
    sess_suffixes = [Path(e).stem[-1] for e in sessions_on_date['sound_bin']]
    sess_idx = [ei for ei, e in enumerate(sess_suffixes) if e == suffix][0]
    sess_info = sessions_on_date.loc[sess_idx]

    return name, date, sess_idx, sess_info


def sync_beh2sound(pupil_obj:SessionPupil,beh_events_df:pd.DataFrame, sound_events_df:pd.DataFrame):
    sync_offset =  sound_events_df['Timestamp'].values[0] - beh_events_df['Timestamp'].values[0]
    print(f'sync_offset = {sync_offset}')
    pupil_obj.sound_writes.loc[:,'Timestamp'] = pupil_obj.sound_writes['Timestamp'] - sync_offset


def get_drug_dates(cohort_config, session_topology,drug_sess_dict,date_start=None,date_end=None):
    muscimol_dates = cohort_config.get('muscimol_dates', [])
    saline_dates = cohort_config.get('saline_dates', [])
    none_dates = [d for d in session_topology['date'].unique() if d not in muscimol_dates + saline_dates]

    # remove any saline date 1 after muscimol dates
    muscimol_dates_dt = [datetime.strptime(str(d), '%y%m%d') for d in muscimol_dates]
    saline_dates_dt = [datetime.strptime(str(d), '%y%m%d') for d in saline_dates]
    saline_dates = [d for d, d_dt in zip(saline_dates,saline_dates_dt)
                    if all(d_dt - np.array(muscimol_dates_dt) != np.timedelta64(1, 'D'))]

    if date_end:
        muscimol_dates = [d for d in muscimol_dates if d < date_end]
        saline_dates = [d for d in saline_dates if d < date_end]
        none_dates = [d for d in none_dates if d < date_end]

    if date_start:
        muscimol_dates = [d for d in muscimol_dates if d >= date_start]
        saline_dates = [d for d in saline_dates if d >= date_start]
        none_dates = [d for d in none_dates if d >= date_start]

    altrand_dates = cohort_config.get('altrand_dates', [])
    if cohort_config.get('none_dates', None):
        none_dates = cohort_config['none_dates']
    else:
        none_dates = [d for d in none_dates]
    if any([saline_dates, muscimol_dates]):
        none_dates = [d for d in none_dates if d > min([min(muscimol_dates), min(saline_dates)])
        and d
                      and d not in altrand_dates]

    if cohort_config.get('exclude_dates', None):
        date_sets = [[d for d in date_set if d not in cohort_config['exclude_dates']]
                     for date_set in [muscimol_dates, saline_dates, none_dates]]
        muscimol_dates, saline_dates, none_dates = date_sets
    [drug_sess_dict.update({subset: drug_sess_dict.get(subset, []) +
                                    sum([[f'{a}_{d}' for a in session_topology['name'].unique() if a != 'DO76']
                                         for d in dates], [])})
     for subset, dates in zip(['muscimol', 'saline','none'], [muscimol_dates, saline_dates, none_dates])]


def get_all_cond_filts():
    return dict(
        rare='Tone_Position==0 & Stage==3 & local_rate >= 0.8 & N_TonesPlayed==4 & Session_Block==0',
        rare_prate='Tone_Position==0 & Stage==3 & PatternPresentation_Rate == 0.9 & N_TonesPlayed==4',
        frequent='Tone_Position==0 & Stage==3 & local_rate <= 0.2 & N_TonesPlayed==4 &Session_Block==0',
        frequent_prate='Tone_Position==0 & Stage==3 & PatternPresentation_Rate == 0.1 & N_TonesPlayed==4',
        mid1='Tone_Position==0 & Stage==3 & local_rate > 0.3 & local_rate < 0.5 & N_TonesPlayed==4',
        mid2='Tone_Position==0 & Stage==3 & local_rate < 0.7 & local_rate >= 0.5 & N_TonesPlayed==4',
        recent='Tone_Position==0 & Stage==3 & n_since_last<=2 & N_TonesPlayed==4 & Session_Block==0 & PatternPresentation_Rate != 0.5',
        distant='Tone_Position==0 & Stage==3 & n_since_last>=3 & N_TonesPlayed==4 & Session_Block==0 & PatternPresentation_Rate != 0.5',
        pattern='Tone_Position==0 & Stage==3 & N_TonesPlayed==4',
        none='Tone_Position==1 & Stage==3',
        earlyX='Tone_Position==0 & Stage==3 & N_TonesPlayed<4',
        earlyX_1tones='Tone_Position==0 & Stage==3 & N_TonesPlayed==1',
        earlyX_2tones='Tone_Position==0 & Stage==3 & N_TonesPlayed==2 & Trial_Outcome==1',
        earlyX_3tones='Tone_Position==0 & Stage==3 & N_TonesPlayed==3 & Trial_Outcome==1',
        alternating='Tone_Position==0 & Stage==3 & N_TonesPlayed==4 & Session_Block==1 & PatternPresentation_Rate == 0.5',
        random='Tone_Position==0 & Stage==3 & N_TonesPlayed==4 & Session_Block==0 & PatternPresentation_Rate == 0.5 ',
               # ' & local_rate >= 0.4 & local_rate <= 0.6',
               # ' & n_since_last==2',
        normal='Tone_Position==0 & Stage==4 & Pattern_Type==0 & N_TonesPlayed==4 & Session_Block ==3',
        normal_exp='Tone_Position==0 & Stage==4 & Pattern_Type==0 & N_TonesPlayed==4 & Session_Block ==2',
        normal_exp_midlate='Tone_Position==0 & Stage==4 & Pattern_Type==0 & N_TonesPlayed==4 & Session_Block ==2 &'
                           'normal_exp_num_cumsum > normal_exp_num_cumsum.max()-20',
        all_devs='Tone_Position==0 & Stage==4 & Pattern_Type!=0 & N_TonesPlayed==4 & Session_Block ==3',
        deviant_C='Tone_Position==0 & Stage==4 & Pattern_Type==1 & N_TonesPlayed==4 & Session_Block ==3',
        hit_all='Stage in [3,4] & Trial_Outcome==1',
        miss_all='Stage in [3,4] & Trial_Outcome==0',
        hit_pattern='Tone_Position==0 & Stage in [3,4] & N_TonesPlayed==4 & Trial_Outcome==1',
        miss_pattern='Tone_Position==0 & Stage in [3,4] & N_TonesPlayed==4 & Trial_Outcome==0',
        hit_none='Tone_Position==1 & Stage in [3,4] & Trial_Outcome==1',
        miss_none='Tone_Position==1 & Stage in [3,4] & Trial_Outcome==0',
        no_early= 'Tone_Position==0 & Stage in [3,4] & N_TonesPlayed==4 & Early_Licks==0',
        no_patt_lick= 'Tone_Position==0 & Stage in [3,4] & N_TonesPlayed==4 & lick_in_patt==0',
        dev_ABCD1 = 'Tone_Position==0 & Stage==4 & Pattern_Type==10 & N_TonesPlayed==4 & Session_Block ==3',
        dev_ABBA1 = 'Tone_Position==0 & Stage==4 & Pattern_Type==11 & N_TonesPlayed==4 & Session_Block ==3',
    )


def group_td_df_across_sessions(sessions_objs:dict,sessnames:list) -> pd.DataFrame:
    all_td_df = []
    for sessname in sessnames:
        name,date = sessname.split('_')
        if not date.isnumeric():
            date=date[:-1]
        assert isinstance(sessions_objs[sessname],Session)
        if sessions_objs[sessname].td_df is None:
            continue
        sess_td = sessions_objs[sessname].td_df
        # add sess name to multiindex
        try:
            sess_td.index = pd.MultiIndex.from_arrays([[sessname]*len(sess_td),[name]*len(sess_td),[date]*len(sess_td),
                                                       sess_td.reset_index().index+1],
                                                      names=['sess','name','date','trial_num'])
        except NameError:
            print(f'{sessname} {sess_td.index}')
        all_td_df.append(sessions_objs[sessname].td_df)
    return pd.concat(all_td_df,axis=0)


def get_n_since_last(td_df:pd.DataFrame,col_name:str,val):
    # idxs = td_df.index
    # td_df[f'n_since_last_{col_name}'] = np.arange(td_df.shape[0])
    # since_last = td_df.query(f'{col_name} == @val').index
    # if not since_last.empty:
    #     for t, tt in zip(since_last, np.pad(since_last, [1, 0])):
    #         td_df.loc[idxs[tt] + 1:idxs[t], f'n_since_last_{col_name}'] = td_df.loc[idxs[tt] + 1:idxs[t], f'n_since_last_{col_name}'] - tt
    #     td_df.loc[idxs[t] + 1:, f'n_since_last_{col_name}'] = td_df.loc[idxs[t] + 1:, f'n_since_last_{col_name}'] - t
    td_df[f'n_since_last_{col_name}'] = calculate_true_streak(td_df[col_name] == val)


def vec_dt_replace(series, year=None, month=None, day=None,
                   hour=None, minute= None, second=None, microsecond=None,nanosecond=None):
    return pd.to_datetime(
        {'year': series.dt.year if year is None else year,
         'month': series.dt.month if month is None else month,
         'day': series.dt.day if day is None else day,
         'hour': series.dt.hour if hour is None else hour,
         'minute': series.dt.minute if minute is None else minute,
         'second': series.dt.second if second is None else second,
         'microsecond': series.dt.microsecond if microsecond is None else microsecond,
         'nanosecond': series.dt.nanosecond if nanosecond is None else nanosecond,
         })

def add_datetimecol(df, colname, timefmt='%H:%M:%S.%f'):
     # utc=True
    # datetime_arr = []
    date_array = df.index.to_frame()['date']
    date_array_dt = pd.to_datetime(date_array,format='%y%m%d').to_list()   # [datetime.strptime(d,'%y%m%d') for d in date_array]
    date_array_dt_ser = pd.Series(date_array_dt)

    s = df[colname]
    s_nans = s.isnull()
    s = s.fillna('00:00:00')
    try:s_split = pd.DataFrame(s.str.split('.').to_list())
    except TypeError: print('typeerror')
    if len(s_split.columns) == 1:
        s_split[1] = np.full_like(s_split[0],'0')
    s_split.columns = ['time_hms','time_decimal']
    s_split['time_decimal'] = s_split['time_decimal'].fillna('0').str.ljust(9,'0')
    s_dt = pd.to_datetime(s_split['time_hms'],format='%H:%M:%S')
    try:s_dt = vec_dt_replace(s_dt,year=date_array_dt_ser.dt.year,month=date_array_dt_ser.dt.month,
                          day=date_array_dt_ser.dt.day, nanosecond=pd.to_numeric(s_split['time_decimal'].str.ljust(9,'0')))
    except:print('error')
    s_dt.iloc[s_nans] = pd.NaT
    df[f'{colname}_dt'] = s_dt.to_numpy()


def get_datetime_series(times, date):
    if isinstance(times, str):
        times = times.split(';')
    if isinstance(date, float):
        return []
    s = pd.Series(times)
    date_array_dt = pd.to_datetime([date]*len(s),format='%y%m%d').to_list()   # [datetime.strptime(d,'%y%m%d') for d in date_array]
    date_array_dt_ser = pd.Series(date_array_dt)
    s_nans = s.isnull()
    if any(s_nans):
        return []
    # s = s.fillna('00:00:00')
    try:
     s_split = pd.DataFrame(s.str.split('.').to_list())
    except TypeError:
     print('typeerror')
    if len(s_split.columns) == 1:
     s_split[1] = np.full_like(s_split[0], '0')
    s_split.columns = ['time_hms', 'time_decimal']
    s_split['time_decimal'] = s_split['time_decimal'].fillna('0').str.ljust(9, '0')
    s_dt = pd.to_datetime(s_split['time_hms'], format='%H:%M:%S')
    try:
     s_dt = vec_dt_replace(s_dt, year=date_array_dt_ser.dt.year, month=date_array_dt_ser.dt.month,
                           day=date_array_dt_ser.dt.day,
                           nanosecond=pd.to_numeric(s_split['time_decimal'].str.ljust(9, '0')))
    except:
     print('error')
    # s_dt.iloc[s_nans] = pd.NaT
    return s_dt.tolist()


def format_timestr(timestr_series,date=None) -> (pd.Series, pd.Series):
    """
    function to add decimal to time strings. also returns datetime series
    :param timestr_series:
    :return:
    """
    s=pd.Series(timestr_series)
    s_split = pd.DataFrame(s.str.split('.').to_list())
    s_dt = pd.to_datetime(s_split[0],format='%H:%M:%S').replace(microsecond=pd.to_numeric(s_split[1]))
    datetime_arr = []
    for t in s:
        if isinstance(t, str):
            t_split = t.split('.')
            t_hms = t_split[0]
            if len(t_split) == 2:
                t_ms = t.split('.')[1]
            else:
                t_ms = 0
            t_hms_dt = datetime.strptime(t_hms, '%H:%M:%S')
            t_ms_micros = round(float(f'0.{t_ms}'), 6) * 1e6
            t_dt = t_hms_dt.replace(microsecond=int(t_ms_micros))
            if date:
                t_dt = t_dt.replace(date[0],date[1],date[2])
            datetime_arr.append(t_dt)

        else:
            datetime_arr.append(np.nan)
    return datetime_arr


def calculate_true_streak(boolean_series):
    """
    Calculate the current streak of True values in a boolean series.

    Parameters:
    boolean_series (pd.Series): A boolean pandas Series.

    Returns:
    pd.Series: A pandas Series with the current streak of True values.
    """
    # Convert boolean series to an integer array (1 for True, 0 for False)
    bool_array = boolean_series.values.astype(int)

    # Find the positions where the value is False (0)
    zero_positions = np.where(bool_array == 0)[0]

    # Create an array to hold the streak values
    streak_array = np.zeros_like(bool_array)

    # Start index for the streaks
    start_idx = 0

    for zero_pos in zero_positions:
        # Fill the streak values up to the position of the next zero
        streak_array[start_idx:zero_pos] = np.arange(1, zero_pos - start_idx + 1)
        start_idx = zero_pos + 1

    # Handle the last streak if it ends at the end of the array
    if start_idx < len(bool_array):
        streak_array[start_idx:] = np.arange(1, len(bool_array) - start_idx + 1)

    return streak_array  # pd.Series(streak_array, index=boolean_series.index)


def get_last_pattern(tone_pos_bool_ser:pd.Series):
    """
    Calculate the current streak of True values in a boolean series.

    Parameters:
    boolean_series (pd.Series): A boolean pandas Series.

    Returns:
    pd.Series: A pandas Series with the current streak of True values.
    """
    # Convert boolean series to an integer array (1 for True, 0 for False)
    bool_array = tone_pos_bool_ser.values.astype(int)

    # Find the positions where the value is False (0)
    zero_positions = np.where(bool_array == 0)[0]+1

    # Create an array to hold the streak values
    streak_array = np.zeros_like(bool_array)

    # Start index for the streaks
    start_idx = 0

    for zero_pos in zero_positions:
        # Fill the streak values up to the position of the next zero
        streak_array[start_idx:zero_pos] = np.arange(1, zero_pos - start_idx + 1)
        start_idx = zero_pos + 1

    # Handle the last streak if it ends at the end of the array
    if start_idx < len(bool_array):
        streak_array[start_idx:] = np.arange(1, len(bool_array) - start_idx + 1)

    return streak_array  # pd.Series(streak_array, index=boolean_series.index)




def get_earlyX_trials(td_df):
    assert all([col in td_df.columns for col in ['Gap_Time_dt','ToneTime_dt']]), \
        'td_df must have Gap_Time_dt and ToneTime_dt columns'
    tone_X_offset = (td_df['Gap_Time_dt'] - td_df['ToneTime_dt']).dt.total_seconds()
    td_df['earlyX'] = (tone_X_offset < 1)


def in_time_window(t2eval,t,window=(-1,2)):
    in_window = all([t2eval >= t+timedelta(seconds=window[0]), t2eval <= t+timedelta(seconds=window[1])])
    return in_window


def get_lick_in_patt_trials(td_df, sess:str):
    assert all([col in td_df.columns for col in ['Gap_Time_dt','ToneTime_dt']]), \
        'td_df must have Gap_Time_dt and ToneTime_dt columns'

    date = sess.split('_')[1]
    sess_date=str(date)
    y,m,d = int(f'20{sess_date[:2]}'),int(sess_date[2:4]),int(sess_date[4:])
    td_df['Lick_Times_dt'] = td_df['Lick_Times'].apply(lambda e: get_datetime_series(e, date))
    td_df['lick_in_patt'] = [any([timedelta(seconds=2) > (lick - tone_t) > timedelta(seconds=0)
                                   for lick in licks]) if licks else False
                              for licks,tone_t in zip(td_df['Lick_Times_dt'],td_df['ToneTime_dt']) ]

def get_cum_sum(td_df,col_name,eval_str):
    td_df[f'{col_name}_cumsum'] = td_df.eval(eval_str).cumsum()


def get_prate_block_num(td_df:pd.DataFrame, prate:float,colname=None):
    if colname is None:
        colname=prate
    td_df['prate_diff'] = td_df['PatternPresentation_Rate'].diff()
    td_df[f'{colname}_block_num'] = td_df.eval(f'prate_diff != 0 and PatternPresentation_Rate == {prate}').cumsum()



def group_licks_across_sessions(sessions_objs: dict, sessnames: list, event: str, cond_name: str, cond_filters: dict):
    all_cond_licks = []
    cond_filter = cond_filters[cond_name]

    for sessname in tqdm(sessnames, total=len(sessnames), desc='group_licks_across_sessions'):
        session = sessions_objs.get(sessname)
        if not session or not isinstance(session, Session):
            continue
        lick_obj = session.lick_obj
        if not lick_obj or not lick_obj.event_licks:
            continue

        # Query trial numbers based on the condition filter
        trial_nums = session.td_df.query(cond_filter).index.get_level_values('trial_num').values

        try:
            cond_licks = lick_obj.event_licks[f'{event}_licks'].loc[:, trial_nums, :]
            all_cond_licks.append(cond_licks)
        except KeyError:
            continue

    if all_cond_licks:
        return pd.concat(all_cond_licks, axis=0)
    else:
        return pd.DataFrame()


def filter_session(sessions:dict, sessname:str,stages:list, drug_sess_dict:dict,filt4patt=True):
    if not all([any([stage in sessions[sessname].td_df['Stage'].values for stage in stages]),
                len(sessions[sessname].td_df) > 100, get_main_sess_patterns(td_df=sessions[sessname].td_df) if filt4patt else True,
                sessions[sessname].td_df['Trial_Outcome'].mean()>0.25]):
        if drug_sess_dict:
            if sessname in drug_sess_dict['muscimol']:
                drug_sess_dict['muscimol'].remove(sessname)
            if sessname in drug_sess_dict['saline']:
                drug_sess_dict['saline'].remove(sessname)
            if sessname in drug_sess_dict['none']:
                drug_sess_dict['none'].remove(sessname)
        sessions.pop(sessname)


def get_cumsum_columns(sessions, sessname):
    get_cum_sum(sessions[sessname].td_df, 'alternating_num', 'Tone_Position==0 & Session_Block==1')
    get_cum_sum(sessions[sessname].td_df, 'random_num',
                'Tone_Position==0 & Session_Block==0 & PatternPresentation_Rate==0.5')
    get_cum_sum(sessions[sessname].td_df, 'normal_num', 'Tone_Position==0 & Session_Block==3 & Pattern_Type==0')
    get_cum_sum(sessions[sessname].td_df, 'normal_exp_num', 'Tone_Position==0 & Session_Block==2 & Pattern_Type==0')
    get_cum_sum(sessions[sessname].td_df, 'deviant_C_num', 'Tone_Position==0 & Session_Block==3 & Pattern_Type==1')
