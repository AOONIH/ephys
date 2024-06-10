import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ephys_analysis_funcs import posix_from_win
from argparse import ArgumentParser
import yaml
import platform
import re
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np


def mksessdir(sessdir:Path):
    if not sessdir.is_dir():
        sessdir.mkdir(parents=True)


def extract_date(filename_str:str,out_fmt:str='%y%m%d'):
    try:
        date = re.search(r'\d{4}-\d{2}-\d{2}', filename_str).group(0)
        date = datetime.strptime(date, '%Y-%m-%d').strftime('%y%m%d')
    except AttributeError:
        date = re.search(r'\d{2}\d{2}\d{2}', filename_str).group(0)
    return datetime.strptime(date, '%y%m%d').strftime(out_fmt)

def copy_data(sessdir:Path, projectdir:Path, datadir_lbl:str, dir2copy:str):
    name = sessdir.stem[:4]

    date = extract_date(sessdir.stem)
    sess_num = sessdir.stem[-3:]
    newsessdir = projectdir / 'rawdata' / name / f'{date}_{sess_num}' / datadir_lbl
    mksessdir(newsessdir)

    if datadir_lbl == 'ephys':
        assert dir2copy == 'Record Node 101'
    copydir = sessdir / dir2copy
    # else:
    #     raise NotImplementedError(f'{datadir_lbl} not implemented')
    # print(f'{contents=}')
    print(f'copying {copydir} to {newsessdir}')
    if dir2copy == 'CONTENTS':
        shutil.copytree(copydir,newsessdir)
    else:
        if not (newsessdir / dir2copy).is_dir():
            shutil.copytree(copydir,newsessdir / dir2copy)


def copy_data_w_topology(sess_topology:pd.Series,projectdir,**kwargs):  # sessdir:Path, projectdir:Path, datadir_lbl:str, dir2copy:str
    # print(f'{sess_topology.index =}')
    for i,data in enumerate(sess_topology):
        data_lbl = sess_topology.index[i]
        if not isinstance(data,Path):
            continue
        datadir_lbl = kwargs.get(data_lbl, data_lbl.replace('_dir',''))

        to_copy : Path = data

        name = sess_topology['name']
        date = sess_topology['date']
        sess_suffix = sess_topology['sess_order']

        newsessdir = projectdir / 'rawdata' / name / f'{date}_{sess_suffix}' / datadir_lbl
        if not newsessdir.is_dir():
            mksessdir(newsessdir)
        if datadir_lbl == 'ephys':
            if not to_copy.stem == 'Record Node 101':
                to_copy = to_copy/'Record Node 101'
                assert to_copy.is_dir(), f'{to_copy} is not a dir'
        print(f'copying {to_copy} to {newsessdir}')
        if to_copy.is_dir():
            shutil.copytree(to_copy,newsessdir,dirs_exist_ok=True)
        else:
            shutil.copy2(to_copy,newsessdir)


def match2ephys(sessdirs:list[Path],to_matchdirsss:list[list[Path]],datatype_lbls:list[str])->dict:
    print(f'{to_matchdirsss,datatype_lbls=}')
    sess_dict = {}
    sess_order = []
    for to_matchdirs,datatype_lbl in zip(to_matchdirsss,datatype_lbls):
        if len(to_matchdirs) == 0:
            print(f'No matching dirs for {sessdirs = }')
            return dict()
        if len (sessdirs) != len(to_matchdirs):
            print(f'Cant match dirs {sessdirs=} to {datatype_lbl}')
            # print([f'file{ei} : {e.stat().st_size / 1e6} mb' for ei, e in enumerate(to_matchdirs)])
            print([f'file {e.stem} : {e.stat().st_size / 1e6} mb' if e.is_file() else
                   f'dir {e.stem} : {sum(f.stat().st_size for f in e.rglob("*")) / 1e6} mb'
                   for ei, e in enumerate(to_matchdirs)])
            mapping = None
            while not mapping:
                try:
                    mapping = input('Provide mapping')
                    if mapping == 'SKIP':
                        return None
                    mapping = [[int(ee) for ee in e.split(',')] for e in mapping.split(';')]
                except:
                    print('invalid mapping, try again')
            print(f'{mapping = }')
            assert len(mapping)==2 and len(mapping[0])==len(mapping[1])
            sessdirs = [sessdirs[e] for e in mapping[0]]
            if sess_dict.get('ephys_dir'):
                assert sess_dict['ephys_dir'] == sessdirs, 'non matching sessdirs'
            else:
                sess_dict['ephys_dir'] = sessdirs
            to_matchdirs = [to_matchdirs[e] for e in mapping[1]]
            dtype_name = f'{datatype_lbl}' if 'bin' in datatype_lbl else f'{datatype_lbl}_dir'
        sess_dict[f'{datatype_lbl}_dir' ] = to_matchdirs
        sess_dict['ephys_dir'] = sessdirs
        if sess_dict.get('sess_order'):
            continue
        if len(sessdirs) == 1:
            sess_order = ['main']
        elif len(sessdirs) == 3:
            sess_order = ['pre','main','post']
        else:
            # sess_order = []
            while not sess_order:
                _order = input(f'provide pre,post, or main order {sessdirs = }')
                _order = _order.split(',')
                print(f'{_order = }')
                if all([e in ['pre','post','main'] for e in _order]) and len(_order) == len(sessdirs):
                    # print('valid')
                    sess_order = _order
                else:
                    print(f'bad input:{_order,sessdirs,to_matchdirs = }')
                    print(f'{all([e in ["pre","post","main"] for e in _order]) = }')
                    print(f'{len(_order)==len(sessdirs) = }')
                    # print(f'{len(sessdirs) = }')

            # sess_order = input(f'provide pre,post, or main order {sessdirs = }')
            print(f'{sess_order = }')
            assert len(sess_order) == len(sessdirs) and all([e in ['pre','post','main'] for e in sess_order])
        sess_dict['sess_order'] = sess_order
        # return {'ephys_dir':sessdirs,f'{datatype_lbl}_dir':to_matchdirs, 'sess_order':sess_order}
    return sess_dict


def get_sessdirs(datadir:Path, name:str, date:str,pattern='*<name>*<date>*'):
    # print(f'{name,date = }')
    # print(list(datadir.glob(f'*'))[:2])
    date_refmt = datetime.strptime(date, '%y%m%d').strftime('%Y-%m-%d')
    matching_dirs = sorted(np.unique(list(datadir.glob(pattern.replace('<name>',name).replace('<date>',date)))+
                                     list(datadir.glob(pattern.replace('<name>',name).replace('<date>',date_refmt)))))
    matching_dirs = [e for e in matching_dirs if 'concat' not in e.stem]
    return matching_dirs


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('topdir')
    parser.add_argument('projectdir')
    parser.add_argument('datadir_lbl')
    parser.add_argument('dir2copy')

    with open(parser.parse_args().config_file,'r') as file:
        config = yaml.safe_load(file)
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])
    assert ceph_dir.is_dir()

    args = parser.parse_args()
    datadir = ceph_dir / posix_from_win(args.topdir)
    projectdir = ceph_dir / posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test')
    assert datadir.is_dir() and projectdir.is_dir()
    sess_dirs = [e for e in list(datadir.glob('*'))
                 if any([e.stem.startswith(name) for name in ['DO79','DO80','DO81','DO82']])and 'concat' not in e.stem]

    sess_dirs = [e for e in sess_dirs if re.match(r'^[a-zA-Z]{2}\d{2}_.*_\d{3}$',e.stem)]
    assert all([re.match(r'^[a-zA-Z]{2}\d{2}_.*_\d{3}$',e.stem) for e in sess_dirs])  # test for valid session names

    # unique_dates = set([extract_date(sess_dir.stem) for sess_dir in sess_dirs])
    unique_names = set([sess_dir.stem[:4] for sess_dir in sess_dirs])
    dirs2match = [ceph_dir / posix_from_win(r'X:\Dammy\mouse_pupillometry\mouse_hf'),ceph_dir / posix_from_win(r'X:\Dammy\harpbins')]
    d_lbls = ['videos','beh_bin']
    d_patterns = ['*<name>_<date>_*', f'<name>_HitData_<date>*.bin']
    all_sess_topology = []

    csv_path = projectdir / 'session_topology.csv'
    if csv_path.exists():
        old_all_sess_topology = pd.read_csv(csv_path)
        old_sessions = list((n,str(d)) for n,d in zip(old_all_sess_topology['name'],old_all_sess_topology['date']))
        # existing_dates = old_all_sess_topology['date'].unique()
    else:
        old_all_sess_topology = pd.DataFrame()
        old_sessions = []

    for name in unique_names:
        # if name == 'DO79':
        #     continue
        name_ephys_dirs = [extract_date(e.stem) for e in sess_dirs if e.stem.startswith(name)]
        unique_dates = set(extract_date(e) for e in name_ephys_dirs)
        ephys_sess = [get_sessdirs(datadir, name, date)
                      for date in tqdm(unique_dates,total=len(unique_dates),desc=f'finding sessions {name}')
                      if (name,date) not in old_sessions]
        # ephys_sess = [e for e in ephys_sess if e]
        match_dir_dfs = []
        # for dir2match,d_lbl, d_pattern in zip(dirs2match,d_lbls,d_patterns):
        print(old_sessions)
        sessions_by_date = [[get_sessdirs(dir2match,name,date,pattern=d_pattern)
                             for date in tqdm(unique_dates,total=len(unique_dates),desc=f'finding sessions {name}')
                             if (name,date) not in old_sessions]
                            for dir2match,d_lbl, d_pattern in zip(dirs2match,d_lbls,d_patterns)]
        # sessions_by_date = [[ee for ee in e if ee] for e in sessions_by_date]
        sessions_by_date = pd.DataFrame(sessions_by_date).T.to_numpy().tolist()

        # assert len(sessions_by_date[0]) == len(sessions_by_date[1])
            # _sesss = [e for e in sess_dirs if not 'concat' in e.stem]
            # for sess in zip(_sesss,sessions_by_date):
            #     session_topology = pd.DataFrame.from_dict(match2ephys(sess[0],sess[1],''),orient='columns')
        session_topology = [pd.DataFrame.from_dict(match2ephys(ephys_dir,other_dirs,d_lbls),
                                                   orient='columns')
                            for ephys_dir,other_dirs in zip(ephys_sess, sessions_by_date)]

        all_sess_topology.extend(session_topology)
        # ephys_sess[f'{d_lbl}_dir'] = session_topology[f'{d_lbl}_dir']

    all_sess_topology = pd.concat(all_sess_topology)
    all_sess_topology.rename(columns={'beh_bin_dir':'beh_bin'},inplace=True)
    # all_sess_paths_df = pd.DataFrame([_sesss,sessions_by_date])
    # [print(len(e),len(ee)) for e,ee in zip(_sesss,sessions_by_date)]
    # all_sess_topology.columns = ['ephys_dir','videos_dir','sess_order']
    if all_sess_topology.empty:
        exit('No new sessions found')
    all_sess_topology['name'] = [e.stem[:4] for e in all_sess_topology['ephys_dir']]
    all_sess_topology['date'] = [extract_date(e.stem) for e in all_sess_topology['ephys_dir']]
    sound_bins = [list(dirs2match[1].glob(f'{name}*SoundData_{sessdate.stem[-7:]}.bin'))
                  for name,sessdate in zip(all_sess_topology['name'],all_sess_topology['beh_bin'])]
    assert all([len(e) == 1 for e in sound_bins])
    all_sess_topology['sound_bin'] = [e[0] for e in sound_bins]
    if not old_all_sess_topology.empty:
        all_sess_topology = pd.concat([old_all_sess_topology, all_sess_topology])
    all_sess_topology.reset_index(drop=True, inplace=True)
    all_sess_topology.to_csv(projectdir / 'session_topology.csv',index=False)

    d_lbls = dict(ephys_dir='ephys',vid_dir='videos',beh_bin='behaviour',sound_bin='behaviour')
    for idx, sess in tqdm(all_sess_topology.iterrows(),total=len(all_sess_topology),desc='copying sessions'):
        # copy_data_w_topology(sess,projectdir,**d_lbls)
        pass


    # bad sess: DO79_240517 DO81_240516