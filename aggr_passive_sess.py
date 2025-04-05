import argparse
import pickle
import platform
from pathlib import Path

import matplotlib
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import ttest_ind, sem

from aggregate_ephys_funcs import *
from behviour_analysis_funcs import get_all_cond_filts
from ephys_analysis_funcs import posix_from_win, plot_sorted_psth, format_axis, plot_2d_array_with_subplots
from unit_analysis import get_participation_rate

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('pkldir')
    args = parser.parse_args()
    sys_os = platform.system().lower()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    home_dir = Path(config[f'home_dir_{sys_os}'])
    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])

    session_topology_path = ceph_dir / posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_ephys_2401.csv')
    session_topology = pd.read_csv(session_topology_path)

    cond_filters = get_all_cond_filts()

    animals = ['DO79','DO81']
    all_sess_info = session_topology.query('sess_order in ["pre","post"] and name in @animals')

    # animals = ['DO80', 'DO81', 'DO82']
    sessions2use = [Path(e).stem.replace('_SoundData','')
                    for e in all_sess_info['sound_bin'].values]

    pkldir = ceph_dir / posix_from_win(args.pkldir)
    pkls = list(pkldir.glob('*pkl'))
    fam_aggr_sess_pkl_path = Path(r'D:\ephys\aggr_passive_sessions.pkl')
    ow_flag = False
    if fam_aggr_sess_pkl_path.is_file() and not ow_flag:
        with open(fam_aggr_sess_pkl_path, 'rb') as f:
            sessions = pickle.load(f)
    else:
        pkls2load = [pkl for pkl in pkls if Path(pkl).stem in sessions2use]
        sessions = load_aggregate_sessions(pkls2load,)

        with open(fam_aggr_sess_pkl_path, 'wb') as f:
            pickle.dump(sessions, f)

    for sess in sessions:
        old_style_keys = list('ABCD')
        for old_key in old_style_keys:
            if old_key in sessions[sess].sound_event_dict:
                sessions[sess].sound_event_dict[f'{old_key}-0'] = sessions[sess].sound_event_dict.pop(old_key)


    # [sess for sess in sessions2use if not any([sess in Path(pkl).stem for pkl in pkls2load])]

    # [sessions.pop(sess) for sess in list(sessions.keys()) if 'A-0' not in list(sessions[sess].sound_event_dict.keys())]
    # [sessions.pop(sess) for sess in list(sessions.keys()) if 'D-1' not in list(sessions[sess].sound_event_dict.keys())]
    # [sessions.pop(sess) for sess in list(sessions.keys()) if 'A-2' in list(sessions[sess].sound_event_dict.keys())]
    [sessions.pop(sess) for sess in list(sessions.keys()) if not hasattr(sessions[sess].spike_obj,'unit_means')]
    [print(sess,sessions[sess].sound_event_dict.keys()) for sess in list(sessions.keys()) ]

    # update styles
    plt.style.use('figure_stylesheet.mplstyle')
    matplotlib.rcParams['figure.figsize'] = (4,3)

    aggr_figdir = ceph_dir / 'Dammy' / 'figures' / 'passive_aggr_plots'
    if not aggr_figdir.is_dir():
        aggr_figdir.mkdir(parents=True)

    # get active units
    window = (-.5, 1)
    event_responses_4_active_units = aggregate_event_reponses(sessions, events=None,  # [f'{pip}-0' for pip in 'ABCD']
                                                              events2exclude=['trial_start', ], window=window,
                                                              pred_from_psth_kwargs={'use_unit_zscore': True,
                                                                                     'use_iti_zscore': False,
                                                                                     'baseline': 0, 'mean': None,
                                                                                     'mean_axis': 0})
    concatenated_responses = {
        e: np.concatenate([event_responses_4_active_units[sessname][e].mean(axis=0)
                           for sessname in event_responses_4_active_units])
        for e in list(event_responses_4_active_units.values())[0].keys()}
    concatenated_sem = {
        e: np.concatenate([sem(event_responses_4_active_units[sessname][e])
                           for sessname in event_responses_4_active_units])
        for e in list(event_responses_4_active_units.values())[0].keys()}


    pips_2_plot =['A-0', 'B-0', 'C-0', 'D-0']
    all_psth_plot = plt.subplots(ncols=len(pips_2_plot), nrows=1, sharex=True, sharey=True,gridspec_kw={'wspace':0.05})
    for pi,pip in enumerate(pips_2_plot):
        cmap_norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
        plot_sorted_psth(event_responses_4_active_units, pip, pips_2_plot[0], window=window, sort_window=[0.1, 0.25],
                         im_kwargs=dict(norm=cmap_norm, cmap='bwr',plot_cbar=False),
                         sessname_filter=animals,plot_ts=False,
                         plot=(all_psth_plot[0], all_psth_plot[1][pi]))
        format_axis(all_psth_plot[1][pi], vlines=[0], ylabel='',
                    xlabel=f'')
        format_axis(all_psth_plot[1][pi], vlines=[0])
        all_psth_plot[1][pi].locator_params(axis='both', nbins=2,integer=True)
    all_psth_plot[1][0].set_ylabel('Unit #')
    all_psth_plot[0].set_size_inches(3.5,3)
    all_psth_plot[0].set_layout_engine('constrained')
    all_psth_plot[0].show()
    all_psth_plot[0].savefig(aggr_figdir / 'all_psth.pdf')

    # get active units
    active_units_by_pip = {pip: np.hstack([get_participation_rate(event_responses_4_active_units[sess][pip], window,
                                                                  [0.1, 0.25], 2, max_func=np.max)
                                           for sess in event_responses_4_active_units
                                           if any(e in sess for e in animals)])
                           for pip in pips_2_plot}
    activity_map_arr = np.array(list(active_units_by_pip.values()))
    activity_map = plot_2d_array_with_subplots(activity_map_arr.T > 0.5, plot_cbar=False,
                                               cmap='seismic', interpolation='none',
                                               norm=matplotlib.colors.PowerNorm(1))
    activity_map[1].set_xticks(np.arange(len(active_units_by_pip)))
    activity_map[1].set_yticklabels([])
    [activity_map[1].axvline(i + 0.5, c='k', lw=2) for i in range(len(active_units_by_pip))]
    activity_map[1].set_xticklabels([e.split('-')[0] for e in list(active_units_by_pip.keys())])
    activity_map[0].set_size_inches(2, 6)
    activity_map[0].set_layout_engine('tight')
    activity_map[0].show()

    by_pip_activity_sums = (activity_map_arr>0.5).sum(axis=1)
    by_pip_activity_sums_plot = plt.subplots()
    by_pip_activity_sums_plot[1].bar(np.arange(len(by_pip_activity_sums)),by_pip_activity_sums,
                                     tick_label=[e.split('-')[0] for e in list(active_units_by_pip.keys())],
                                     fc='lightgrey',ec='k')

    by_pip_activity_sums_plot[0].set_layout_engine('tight')
    by_pip_activity_sums_plot[0].show()

    by_pip_activity_sums_plot[0].savefig(aggr_figdir/'activity_sums_hipp_only.pdf')

    active_units_by_across_pips_patt_only = [(np.argwhere(unit>0.5)) for unit in activity_map_arr[[0,1,2,3,],:].T]
    active_units_by_across_pips = [(np.argwhere(unit>0.5)) for unit in activity_map_arr.T]
    selective_active_units_plot = plt.subplots()
    selective_pips = [[pip_i in unit and len(unit)==1 for pip_i in range(4)]
                      for unit in active_units_by_across_pips_patt_only]
    selective_units_sum = np.array(selective_pips).sum(axis=0)
    print('Proportion of active units: ', [round(e/activity_map_arr.shape[1]*100,2) for e in selective_units_sum])
    [selective_active_units_plot[1].bar(np.arange(0,len(sums)*2,2)+si,sums,
                                        tick_label=[e.split('-')[0] for e in list(active_units_by_pip.keys())[:4]],
                                      fc=c,ec='k')
     for si, (sums,c) in enumerate(zip([by_pip_activity_sums, selective_units_sum],['lightgrey','darkslateblue']))]
    format_axis(selective_active_units_plot[1])
    selective_active_units_plot[1].locator_params(axis='y', nbins=4, tight=True, integer=True)
    selective_active_units_plot[0].set_layout_engine('tight')
    selective_active_units_plot[0].show()
    selective_active_units_plot[0].savefig(aggr_figdir/'selective_units.pdf')

    # plot active A unit
    eg_A_units = sorted([1783] + [np.where(np.array(selective_pips)[:,i]==True)[0][1] for i in range(1,4)])
    plot_cols = ['#2B0504', '#4C6085','#874000','#BCABAE']
    # eg_A_units = [218,331,714,2531][:-1]
    unit_psths = plt.subplots(ncols=len(eg_A_units),figsize=(8,2),sharey=True,sharex=True)
    for eg_A_unit,unit_psth in zip(eg_A_units,unit_psths[1].flatten()):
        [unit_psth.plot(np.linspace(window[0],window[1],concatenated_responses['A-0'].shape[-1]),
                        concatenated_responses[pip][eg_A_unit], lw=2,
                        label=pip.split('-')[0], color=c,) for pip,c in zip(pips_2_plot,plot_cols)]
        [unit_psth.fill_between(np.linspace(window[0],window[1],concatenated_responses['A-0'].shape[-1]),
                                   concatenated_responses[pip][eg_A_unit]- concatenated_sem[pip][eg_A_unit],
                                   concatenated_responses[pip][eg_A_unit]+ concatenated_sem[pip][eg_A_unit],
                                   fc=c,alpha=0.1) for pip,c in zip(pips_2_plot,plot_cols)]
        unit_psth.set_title(f'Unit {eg_A_unit}', fontsize=10)
        # format_s(unit_psth[1],ylim=[-.5,2])
        unit_psth.locator_params(axis='both', nbins=2, tight=True, integer=True)
    unit_psths[1][-1].legend(loc='upper right',ncol=1)
    unit_psths[0].set_layout_engine('tight')
    unit_psths[0].show()
    unit_psths[0].savefig(aggr_figdir/f'unit_psth_A_units.pdf')

    cross_active_units = [[[all([p in unit for p in [pip_i,pip_j]])
                            for pip_i in range(len(active_units_by_pip))]
                           for pip_j in range(len(active_units_by_pip))]
                           for unit in active_units_by_across_pips]
    cross_active_units_arr = np.array(cross_active_units)
    # cross_active_units_arr = cross_active_units_arr.transpose((2,0,1))
    cross_active_units_arr = cross_active_units_arr.sum(axis=0)
    print('Total active units: ',[np.sum(row) for row in cross_active_units_arr])
    print('% active units: ',[round(np.sum(row)/activity_map_arr.shape[1]*100,2) for row in cross_active_units_arr])
    # cross_active_units_arr = np.array([row/row_diag for row,row_diag in zip(cross_active_units_arr,
    #                                                                        np.diagonal(cross_active_units_arr))])

    cross_active_units_plot = plot_2d_array_with_subplots(cross_active_units_arr, plot_cbar=True,
                                                          cmap='RdPu',
                                                          norm=matplotlib.colors.PowerNorm(.3)
                                                          )
    cross_active_units_plot[1].set_xticks(np.arange(len(active_units_by_pip)))
    cross_active_units_plot[1].set_yticks(np.arange(len(active_units_by_pip)))
    [cross_active_units_plot[1].text(pip_i,pip_i, pip_sum, va='center', ha='center',fontsize=9,color='w')
     for pip_i,pip_sum in enumerate(np.diagonal(cross_active_units_arr))]
    cross_active_units_plot[1].set_xticklabels([e.split('-')[0] for e in list(active_units_by_pip.keys())])
    cross_active_units_plot[1].set_yticklabels([e.split('-')[0] for e in list(active_units_by_pip.keys())])
    cross_active_units_plot[1].invert_yaxis()
    cross_active_units_plot[1].set_aspect('equal')
    cross_active_units_plot[0].set_layout_engine('tight')
    cross_active_units_plot[0].show()
    cross_active_units_plot[0].savefig(aggr_figdir / 'cross_active_units.pdf')

    # cross active units abcd only
    cross_active_units_abcd_only_plot = plot_2d_array_with_subplots(cross_active_units_arr[0:4,0:4], plot_cbar=True,
                                                          cmap='RdPu',
                                                          norm=matplotlib.colors.PowerNorm(.4))
    cross_active_units_abcd_only_plot[1].set_xticks(np.arange(len('ABCD')))
    cross_active_units_abcd_only_plot[1].set_yticks(np.arange(len('ABCD')))
    cross_active_units_abcd_only_plot[1].set_xticklabels([e.split('-')[0] for e in list(active_units_by_pip.keys())
                                                          if e.split('-')[0] in 'ABCD'])
    cross_active_units_abcd_only_plot[1].set_yticklabels([e.split('-')[0] for e in list(active_units_by_pip.keys())
                                                          if e.split('-')[0] in 'ABCD'])
    cross_active_units_abcd_only_plot[1].invert_yaxis()
    cross_active_units_abcd_only_plot[0].set_layout_engine('tight')
    cross_active_units_abcd_only_plot[0].show()
    # cross_active_units_abcd_only_plot[0].savefig(aggr_figdir / 'cross_active_units_abcd_only.pdf')

    # decode

    pvp_dec_dict = {}
    # pips2decode = [['A-0', 'A-1'], ['A-0', 'A-2'], ['A-1', 'A-2'],['A-0','A-0']]
    pips2decode = [pips_2_plot]
    for pips in pips2decode[:]:
        dec_sffx = "_vs_".join(pips)
        for sessname in tqdm(event_responses_4_active_units.keys(), desc='decoding',
                             total=len(event_responses_4_active_units.keys())):
            #     if sessname != 'DO79_240717a':
            #         continue
            # for sessname in ['DO79_240719a']:
            xs_list = [np.vstack([event_responses_4_active_units[sessname][p] for p in pip.split(';')]) for pip in pips]
            xs = np.vstack([x[:, :, -55:100].mean(axis=-1) for x in xs_list])
            ys = np.hstack([np.full(x.shape[0], ci) for ci, x in enumerate(xs_list)])
            # pre_split_idx = xs_list[0].shape[0]
            pvp_dec_dict[f'{sessname}_{dec_sffx}'] = decode_responses(xs, ys, n_runs=1, dec_kwargs={'cv_folds': 10, })
            pvp_dec_dict[f'{sessname}_{dec_sffx}']['data'].plot_confusion_matrix(pips)
        # plot accuracy
        pvp_accuracy = np.array([pvp_dec_dict[dec_name]['data'].accuracy
                                 for dec_name in pvp_dec_dict.keys() if dec_sffx in dec_name
                                 and any([e in dec_name for e in animals])])
        pvp_shuffle_accs = np.array([pvp_dec_dict[dec_name]['shuffled'].accuracy
                                     for dec_name in pvp_dec_dict.keys() if dec_sffx in dec_name
                                     and any([e in dec_name for e in animals])])
        pvp_accuracy_plot = plt.subplots()
        pvp_accuracy_plot[1].boxplot([pvp_accuracy.mean(axis=1),
                                      pvp_shuffle_accs.mean(axis=1)], labels=['data', 'shuffle'],widths=0.3,
                                     showmeans=False, meanprops=dict(mfc='k'), )
        # pvp_accuracy_plot[1].set_title(f'{dec_sffx}')
        format_axis(pvp_accuracy_plot[1], hlines=[1 / len(pips)])
        pvp_accuracy_plot[1].set_ylabel('Decoding accuracy')
        pvp_accuracy_plot[0].set_size_inches(2, 2.5)
        pvp_accuracy_plot[0].show()
        pvp_accuracy_plot[0].savefig(aggr_figdir / f'{dec_sffx}_accuracy.pdf')
        # ttest
        ttest = ttest_ind(pvp_accuracy.mean(axis=1), pvp_shuffle_accs.mean(axis=1),
                          alternative='greater', equal_var=False)
        print(f'{dec_sffx} ttest: pval =  {ttest[1]}. Mean accuracy of {pips} is {pvp_accuracy.mean():.3f}.')

    cmap_norm_4way = TwoSlopeNorm(vmin=0.15, vcenter=0.25, vmax=0.35)
    all_v_all_tag = "_vs_".join(pips2decode[-1])
    all_cms = np.array(
        [pvp_dec_dict[dec_name]['data'].cm for dec_name in pvp_dec_dict if all_v_all_tag in dec_name])
    cm_plot = plot_aggr_cm(all_cms, im_kwargs=dict(norm=cmap_norm_4way), include_values=False,
                           labels=[e.split('-')[0] for e in pips2decode[-1]], cmap='bwr',)

    cm_plot[0].set_size_inches((3, 3))
    cm_plot[0].show()
    cm_plot[0].set_layout_engine('tight')
    cm_plot[0].savefig(aggr_figdir / f'{all_v_all_tag}_cm.pdf')