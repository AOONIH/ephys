from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
import matplotlib

from spike_time_utils import zscore_by_trial

matplotlib.use('TkAgg')
from matplotlib.colors import TwoSlopeNorm
from matplotlib.rcsetup import cycler

from aggregate_ephys_funcs import decode_responses, plot_aggr_cm

from scipy.linalg import orthogonal_procrustes
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from aggregate_psth_analysis import AggregateSession
from behviour_analysis_funcs import get_main_sess_td_df
from plot_funcs import plot_shaded_error_ts, format_axis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import umap
from population_analysis_funcs import PopPCA
#
# td_path_pattern = 'data/Dammy/<name>/TrialData'
# with open('config.yaml','r') as f:
#     config = yaml.safe_load(f)
# home_dir = Path(config['home_dir_windows'])
# session_topolgy = pd.read_csv(r"X:\Dammy\Xdetection_mouse_hf_test\session_topology_ephys_2401.csv")
# td_paths = [Path(sess_info['sound_bin'].replace('_SoundData', '_TrialData')).with_suffix('.csv').name
#             for _, sess_info in session_topolgy.iterrows()]
# abs_td_paths = [home_dir / td_path_pattern.replace('<name>', sess_info['name']) / td_path
#                 for (td_path, (_, sess_info)) in zip(td_paths, session_topolgy.iterrows())]
#
#
# sessnames = [Path(sess_info['sound_bin'].replace('_SoundData', '')).stem
#              for _, sess_info in session_topolgy.iterrows()]
# td_dfs = {sessname: get_main_sess_td_df(_main_sess_td_name=abs_td_path, _home_dir=home_dir)[1]
#           for sessname, abs_td_path in zip(sessnames, abs_td_paths)
#           }
#
# session_topolgy['tdata_file'] = list(td_dfs.values())

import joblib
from scipy.stats import sem, ttest_ind
from plot_funcs import unique_legend
#
resps_pkl = Path(r"D:\ephys\abstraction_by_pip_resps_ephys_no_zscore_2401_2504.joblib")
event_responses = joblib.load(resps_pkl)
# # event_responses = {k: {kk:np.cumsum(vv,axis=-1) for kk,vv in v.items()} for k,v in event_responses.items()}
#
# pips_as_ints = {pip: pip_i for pip_i, pip in enumerate([f'{pip}-{pip_i}' for pip in 'ABCD' for pip_i in range(3)])}
# events_by_property = {
#     'ptype_i': {pip: 0 if int(pip.split('-')[1]) < 2 else 1 for pip in [f'{p}-{i}' for i in range(3) for p in 'ABCD']},
# }
#
# abcd_abba1_com_by_name = {}
# plt.style.use('figure_stylesheet.mplstyle')
#
# plot_config_path = Path(r'H:\ephys\abstraction_by_pip_plot_config.yaml')
# with open(plot_config_path, 'r') as f:
#     plot_config = yaml.load(f, Loader=yaml.FullLoader)
#
# aggr_resps = AggregateSession(resps_pkl.parent,None,plot_config,plot_config['pips_2_plot'])
# aggr_resps.aggregate_mean_sess_responses( filt_by_trial_num=True,
#                                           pip_4_filt=[['A-1','A-2']], pip_2_filt=['A-0','B-0','C-0','D-0'],
#                                           filt_func=lambda tn, ft: [max(tn[tn < e] - e) + e for e in ft],
#                                           )
#
# for animal in ['DO81', 'DO79', 'DO95', 'DO97', ]:    # sim using full population
#     # concatenated_event_responses_hipp_only = {
#     #     e: np.concatenate([event_responses[sessname][e].mean(axis=0) if e != 'A-0' else
#     #                        event_responses[sessname][e][100:115].mean(axis=0)
#     #                        for sessname in event_responses
#     #                        # if any([animal in sessname for animal in ['DO81']])],)
#     #                        if animal in sessname])
#     #     for e in list(event_responses.values())[0].keys()}
#     concatenated_event_responses_hipp_only = {e:resps[[animal in s for s in aggr_resps.concatenated_sessnames]]
#                                               for e, resps in aggr_resps.concatenated_event_responses.items()}
#     pips_2_comp = ['D-1', 'D-0', 'D-2', ]
#     dev_comps = {}
#     for pip in 'ABCD':
#         pips_2_comp = [f'{pip}-1', f'{pip}-0', f'{pip}-2', ]
#         resp_vectors = [concatenated_event_responses_hipp_only[e][:,-5:].mean(axis=1) for e in pips_2_comp]
#         sim_dev_to_norms = cosine_similarity(resp_vectors)
#         # sim_mat_plot = plot_similarity_mat(sim_dev_to_norms,pips_2_comp,'Greys',im_kwargs=dict(vmax=1,vmin=0.6,)),
#         # sim_mat_plot[0][0].show()
#         # sim_mat_plot[0][0].savefig(dev_ABBA1_figdir / f"sim_mat_new_{'_'.join(pips_2_comp)}_grays.pdf")
#         dev_comps[pip] = sim_dev_to_norms[0, 1:]
#     abcd_abba1_com_by_name[animal] = dev_comps
#     resp_vectors = [concatenated_event_responses_hipp_only[e][:,-5:].mean(axis=1) for e in ['D-1','D-0','C-0']]
#     sim_dev_to_norms = cosine_similarity(resp_vectors)
#
#     dev_comp_plot = plt.subplots()
#     for pip_i, (pip,pip_sims) in enumerate(dev_comps.items()):
#         [dev_comp_plot[1].scatter(pip_i+offset, sim, label=lbl,c=c,s=50)
#          for offset,sim,lbl,c in zip([-0.1,0.1],pip_sims,['ABCD(0)','ABBA(1)',],['blue','red'])]
#     # format_axis(dev_comp_plot[1],vlines=list(range(len(dev_comps))),ls='--',lw=0.2)
#     dev_comp_plot[1].set_ylabel('Cosine similarity')
#     # dev_comp_plot[1].set_xlabel('unit')
#     # dev_comp_plot[1].legend()
#     # dev_comp_plot[1].set_yticks([0.6,0.7])
#     # dev_comp_plot[1].set_yticklabels([0.6,0.7])
#     dev_comp_plot[1].set_xticks(list(range(len(dev_comps))))
#     dev_comp_plot[1].set_xticklabels(list(dev_comps.keys()))
#     dev_comp_plot[0].set_layout_engine('constrained')
#     dev_comp_plot[0].set_size_inches(2, 2)
#     # dev_comp_plot[0].show()
#     # dev_comp_plot[0].savefig( f"dev_comp_scatter_new_{animal}.pdf")
#
# # all animal dev comp plot
# dev_comp_all_animals_plot = plt.subplots()
# dev_comp_df = pd.DataFrame.from_dict(abcd_abba1_com_by_name).T
# [[dev_comp_all_animals_plot[1].errorbar(pip_i+pos,np.mean([sims[sim_i] for sims in dev_comp_df[pip]]),
#                                        yerr=sem([sims[sim_i] for sims in dev_comp_df[pip]]),
#                                        c=c,label=lbl,capsize=20,fmt='o')
#  for pip_i,pip in enumerate(dev_comp_df)]
#  for sim_i, (lbl,c,pos) in enumerate(zip(['ABCD(0)','ABBA(1)',][:],['#00008bff','#ff8610ff'],
#                                          [-.05,0.05]))]
#
# [dev_comp_all_animals_plot[1].plot(np.arange(dev_comp_df.shape[1])+pos,
#     dev_comp_df.explode(dev_comp_df.columns.tolist()).iloc[sim_i::2].mean(axis=0), c=c)
#     for sim_i, (lbl,c,pos) in enumerate(zip(['ABCD(0)','ABBA(1)',][:],['#00008bff','#ff8610ff'],
#                                          [-.05,0.05]))]
# dev_comp_all_animals_plot[1].set_xticks(list(range(dev_comp_df.shape[1])))
# dev_comp_all_animals_plot[1].set_xticklabels([f'pip {i}' for i in range(dev_comp_df.shape[1])])
# unique_legend(dev_comp_all_animals_plot)
# dev_comp_all_animals_plot[0].show()
# dev_comp_all_animals_plot[0].set_size_inches(2.5,2.5)
# dev_comp_all_animals_plot[0].set_layout_engine('tight')
# # assert False
# dev_comp_all_animals_plot[0].savefig('all_mice_similarity_comp.pdf')
#
# # ttest
# dev_comp_df_by_rule = [dev_comp_df.explode(dev_comp_df.columns.tolist()).iloc[sim_i::2].astype(float)
#                        for sim_i,_ in enumerate(['ABCD(0)','ABBA(1)'])]
# ttest_ind(dev_comp_df_by_rule[0],dev_comp_df_by_rule[1],equal_var=True,alternative="greater")
# ttest_ind(dev_comp_df_by_rule[0]['D'],dev_comp_df_by_rule[0]['A'],equal_var=True,alternative="greater")
# ttest_ind(dev_comp_df_by_rule[1]['D'],dev_comp_df_by_rule[1]['A'],equal_var=True,alternative="greater")
# ttest_ind(dev_comp_df_by_rule[1]['A'],dev_comp_df_by_rule[0]['A'],equal_var=True,alternative="greater")
#
full_pattern_responses = joblib.load(r"D:\ephys\abstraction_resps_ephys_no_zscore_2401_2504.joblib")
full_pattern_responses = zscore_by_trial(full_pattern_responses)
# # full_pattern_responses = {k: {kk:np.cumsum(vv,axis=-1) for kk,vv in v.items()} for k,v in full_pattern_responses.items()}

plot_config_path = Path(r'H:\ephys\abstraction_plot_config.yaml')
with open(plot_config_path, 'r') as f:
    plot_config = yaml.load(f, Loader=yaml.FullLoader)

plt.style.use('figure_stylesheet.mplstyle')
plt.rcParams['axes.prop_cycle'] = cycler(color=['k','#e8739bff','#e47b15ff'])

aggr_resps = AggregateSession(resps_pkl.parent,None,plot_config,plot_config['pips_2_plot'])
aggr_resps.aggregate_mean_sess_responses( filt_by_trial_num=True,
                                          pip_4_filt=[['A-1']], pip_2_filt=['A-0',],
                                          filt_func=lambda tn, ft: [max(tn[tn < e] - e) + e for e in ft],
                                          reload_save=False, concat_savename=Path('aggr_abstr.joblib')
                                          )
aggr_resps.pca_pseudo_pop(pca_name='abstraction_pca')

# for pca_combs in plot_config['abstraction_pca']['pcas2plot']:
for pca_combs in combinations(list(range(5)),3):
    aggr_resps.plot_3d_pca('abstraction_pca', pca_combs, Path(r'X:\Dammy\figures\pca_plots'),
                                plot_config['abstraction_pca'])

# use non zscored resps
zscore_by_trial_resps = {}
for sess, sess_resps in full_pattern_responses.items():
    zscore_by_trial_resps[sess] = {}
    for pip, pip_resps in sess_resps.items():
        zscore_by_trial_resps[sess][pip] = (pip_resps-pip_resps.mean(axis=-1,keepdims=True))/pip_resps.std(axis=-1,keepdims=True)

concat_resps_4_pca = {
        e: np.concatenate([np.nanmean(zscore_by_trial_resps[sessname][e],axis=0) if e != 'A-0' else
                           np.nanmean(zscore_by_trial_resps[sessname][e][100:115],axis=0)
                           for sessname in zscore_by_trial_resps])
        for e in list(zscore_by_trial_resps.values())[0].keys()}
# remove nans by row
resps_stacked = np.concatenate(list(concat_resps_4_pca.values()),axis=1)
resps_stacked_no_nans = resps_stacked[~np.any(np.isnan(resps_stacked),axis=1)]
resps_unstacked = np.split(resps_stacked_no_nans,3,axis=1)
for pip, no_nan_resps in zip(concat_resps_4_pca, resps_unstacked):
    concat_resps_4_pca[pip] = no_nan_resps
resps_4_pca = {'by_class':concat_resps_4_pca}

full_pattern_pca = PopPCA(resps_4_pca)
# full_pattern_pca.eig_vals[2][0].show()
full_pattern_pca.get_trial_averaged_pca(standardise=False)
full_pattern_pca.get_projected_pca_ts(standardise=False)
window = plot_config['window']
full_pattern_pca.plot_pca_ts(window, fig_kwargs={'figsize': (120, 8)},
                             plot_separately=False, n_comp_toplot=5,
                             lss=['-', '--', '-', '--'], plt_cols=['C' + str(i) for i in [0, 0, 1, 1]]
                             )
# plot 3d projections
x_ser = np.round(np.linspace(*window, concat_resps_4_pca['A-0'].shape[-1]), 2)
full_pattern_pca.plot_3d_pca_ts('by_class', [-1, 2], x_ser=x_ser, smoothing=1, pca_comps_2plot=[0,1,3], t_end=1,
                                plot_out_event=False,
                                scatter_times=[0.5],scatter_kwargs={'marker':'*','s':50,'c':'k'})
# save 3d plot
# full_pattern_pca.proj_3d_plot.savefig(f'full_pattern_norm_dev_pca_3d_plot_aggregate_sessions.pdf')
full_pattern_pca.plot_2d_pca_ts('by_class', [-1, 2], x_ser=x_ser, smoothing=2, pca_comps_2plot=[0,1], t_end=1.1,
                                plot_out_event=False,
                                scatter_times=[0.5],scatter_kwargs={'marker':'*','s':50,'c':'k'})
full_pattern_pca.scatter_pca_points('by_class',[0.1],x_ser)
full_pattern_pca.scatter_plot[0].show()

sim_win_size = 5
sim_to_comp_vector_by_animal = {}
pip = 'A'
pips_2_comp = [f'{pip}-1', f'{pip}-0', f'{pip}-2', ]

for animal in ['DO81', 'DO79', 'DO95', 'DO97', ]:    # sim using full population
    concatenated_event_responses_hipp_only = {}

    # loop over events (keys from the first session dict)
    for e in list(zscore_by_trial_resps.values())[0].keys():
        collected = []  # to hold all session responses for this event

        for sessname in zscore_by_trial_resps:
            if animal in sessname:
                if e != 'A-0':
                    # standard mean across trials
                    vals = zscore_by_trial_resps[sessname][e].mean(axis=0)
                else:
                    # bootstrap mean from index 100 onwards
                    data = zscore_by_trial_resps[sessname][e][100:]  # restrict to index >= 100
                    n_trials = data.shape[0]
                    # bootstrap: pick 20 trials with replacement
                    sample_idxs = [np.random.choice(n_trials, size=15, replace=True) for _ in range(100)]
                    sampled = np.mean([data[sample_idx] for sample_idx in sample_idxs],axis=0)
                    vals = sampled.mean(axis=0)  # mean across bootstrapped trials

                collected.append(vals)

        concatenated_event_responses_hipp_only[e] = np.concatenate(collected)
    # remove nans
    resps_stacked = np.concatenate(list(concatenated_event_responses_hipp_only.values()), axis=1)
    resps_stacked_no_nans = resps_stacked[~np.any(np.isnan(resps_stacked), axis=1)]
    resps_unstacked = np.split(resps_stacked_no_nans, 3, axis=1)
    for pip, no_nan_resps in zip(concat_resps_4_pca, resps_unstacked):
        concatenated_event_responses_hipp_only[pip] = no_nan_resps
    # concatenated_event_responses_hipp_only = {e: resps[[animal in s for s in aggr_resps.concatenated_sessnames]]
    #                                           for e, resps in aggr_resps.concatenated_event_responses.items()}
    #
    comp_pip_resp = concatenated_event_responses_hipp_only[pips_2_comp[0]]
    comp_pip_vector = comp_pip_resp[:,-50:50-sim_win_size].mean(axis=1)

    sim_to_comp_vector_ts = []
    resp_x_ser = np.arange(sim_win_size,comp_pip_resp.shape[1])
    for t in tqdm(resp_x_ser, total=len(resp_x_ser), desc=f'{animal} sim over t'):
        resp_vectors = [concatenated_event_responses_hipp_only[e][:,t:t+sim_win_size].mean(axis=1)
                        for e in pips_2_comp]
        # sim_to_comp_vector_ts.append(cosine_similarity([comp_pip_vector]+resp_vectors)[0,1:])
        sim_to_comp_vector_ts.append(cosine_similarity([comp_pip_resp[:,t:t+sim_win_size].mean(axis=1)]+resp_vectors)[0,1:])
        # sim_to_comp_vector_ts.append(cosine_similarity([comp_pip_resp[:,t:t+sim_win_size].mean(axis=1)]
        #                                                +resp_vectors)[0,1:])

    sim_to_comp_vector_by_animal[animal] = pd.DataFrame(np.array(sim_to_comp_vector_ts).T,
                                                        index=pips_2_comp)

sim_to_comp_vector_df = pd.concat(list(sim_to_comp_vector_by_animal.values()))
resp_x_ser_s = np.round(np.arange(-0.5,1.5+0.01,0.01,),2)
sim_to_comp_vector_df.columns = resp_x_ser_s[resp_x_ser]

sim_over_t_plot = plt.subplots()

[(sim_over_t_plot[1].plot(sim_to_comp_vector_df.loc[pip].mean(axis=0),c=col,label=pip),
 plot_shaded_error_ts(sim_over_t_plot[1],sim_to_comp_vector_df.columns,
                      sim_to_comp_vector_df.loc[pip].mean(axis=0),
                      sim_to_comp_vector_df.loc[pip].sem(axis=0),
                      fc=col,alpha=0.1))
 for pip,col in zip(pips_2_comp[1:], ['#e8739bff','k','#e47b15ff'][1:])]
format_axis(sim_over_t_plot[1],vspan=[(t,t+0.15) for t in np.arange(0,1,0.25)])
sim_over_t_plot[1].legend()
sim_over_t_plot[0].show()


pvp_dec_dict = {}
# pips2decode = [['A-0', 'A-1'], ['A-0', 'A-2'], ['A-1', 'A-2'],['A-0','A-0'],['A-0','A-1','A-2'],['A-0','A-1;A-2']]
pips2decode = [['A-0', 'A-1'], ['A-0', 'A-2'], ['A-1', 'A-2'],['A-0','A-1','A-2']]
resp_x_ser = np.round(np.linspace(-0.25,1,list(full_pattern_responses.values())[0][pips2decode[0][0]].shape[-1]),2)
window_s = [0.8,1.25]
window_idxs = np.logical_and(resp_x_ser >= window_s[0], resp_x_ser <= window_s[1])
for pips in pips2decode:
    dec_sffx = "_vs_".join(pips)
    for sessname in tqdm(full_pattern_responses.keys(), desc='decoding',
                         total=len(full_pattern_responses.keys())):
        xs_list = []
        if pips[0] == pips[1]:
            # split in half
            xs_list = np.array_split(full_pattern_responses[sessname][pips[0]][30:90],2)
        else:
            for pip in pips:
                # norm_idxs = np.logical_or(*[norm_idxs_by_dev_pip[pip][sessname] for pip in ['A-1', 'A-2']])
                if pip == 'A-0' and any([p in sum(pips2decode,[]) for p in ['A-1', 'A-2']]):
                    # xs_list.append(full_pattern_responses[sessname][pip][norm_idxs])
                    xs_list.append(full_pattern_responses[sessname][pip][100:300][::15])
                else:
                    xs_list.append(full_pattern_responses[sessname][pip])
                    # xs_list.append(full_pattern_responses[sessname]['A-0'][norm_idxs])
            if any([x.shape[0]<8 for x in xs_list]):
                continue
        xs = np.vstack([x[:, :, window_idxs].max(axis=-1) for x in xs_list])
        ys = np.hstack([np.full(x.shape[0], ci) for ci, x in enumerate(xs_list)])
        pvp_dec_dict[f'{sessname}_{dec_sffx}'] = decode_responses(xs, ys,dec_kwargs={'cv_folds':4,'n_runs':20})
        pvp_dec_dict[f'{sessname}_{dec_sffx}']['data'].plot_confusion_matrix(dec_sffx.split('_vs_'),)
    # plot accuracy
    pvp_accuracy = np.array([pvp_dec_dict[dec_name]['data'].accuracy
                             for dec_name in pvp_dec_dict.keys() if dec_sffx in dec_name])
    pvp_shuffle_accs = np.array([pvp_dec_dict[dec_name]['shuffled'].accuracy
                                 for dec_name in pvp_dec_dict.keys() if dec_sffx in dec_name])
    pvp_accuracy_plot = plt.subplots()
    pvp_accuracy_plot[1].boxplot([pvp_accuracy.mean(axis=1),
                                  pvp_shuffle_accs.mean(axis=1)], labels=['data', 'shuffle'],
                                 showmeans=False, meanprops=dict(mfc='k'), )
    pvp_accuracy_plot[1].set_ylabel('Accuracy')
    pvp_accuracy_plot[1].set_title(f'Accuracy of {dec_sffx}')
    format_axis(pvp_accuracy_plot[1],hlines=[0.5])
    pvp_accuracy_plot[0].show()
    pvp_accuracy_plot[0].savefig(f'{dec_sffx}_accuracy.pdf')
    # ttest
    ttest = ttest_ind(pvp_accuracy.mean(axis=1), pvp_shuffle_accs.mean(axis=1),
                      alternative='greater', equal_var=False)
    print(f'{dec_sffx} ttest: pval =  {ttest[1]}. Mean accuracy of {pips} is {pvp_accuracy.mean():.3f}.')

cmap_norm_3way = TwoSlopeNorm(vmin=0.2, vcenter=0.33333, vmax=0.45)
all_v_all_tag = "_vs_".join(pips2decode[-2])
all_cms = np.array([pvp_dec_dict[dec_name]['data'].cm for dec_name in pvp_dec_dict if all_v_all_tag in dec_name])
cm_plot = plot_aggr_cm(all_cms, im_kwargs=dict(norm=cmap_norm_3way), include_values=True,
                       labels=['A-0','A-1', 'A-2'], cmap='bwr')

cm_plot[0].set_size_inches((3, 3))
cm_plot[0].show()


# concatenated_event_responses_all = {
#         e: np.concatenate([event_responses[sessname][e].mean(axis=0) if '0' not in e else
#                            event_responses[sessname][e][100:115].mean(axis=0)
#                            for sessname in event_responses
#                            if animal in sessname])
#         for e in list(event_responses.values())[0].keys()}
concatenated_event_responses_all = aggr_resps.concatenated_event_responses

reducer = umap.UMAP(
    n_neighbors=5,     # adjust for local vs global structure
    min_dist=1,       # smaller → more clustered
    n_components=3,     # 2D projection
    random_state=42
)
embedding_start = reducer.fit_transform(StandardScaler().fit_transform(np.hstack([resp[:,-5:].mean(axis=1)
                                                   for pip,resp in concatenated_event_responses_all.items()
                                                   if 'A' in pip]).reshape(-1, 1)))
embedding_end = reducer.fit_transform(StandardScaler().fit_transform(np.hstack([resp[:,-5:].mean(axis=1)
                                                 for pip, resp in concatenated_event_responses_all.items()
                                                 if 'D' in pip]).reshape(-1, 1)))

labels = [[ptype]*concatenated_event_responses_all[pip].shape[0]
          for pip,ptype in zip(['A-0', 'A-1','A-2'],['ABCD(0)','ABCD(1)','ABBA(1)',])]
labels = sum(labels, [])
cols = [[col]*concatenated_event_responses_all[pip].shape[0]
          # for pip,col in zip(['A-0', 'A-1','A-2'],['k','#00008bff','#ff8610ff'])]
          for pip,col in zip(['A-0', 'A-1','A-2'],['C0','C1','C2'])]

cols = sum(cols, [])

umap_plot = plt.subplots(ncols=2)
for i, embedding in enumerate([embedding_start, embedding_end]):
    for ptype, col in zip(['ABCD(0)', 'ABCD(1)', 'ABBA(1)', ], ['C0','C1','C2']):
        mask = np.array(labels)==ptype
        ii = labels.index(ptype)
        umap_plot[1][i].scatter(embedding[:,0][mask][0],embedding[:,2][mask][0],
                                 color=col,label=ptype,alpha=0.7)
umap_plot[0].set_layout_engine('tight')
umap_plot[0].show()


# pca by sess

pca_by_sess_by_pip = {sess: {pip_i: PCA(n_components=20).fit_transform(np.vstack([resps[:,:,-5:].mean(axis=-1)
                                                                                for pip, resps in sess_resps.items()
                                                                                if pip_i in pip]))
                             for pip_i in list('ABCD')}
                      for sess,sess_resps in event_responses.items()}
labels_by_trial_by_pip = {sess: {pip_i: np.hstack([[pip]*resps.shape[0] for pip, resps in sess_resps.items()
                                                  if pip_i in pip])
                             for pip_i in list('ABCD')}
                          for sess,sess_resps in event_responses.items()}

aligned_pcas_by_pip = {}
for pip_i in list('ABCD'):
    pcas_cross_sess = [pcas[pip_i] for pcas in list(pca_by_sess_by_pip.values())]
    # aligned_pcas = [pca @ orthogonal_procrustes(pcas_cross_sess[0][-200:],pca[-200:])[0] for pca in pcas_cross_sess[1:]]
    aligned_pcas = [CCA(n_components=20).fit_transform(pcas_cross_sess[0][-200:],pca[-200:])[1] for pca in pcas_cross_sess[1:]]
    aligned_pcas_by_pip[pip_i] = [pcas_cross_sess[0][-200:]]+aligned_pcas

# plot pca
for pca_comp in list(combinations(range(3),2)):
    aligned_pca_plot = plt.subplots(ncols=len(aligned_pcas_by_pip))
    unique_pip_lbls = {pip_i:list(set(list(labels_by_trial_by_pip.values())[0][pip_i])) for pip_i in list('ABCD')}
    for i, pip_i in enumerate(aligned_pcas_by_pip):
        for ptype_i, ptype in enumerate(sorted(unique_pip_lbls[pip_i])):
            mean_ptype_by_sess = [pca[(lbls[pip_i]==ptype)[-200:]].mean(axis=0) for pca,lbls in zip(aligned_pcas_by_pip[pip_i],
                                                                                            list(labels_by_trial_by_pip.values()))]
            mean_ptype_by_sess = np.array(mean_ptype_by_sess)
            aligned_pca_plot[1][i].scatter(mean_ptype_by_sess[:,pca_comp[0]],mean_ptype_by_sess[:,pca_comp[1]],
                                        label=ptype,alpha=0.2,c=f'C{ptype_i}')
    aligned_pca_plot[0].set_layout_engine('tight')
    unique_legend((aligned_pca_plot[0],aligned_pca_plot[1][0]))
    aligned_pca_plot[0].show()

 # check patt ids
e_f = joblib.load(r"D:\ephys\abstraction_resps_ephys_2401_2504_features.joblib")
all_feats_by_stim = [pd.concat([s[pip]['td_df'] for s in list(e_f.values())]) for pip in ['A-0','A-1','A-2']]
for df in all_feats_by_stim:
    print(df['PatternID'].unique())


import slicetca
import torch
import scipy.ndimage as spnd


device = ('cuda' if torch.cuda.is_available() else 'cpu')

# your_data is a numpy array of shape (trials, neurons, time).
sess_neural_data = list(full_pattern_responses.values())[10]

neural_data = np.concatenate([sess_neural_data[e] if e != 'A-0' else
                               sess_neural_data[e][100:115]
                               for e in sess_neural_data])

all_pip_lbls = np.hstack([np.full(sess_neural_data[e].shape[0],ei) if e != 'A-0' else
                          np.full(sess_neural_data[e][100:115].shape[0],ei)
                          for ei,e in enumerate(sess_neural_data)])

neural_data = spnd.gaussian_filter1d(neural_data, sigma=2, axis=-1)
neural_data = np.array([d / d.max() for d in np.array([d - d.min() for d in neural_data])])

neural_data_tensor = torch.tensor(neural_data, dtype=torch.float, device=device)
neural_data_tensor = neural_data_tensor/neural_data_tensor.std()

components, model = slicetca.decompose(neural_data_tensor,
                                       number_components=(4,2,1),
                                       positive=True,
                                       learning_rate=5*10**-3,
                                       min_std=10**-5,
                                       max_iter=10000,
                                       seed=0)

# we sort the neurons of the trial slices according to their peak activity in the first slice.
neuron_sorting_peak_time = np.argsort(np.argmax(components[0][1][0], axis=1))
trial_colors = np.array([matplotlib.colormaps['gist_rainbow'](np.mod((i + 3*np.pi)/(np.pi * 2),1))[:3]
                         for i in all_pip_lbls])


# call plotting function, indicating index for sorting trials and colors for different angles as well as time
axes = slicetca.plot(model,
              variables=('trial', 'neuron', 'time'),
              colors=(trial_colors, None, None), # we only want the trials to be colored
              ticks=(None, None, np.linspace(0,neural_data.shape[-1],5)), # we only want to modify the time ticks
              tick_labels=(None, None, np.linspace(-0.5,1.5,5)),
              sorting_indices=(all_pip_lbls, neuron_sorting_peak_time, None),
              quantile=0.99)
plt.show()

reconstruction_full = model.construct().numpy(force=True)
# reconstruct from the trial-slicing partition (i.e., sum of all 4 trial-slicing component reconstructions)
reconstruction_trial_slicing = model.construct_single_partition(partition=0).numpy(force=True)

# construct single time-slicing component
reconstruction_time_slicing = model.construct_single_component(partition=2, k=0).numpy(force=True)
fig,ax = plt.subplots()
for trial, col in zip(reconstruction_trial_slicing,all_pip_lbls):
    ax.plot(np.linspace(-0.5,1.5,trial.shape[-1]),
            trial.mean(axis=0),c=f'C{col}',lw=.5)
fig.show()

fig,ax = plt.subplots()
for col in np.unique(all_pip_lbls):
    ax.plot(np.linspace(-0.5,1.5,trial.shape[-1]),
            reconstruction_full[all_pip_lbls==col].mean(axis=0).mean(axis=0),c=f'C{col}',lw=.5)
fig.show()

trajs = [reconstruction_full[all_pip_lbls==col] for col in np.unique(all_pip_lbls)]
sim_by_t = [cosine_similarity([e[:,t] for e in reconstruction_full]) for t in [50,75,100,125,150,200]]
from neural_similarity_funcs import plot_similarity_mat
for sim,tit in zip(sim_by_t, [50,75,100,125,150,200]):
    sim_plot = plot_similarity_mat(sim,np.array(all_pip_lbls),cmap='Reds',im_kwargs={'vmin':0.85,'vmax':1})
    sim_plot[1].set_title(tit)
    sim_plot[0].show()

train_mask, test_mask = slicetca.block_mask(dimensions=neural_data.shape,
                                            train_blocks_dimensions=(1, 1, 10), # Note that the blocks will be of size 2*train_blocks_dimensions + 1
                                            test_blocks_dimensions=(1, 1, 5), # Same, 2*test_blocks_dimensions + 1
                                            fraction_test=0.1,
                                            device=device)

loss_grid, seed_grid = slicetca.grid_search(neural_data_tensor,
                                            min_ranks = [3, 0, 0],
                                            max_ranks = [5, 2, 2],
                                            sample_size=4,
                                            mask_train=train_mask,
                                            mask_test=test_mask,
                                            processes_grid=4,
                                            seed=1,
                                            min_std=10**-4,
                                            learning_rate=5*10**-3,
                                            max_iter=10**4,
                                            positive=True)