import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from ephys_analysis_funcs import plot_2d_array_with_subplots
from itertools import combinations
from tqdm import tqdm
from copy import deepcopy as copy


def plot_similarity_mat(sim_mat: np.ndarray, pip_lbls: [str, ], cmap='viridis', plot=None, reorder_idx=None,plot_cbar=True,
                        im_kwargs=None,) -> (plt.Figure,plt.Axes):
    if reorder_idx:
        assert len(reorder_idx) == sim_mat.shape[0] and len(reorder_idx) == len(set(reorder_idx))
        sim_mat = sim_mat[reorder_idx, :]
        sim_mat = sim_mat[:, reorder_idx]
    # sim_mat = np.ma.array(sim_mat,np.tri(sim_mat.shape[0], k=-1))
    cmap = matplotlib.cm.get_cmap(cmap)
    cmap.set_bad(color='white')
    similarity_plot = plot_2d_array_with_subplots(sim_mat, cmap=cmap, cbar_height=sim_mat.shape[0], plot=plot,
                                                  plot_cbar=plot_cbar,
                                                  **im_kwargs if im_kwargs else {})
    similarity_plot[1].set_xticks(np.arange(len(pip_lbls)), labels=pip_lbls)
    similarity_plot[1].set_xticks(np.arange(start=-0.5,stop=len(pip_lbls)), minor=True)
    similarity_plot[1].set_yticks(np.arange(len(pip_lbls)), labels=pip_lbls)
    similarity_plot[1].set_yticks(np.arange(start=-0.5,stop=len(pip_lbls)), minor=True)
    similarity_plot[1].invert_yaxis()
    similarity_plot[1].grid(which='minor', color='k',lw=1,alpha=0.25)
    # similarity_plot[0].set_size_inches(10, 8)
    return similarity_plot


def get_sim_mat_over_time(pop_rate_mat_list:list, mean=np.mean, mean_axis=0,t_dim=2):
    x_ser = np.arange(pop_rate_mat_list[0].shape[t_dim])
    sim_over_time = [cosine_similarity([mean(pop_rate_mat[:,:,t], axis=mean_axis)
                                        for pop_rate_mat in pop_rate_mat_list])
                     for t in x_ser]
    sim_over_time = [A[~np.eye(A.shape[0],dtype=bool)].reshape(A.shape[0],-1) if A.shape[0] > 1 else A
                     for A in sim_over_time]
    return np.array(sim_over_time)


def get_list_pips_by_property(pip_desc:dict, property_name:str, pip_positions: [int]):
    list_pip_groups_by_idx = [[pip for pip in pip_desc if pip_desc[pip][property_name] == prop_val and
                               pip_desc[pip]['position'] in pip_positions]
                              for prop_val in np.unique([pip_desc[pip][property_name] for pip in pip_desc
                                                         if pip_desc[pip]['position'] in pip_positions])]
    return list_pip_groups_by_idx


def get_reordered_idx(pip_desc,sort_keys,subset=None):
    pip_lbls = list(pip_desc.keys())
    if not subset:
        subset = pip_lbls
    plot_names = [pip_desc[i]['desc']
                  for i in [p for p in sorted(pip_desc, key=lambda x: [pip_desc[x][sort_key]
                                                                       for sort_key in sort_keys])
                            if p in subset]]
    plot_order = [pip_lbls.index(i)
                  for i in [p for p in sorted(pip_desc, key=lambda x: [pip_desc[x][sort_key]
                                                                       for sort_key in sort_keys])
                            if p in subset]]
    return plot_names, plot_order


def compute_self_similarity(pop_rate_mat:np.ndarray, t=-1, cv_folds=5,mean_flag=False):
    assert cv_folds > 1, 'cv_folds must be > 1'
    all_splits = list(combinations(range(cv_folds),cv_folds-1))
    split_pop_mats = np.array_split(pop_rate_mat, cv_folds)[:cv_folds]
    all_train_mats = [[split_pop_mats[i].mean(axis=0) for i in split] for split in all_splits]
    # [print(e.shape) for e in all_train_mats[0]]
    all_test_mats = [[split_pop_mats[i].mean(axis=0) for i in range(cv_folds) if i not in split]
                     for split in all_splits]
    fold_sims = [[cosine_similarity([e[:,t] if not mean_flag else e.mean(axis=1)
                                     for e in [np.mean(train_mats,axis=0), np.mean(test_mats,axis=0)]])[0,1]]
                 for train_mats, test_mats in zip(all_train_mats, all_test_mats)]

    return fold_sims


def compare_pip_sims_2way(pop_rate_mats: [np.ndarray,np.ndarray], n_shuffles=1000, t=-1, mean_flag=True):
    # assert len(pop_rate_mats) == 2
    self_sims_by_halves = [[[cosine_similarity([(np.squeeze(e).mean(axis=0) if mean_flag else np.squeeze(e))[:, t]
                                                for e in np.array_split(rate_mat[shuffle],2)])]
                            for shuffle in tqdm([np.random.permutation(rate_mat.shape[0])
                                                 for _ in range(n_shuffles)], desc='shuffle self sims', total= n_shuffles,
                                                disable=True)
                            ]
                           for rate_mat in pop_rate_mats]

    self_sims_by_halves = np.squeeze(np.array(self_sims_by_halves))
    shuffled_idxs = [[np.random.permutation(rate_mat.shape[0]) for rate_mat in pop_rate_mats] for _ in range(n_shuffles)]

    # [print([np.array_split(e[idx],2)[0].mean(axis=0)[:,-1].shape
    #                     for e, idx in zip(pop_rate_mats, idxs)])
    #  for idxs in tqdm(shuffled_idxs, desc='shuffle across sims', total=n_shuffles)]
    if len(pop_rate_mats) == 1:
        return self_sims_by_halves, None
    across_sims_by_halves = [cosine_similarity([(np.squeeze(e)[idx[::2]].mean(axis=0) if mean_flag else
                                                 np.squeeze(e)[idx[::2]])[:,t]
                                                for e, idx in zip(pop_rate_mats, idxs) ])
                             for idxs in tqdm(shuffled_idxs, desc='shuffle across sims', total=n_shuffles)]

    across_sims_by_halves = np.squeeze(np.array(across_sims_by_halves))

    return self_sims_by_halves, across_sims_by_halves,shuffled_idxs


def plot_sim_by_pip(event_psth_dict, sim_mat, fig, axes, pip_desc, cmap='bwr',im_kwargs=None):
    for pi, pip2use in enumerate('ABCD'):
        pip_is = [ii for ii, p in enumerate(event_psth_dict) if pip2use in p]
        subset_pips = [p for p in event_psth_dict if pip2use in p]
        similarity_to_pip = sim_mat[pip_is][:, pip_is]
        reordered_names, reordered_idxs = get_reordered_idx(pip_desc, ['ptype_i'],
                                                            subset=subset_pips)
        reordered_idxs = [int(idx / 4) for idx in reordered_idxs]
        pip_plot_lbls = [e.split(' ')[-1] for e in reordered_names]
        plot_similarity_mat(similarity_to_pip, pip_plot_lbls,
                            reorder_idx=reordered_idxs,
                            cmap=cmap, plot=(fig, axes[pi]),
                            im_kwargs=im_kwargs,
                            plot_cbar=True if pi == len('ABCD') - 1 else False)
        axes[pi].set_title(f'pip {pi}')


def plot_sim_by_grouping(sim_mat,grouping,pip_desc,cmap='Reds',plot=None,im_kwargs=None):
    if plot is None:
        plot = plt.subplots()
    # sort_keys = grouping
    plot_names,plot_order = get_reordered_idx(pip_desc, grouping)
    sim_plot = plot_similarity_mat(sim_mat, plot_names, cmap=cmap,reorder_idx=plot_order,im_kwargs=im_kwargs,
                                   plot=plot)

    return sim_plot
