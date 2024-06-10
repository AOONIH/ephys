import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from ephys_analysis_funcs import plot_2d_array_with_subplots



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


def get_reordered_idx(pip_desc,pip_lbls,sort_keys,subset=None):
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