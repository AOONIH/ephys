from ephys_analysis_funcs import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
def get_population_pca(rate_arr:np.ndarray):

    assert rate_arr.ndim == 3
