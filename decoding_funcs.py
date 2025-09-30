import multiprocessing
import os
from pathlib import Path

from joblib import Parallel, delayed

from functools import partial

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, StratifiedShuffleSplit
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, LogisticRegression
from sklearn import svm
import pandas as pd
from tqdm import tqdm

from io_utils import load_sess_pkl
from plot_funcs import plot_decoder_accuracy


class Decoder:
    def __init__(self, predictors, features, model_name, ):
        self.predictors = predictors
        self.features = features
        self.model_name = model_name
        self.models = None
        self.predictions = None
        self.accuracy = None
        self.fold_accuracy = None
        self.accuracy_plot = None
        self.cm = None
        self.cm_plot = None
        self.prediction_ts = None

    def decode(self, dec_kwargs, parallel_flag=False, **kwargs):
        # if not dec_kwargs.get('cv_folds', 0):
        kwargs = {**kwargs, **dec_kwargs}
        n_runs = kwargs.get('n_runs', dec_kwargs.get('n_runs', 500))
        # else:
        #     n_runs = kwargs.get('n_runs', 1)
        # cv_folds = kwargs.get('cv_folds',None)
        # if not kwargs.get('balance',False):
        if not isinstance(self.predictors, (list | tuple)):
            _preds_list = [self.predictors[self.features==ftr] for ftr in np.unique(self.features)]
            _feats_list = [self.features[self.features==ftr] for ftr in np.unique(self.features)]
            # preds, feats = [self.predictors] * n_runs, [self.features] * n_runs
            if kwargs.get('balance_predictors',True):
                n_run_pred = [balance_predictors(_preds_list,_feats_list) for _ in range(n_runs)]
                preds = [run[0] for run in n_run_pred]
                feats = [run[1] for run in n_run_pred]
            else:
                preds = [self.predictors]*n_runs
                feats = [self.features]*n_runs

        else:
            n_run_pred = [balance_predictors(self.predictors, self.features) for i in range(n_runs)]
            preds = [run[0] for run in n_run_pred]
            feats = [run[1] for run in n_run_pred]
        if parallel_flag:
            print('running in parallel')

            # Stack all shuffles: shape = (100, n_samples, n_features)
            preds_arr = np.stack(preds)
            feats_arr = np.stack(feats)

            temp_dir = dec_kwargs.get('temp_dir', 'D:')
            temp_dir = Path(temp_dir)
            preds_fn = temp_dir / "shuffled_preds.dat"
            feats_fn = temp_dir /  "shuffled_feats.dat"
            np.memmap(preds_fn, dtype='float64', mode='w+', shape=preds_arr.shape)[:] = preds_arr
            np.memmap(feats_fn, dtype='float64', mode='w+', shape=feats_arr.shape)[:] = feats_arr

            results = Parallel(n_jobs=os.cpu_count()-1)(
                delayed(run_decoder)(model=self.model_name,predictors=None, features=None,
                                     i=i, shape=preds[0].shape,preds_fn=preds_fn, feats_fn=feats_fn,
                                     **dec_kwargs)
                for i in range(preds_arr.shape[0])
            )

            # with multiprocessing.Pool(initializer=init_pool_processes) as pool:
            #     results = list(tqdm(pool.starmap(partial(run_decoder, model=self.model_name,
            #                                              predictors=preds, features=feats,
            #                                              **dec_kwargs),
            #                                      # zip(preds, feats)
            #                                      range(n_runs)),
            #                         total=n_runs))
        else:
            results = [run_decoder(model=self.model_name,predictors=preds[run_i], features=feats[run_i],
                                                         **dec_kwargs)
                       for run_i in tqdm(range(n_runs), total=n_runs, desc='Decoding single threaded',disable=True)]
        # results = []
        # register_at_fork(after_in_child=np.random.seed)
        # with multiprocessing.Pool(initializer=np.random.seed) as pool:
        #     results = list(tqdm(pool.imap(partial(run_decoder, features=self.features, shuffle=to_shuffle,
        #                                           model=self.model_name, cv_folds=cv_folds),
        #                                   [self.predictors] * 1), total=n_runs))
        # for n in range(n_runs):
        #     results.append(run_decoder(self.predictors,self.features,model=self.model_name,**dec_kwargs))
        self.accuracy = [res[0] for res in results]
        self.models = [res[1] for res in results]
        self.fold_accuracy = [list(res[2] for res in results)]
        self.predictions = [list(res[3]) for res in results]

    def map_decoding_ts(self, t_ser, model_i=0, y_lbl=0):
        dec_models = np.array(self.models).flatten()
        # with multiprocessing.Pool() as pool:
        #     results = tqdm(pool.imap())
        self.prediction_ts = np.array([m.predict(t_ser.T) for m in dec_models])
        if isinstance(y_lbl, list):
            assert len(y_lbl) == self.prediction_ts.shape[0]
            _arr = np.array(row == lbl for row, lbl in zip(self.prediction_ts, y_lbl))
            self.prediction_ts = _arr
        return self.prediction_ts

    def plot_decoder_accuracy(self, labels, plt_kwargs=None, **kwargs):
        fig, ax = kwargs.get('plot', (None, None))
        start_loc = kwargs.get('start_loc', 0)
        n_features = kwargs.get('n_features', None)

        if len(labels) <= 2:
            self.accuracy_plot = plot_decoder_accuracy(self.fold_accuracy, labels, fig=fig, ax=ax,
                                                       plt_kwargs=plt_kwargs,
                                                       start_loc=start_loc, n_features=n_features)
        else:
            unique_lbls = np.arange(len(labels))
            y_tests_folds = [np.hstack(ee) for ee in [e[0] for e in self.predictions]]
            y_preds_folds = [np.hstack(ee) for ee in [e[1] for e in self.predictions]]

            lbl_accuracy_list = [[(fold_preds[fold_tests == lbl] == lbl).mean() for lbl in unique_lbls]
                                 for fold_tests, fold_preds in zip(y_tests_folds, y_preds_folds)]
            lbl_accuracy_list = np.array(lbl_accuracy_list).T
            # for lbl_i, (lbl, lbl_acc) in enumerate(zip(labels,lbl_accuracy_list)):
            self.accuracy_plot = plot_decoder_accuracy(lbl_accuracy_list, labels, fig=fig, ax=ax, plt_kwargs=plt_kwargs,
                                                       start_loc=start_loc, n_features=len(labels))
            self.accuracy_plot[1].legend(ncols=len(labels))

        # self.accuracy_plot[0].show()

    def plot_confusion_matrix(self, labels, **kwargs):
        y_tests = np.hstack([np.hstack(ee) for ee in [e[0] for e in self.predictions]])
        y_preds = np.hstack([np.hstack(ee) for ee in [e[1] for e in self.predictions]])
        self.cm = confusion_matrix(y_tests, y_preds, normalize='true')
        cm_plot_ = ConfusionMatrixDisplay(self.cm, display_labels=labels, )
        cm_plot__ = cm_plot_.plot(**kwargs)
        self.cm_plot = cm_plot__.figure_, cm_plot__.ax_
        self.cm_plot[1].invert_yaxis()


def make_design_matrix(stim, d=25):
    """Create time-lag design matrix from stimulus intensity vector.
    Args:
      stim (1D array): Stimulus intensity at each time point.
      d (number): Number of time lags to use.
    Returns
      X (2D array): GLM design matrix with shape T, d
    """
    # Create version of stimulus vector with zeros before onset
    padded_stim = np.concatenate([np.zeros(d - 1), stim])
    # Construct a matrix where each row has the d frames of
    # the stimulus preceding and including timepoint t
    T = len(stim)  # Total number of timepoints (hint: number of stimulus frames)
    X = np.zeros((T, d))
    for t in range(T):
        X[t] = padded_stim[t:t + d]
    return X


def predict_spike_counts_lg(stim, spikes, d=25):
    """Compute a vector of predicted spike counts given the stimulus.

  Args:
    stim (1D array): Stimulus values at each timepoint
    spikes (1D array): Spike counts measured at each timepoint
    d (number): Number of time lags to use.

  Returns:
    yhat (1D array): Predicted spikes at each timepoint.

  """

    # Create the design matrix
    y = spikes
    constant = np.ones_like(y)
    X = np.column_stack([constant, make_design_matrix(stim,d)])

    # Get the MLE weights for the LG model
    theta = np.linalg.inv(X.T @ X) @ X.T @ y

    # Compute predicted spike counts
    yhat = X @ theta

    return yhat


def run_decoder(predictors, features, shuffle=False,model='svc', pre_split=None,
                extra_datasets=None,seed=1, **kwargs) -> [float,svm.SVC, [float,]]:
    # print(f'pre_split = {pre_split}')

    if predictors is None:
        preds_fn: str = kwargs['preds_fn']
        feats_fn: str = kwargs['feats_fn']
        shape = kwargs['shape']
        predictors = np.memmap(preds_fn, dtype='float64', mode='r', shape=shape)
        f_shape = predictors[:,0].shape
        features = np.memmap(feats_fn, dtype='float64', mode='r', shape=f_shape)

    if model == 'svc':
        model_nb = svm.SVC(C=1,class_weight='balanced')
    elif model == 'ridge':
        model_nb = Ridge(alpha=0.5)
    elif model == 'lasso':
        model_nb = Lasso(alpha=0.5)
    elif model == 'elasticnet':
        model_nb = ElasticNet(alpha=1)
    elif model == 'linear':
        model_nb = LinearRegression()
    elif model == 'logistic':
        model_nb = LogisticRegression(class_weight='balanced',max_iter=10000,solver=kwargs.get('solver','newton-cg'),#solver='newton-cg',
                                      penalty=kwargs.get('penalty','l2'),n_jobs=kwargs.get('n_jobs',1))
    else:
        raise Warning('Invalid model')
    rand_idxs = np.random.choice(predictors.shape[0],predictors.shape[0],replace=False)
    if np.isnan(predictors).any():
        print('Nan in predictors, skipping')
        return np.nan, np.nan, np.nan, np.nan

    cv_folds = kwargs.get('cv_folds',None)
    loo_cv = kwargs.get('loo_cv',False)
    if not any([loo_cv, pre_split, cv_folds]):
        predictors, features = predictors[rand_idxs], features[rand_idxs]
    if shuffle:
        features = np.random.permutation(features.copy())
    # print('yay')
    if cv_folds:
        kf = KFold(n_splits=cv_folds,shuffle=True)
        x_train, x_test, y_train, y_test = [], [], [], []
        for train_idx, test_idx in kf.split(predictors):
            x_train.append(predictors[train_idx]), y_train.append(features[train_idx])
            x_test.append(predictors[test_idx]), y_test.append(features[test_idx])
    elif pre_split:
        x_train, x_test = predictors[:pre_split], predictors[pre_split:]
        y_train, y_test = features[:pre_split], features[pre_split:]
    elif loo_cv:
        loo = LeaveOneOut()
        loo.get_n_splits(predictors)
        x_train, x_test, y_train, y_test = [], [], [], []
        for train_idx, test_idx in loo.split(predictors):
            x_train.append(predictors[train_idx]), y_train.append(features[train_idx])
            x_test.append(predictors[test_idx]), y_test.append(features[test_idx])
    else:
        # test_size = np.random.uniform(0.1,0.4,1)[0]
        test_size = kwargs.get('test_size', 0.2)
        strat_split = StratifiedShuffleSplit(n_splits=kwargs.get('_n_runs',1),test_size=test_size,)
        x_train, x_test, y_train, y_test = [], [], [], []
        for train_idx, test_idx in strat_split.split(predictors,features):
            x_train.append(predictors[train_idx]), y_train.append(features[train_idx])
            x_test.append(predictors[test_idx]), y_test.append(features[test_idx])

    if not isinstance(x_train,list):
        x_train = [x_train]
        y_train = [y_train]
        x_test = [x_test]
        y_test = [y_test]

    perf_list = []
    model_list = []
    y_test_list = []
    pred_list = []
    pred_train_list = []
    pred_proba_list = []
    for fold_i,_ in enumerate(x_train):
        # if shuffle:  # just scrambles
        #     y_train_fold = np.random.choice(y_train[fold_i],y_train[fold_i].shape[0])
        # else:
        #     y_train_fold = y_train[fold_i]
        y_train_fold = y_train[fold_i]
        model_nb.fit(x_train[fold_i], y_train_fold)
        predict_train = model_nb.predict(x_train[fold_i])
        if len(y_test[fold_i].shape) == 0:
            perf = [np.nan]
        else:
            predict = model_nb.predict(x_test[fold_i])
            perf = (np.equal(y_test[fold_i], predict)).mean()

        # predict_prob = model_nb.predict_proba(x_test[fold_i])
        perf_list.append(perf)
        model_list.append(model_nb)
        pred_train_list.append(predict_train)

        y_test_list.append(y_test[fold_i])
        pred_list.append(predict)
        # pred_proba_list.append(predict_prob)

    return np.mean(perf_list), model_list, perf_list, [y_test_list,pred_list], pred_train_list   # , model_nb ,pred_proba_list


def iter_decoder(predictors:np.ndarray,features: np.ndarray, shuffle=False) -> np.ndarray:
    decoder_ts = [run_decoder(col_predictor, col_feature,shuffle=shuffle)
                  for col_predictor, col_feature in zip(predictors.T, features.T)]
    return np.array(decoder_ts)


def init_pool_processes():
    np.random.seed()


def balance_predictors(list_predictors, list_features) -> [[np.ndarray], [np.ndarray]]:
    min_pred_len = min([e.shape[0] for e in list_predictors])
    assert min_pred_len >= 4

    idx_subsets = [(np.random.seed(ei),np.random.choice(e.shape[0], min_pred_len, replace=False,))[1]
                   for ei,e in enumerate(list_predictors)]
    # assert len(np.unique(idx_subset)) == min_pred_len
    predictors = [e[idxs] for e, idxs in zip(list_predictors, idx_subsets)]
    features = [e[idxs] for e, idxs in zip(list_features, idx_subsets)]

    return np.vstack(predictors), np.hstack(features)

def predict_1d(models, t_ser, y_lbl=0):
    # models = models
    # with multiprocessing.Pool() as pool:
    #     results = tqdm(pool.imap())
    prediction_ts = np.array([m.predict(t_ser.T) for m in models])
    if isinstance(y_lbl, list):
        assert len(y_lbl) == prediction_ts.shape[0]
        _arr = np.array(row == lbl for row, lbl in zip(prediction_ts, y_lbl))
        prediction_ts = _arr
    return prediction_ts


def get_decoder_accuracy(sess_obj, decoder_name):
    return sess_obj.decoders[decoder_name].accuracy


def get_decoder_accuracy_from_pkl(pkl_path):
    try:
        sess_obj = load_sess_pkl(pkl_path)
    except:
        print(f'{pkl_path} error')
        return None
    print(f'extracting decoder accuracies from {pkl_path}')
    res = [get_decoder_accuracy(sess_obj, decoder_name) for decoder_name in sess_obj.decoders.keys()]
    keys = list(sess_obj.decoders.keys())
    return dict(zip(keys, res))


def get_property_from_decoder_pkl(pkl_path: str, decoder_name: str, property_name: str):
    try:
        sess_obj = load_sess_pkl(pkl_path)
    except:
        print(f'{pkl_path} error')
        return None
    return getattr(sess_obj.decoders[decoder_name], property_name)
