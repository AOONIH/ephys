import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, LogisticRegression
from sklearn import svm
import pandas as pd


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
        x_train, x_test, y_train, y_test = train_test_split(predictors, features, test_size=test_size,
                                                            )

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
        if shuffle:  # just scrambles
            y_train_fold = np.random.choice(y_train[fold_i],y_train[fold_i].shape[0])
        else:
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


