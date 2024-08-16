import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



# def run_regression(xs,ys, split_kwargs=None,fit_kwargs=None):
#     # Train on training split
#     x_train, y_train, x_test, y_test = train_test_split(xs, ys, **split_kwargs if split_kwargs is None else {})
#     regressor = sm.OLS(y_train, x_train)
#     regressor.fit(**fit_kwargs if fit_kwargs is None else {})
#     y_pred = regressor.predict(x_test)
#     return regressor, y_pred


def run_regression(xs,ys, split_kwargs=None,fit_kwargs=None):
    model = LinearRegression()
    # Train on training split
    # x_train, y_train, x_test, y_test = train_test_split(xs, ys, **split_kwargs if split_kwargs is None else {})
    model.fit(xs, ys)
    # y_pred = model.predict(x_test)
    return model


def run_glm(xs,ys, split_kwargs=None,fit_kwargs=None):
    # Train on training split
    # x_train, y_train, x_test, y_test = train_test_split(xs, ys, **split_kwargs if split_kwargs is None else {})
    # regressor = smf.glm(formula=f'Default ~ {" + ".join(ys.keys())}', data=xs, family=sm.families.Poisson())
    # ys = ys.rename(columns={k: f'Q{i+1}' for i,k in enumerate(ys.keys())})
    # print(f'{ys.columns = }')
    # data = pd.concat([xs, ys], axis=1)
    # regressor = sm.glm(formula=f'0 ~ {" + ".join(ys.keys())}', data=pd.concat([xs, ys], axis=1), family=sm.families.Poisson())
    ys = sm.add_constant(ys,prepend=False)
    # print(f'{ys.columns = }')
    regressor = sm.GLM(xs,ys)
    result = regressor.fit()
    # y_pred = regressor.predict(x_test)
    return regressor,result


