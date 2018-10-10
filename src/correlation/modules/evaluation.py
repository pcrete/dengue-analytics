import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr, spearmanr
from sklearn import model_selection

loo = model_selection.LeaveOneOut()

def eval_leave_one_out(X, y, regr):
    y_pred, y_true  = [], []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y[train_index], y[test_index]

        _=regr.fit(X_train, Y_train)
        pred = regr.predict(X_test)
        y_true.append(np.squeeze(Y_test))
        y_pred.append(np.squeeze(pred))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return y_true, y_pred

def eval_importances(X, y, regr):
    importances = []
    for train_index, test_index in loo.split(X):
        X_train, Y_train = X[train_index], y[train_index]
        regr.fit(X_train, Y_train)
        importances.append(regr.feature_importances_)
    return np.array(importances)
