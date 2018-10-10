from plotly import tools
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import plotly.graph_objs as go

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn import *
from scipy.stats.stats import pearsonr, spearmanr

def plot_correlation(regr, name, X, y, loo):
    
    regr.fit(X, y)   
    
    y_pred, y_true = [], []
    for train_index, test_index in loo.split(X):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y[train_index], y[test_index]

        _=regr.fit(X_train, Y_train)

        pred = regr.predict(X_test)
        y_pred.append(np.squeeze(pred))
        y_true.append(np.squeeze(Y_test))

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    r2 = round(metrics.r2_score(y_true, y_pred),4)
    pearson = round(pearsonr(y_true, y_pred)[0],4)
    spearman = round(spearmanr(y_true, y_pred)[0],4)
    
    print('R-squared:', metrics.r2_score(y_true, y_pred))
    print('Person:', pearsonr(y_true, y_pred))
    print(spearmanr(y_true, y_pred),'\n')

    trace_1 = go.Scatter(
        x = y_true, y = y_pred, mode = 'markers', name='Scatter',
        marker = dict(size = 12, opacity = 0.5)
    )
    
    xs, ys = np.array(y_true), np.array(y_pred)

    regr = linear_model.LinearRegression()
    regr.fit(xs.reshape(-1, 1), ys.reshape(-1, 1))

    ys_pred = regr.predict(xs.reshape(-1, 1))
    trace_2 = go.Scatter(
        x = xs, y = np.squeeze(ys_pred), name='Regression',
        mode = 'lines', line = dict(width = 4)
    )

    name += 'R-squared: ' + str(r2) + \
    ', Pearson: ' + str(pearson) + \
    ', Spearman: ' + str(spearman)
  
    layout = go.Layout(
        title=name,
        width=650,
        yaxis= dict(title='Predicted'),
        xaxis= dict(title='Breteau index'),
        font=dict(size=16)
    )
    fig = go.Figure(data=[trace_1, trace_2], layout=layout)
    iplot(fig)
    
def plot_importance(regr, name, X, y, loo, features_name):
    regr.fit(X, y)
    
    try: coef = regr.coef_.reshape(-1,1)
    except: pass
    try: coef = regr.feature_importances_.reshape(-1,1)
    except: pass
    
    df_coef = pd.DataFrame.from_records(np.concatenate((features_name, coef), axis=1))
    df_coef.columns = ['class', 'coef']
    df_coef['coef'] = df_coef['coef'].astype(float).round(4)
    
    coef = np.squeeze(coef)
    coef = 100.0 * (coef / coef.max())
    sorted_idx = np.argsort(coef)[::-1]
    pos = np.arange(sorted_idx.shape[0]) + .5
    
    trace_3 = go.Bar(
        x = np.squeeze(features_name[sorted_idx]),
        y = coef[sorted_idx],
        name='Variable Importance',
        marker=dict(
                    color='rgb(158,202,225)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        opacity=0.8
    )

    
    layout = go.Layout(
        title=name,
        width=650,
        yaxis= dict(title='Relative Importance'),
        xaxis= dict(title='Breeding site'),
        font=dict(size=16)
    )
    fig = go.Figure(data=[trace_3], layout=layout)
    iplot(fig)