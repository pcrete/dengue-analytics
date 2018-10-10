from scipy.stats.stats import pearsonr, spearmanr
from plotly import tools
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from modules import query, residual, evaluation
from sklearn import *
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score

loo = model_selection.LeaveOneOut()

def correlation_total_bs(df_survey, df_detect, df_area, detect_stat, column='total'):

    titles, traces = [], []
    for dengue_season, province in list(itertools.product([False, True], [['80','81'],['80'],['81']])):
        X, y = [], []
        names = []
        for addrcode in df_survey['addrcode'].unique():

            province_id = str(addrcode)[:2]
            if province_id not in province: continue

            mean_det, std_det = detect_stat[province_id]

            detect = query.get_detect(df_detect, addrcode)
            survey, title = query.get_survey(df_survey, dengue_season, addrcode)
            area = query.get_area(df_area, addrcode)

            if np.isnan(detect) or np.isnan(survey): continue
            if detect > mean_det+1*std_det or detect < mean_det-1*std_det: continue

            X.append(survey)
            y.append(detect/area)
            names.append(addrcode)

        province = ', '.join(province)

        pearson_val = pearsonr(X, y)
        spearman_val = spearmanr(X, y)

        trace_1 = go.Scatter(
            x=X,
            y=y,
            mode='markers',
            name=province,
            text=names,
            marker=dict(size=14, opacity=0.5)
        )

        X = np.array(X)
        y = np.array(y)

        regr = linear_model.LinearRegression()
        regr.fit(X.reshape(-1, 1), y.reshape(-1, 1))
        y_pred = regr.predict(X.reshape(-1, 1))

        trace_2 = go.Scatter(
            x = X,
            y = np.squeeze(y_pred),
            mode = 'lines',
            line = dict(width = 4),
            name=province
        )

        titles.append(
            title+' ('+str(len(X))+' data points)'+', province: '+province+ \
            '<br>Pearson: '+str(round(pearson_val[0],4))+ ', p-value: '+str(round(pearson_val[1],4))+ \
            '<br>Spearman: '+str(round(spearman_val[0],4))+', p-value: '+str(round(spearman_val[1],4))
        )
        traces.append([trace_1, trace_2])

    fig = tools.make_subplots(rows=2, cols=3, subplot_titles=tuple(titles), horizontal_spacing = 0.05, vertical_spacing=0.15)

    k = 0
    for i in range(2):
        for j in range(3):
            fig.append_trace(traces[k][0], i+1, j+1)
            fig.append_trace(traces[k][1], i+1, j+1)

            fig['layout']['xaxis'+str(k+1)].update(title='Breteau index')
            fig['layout']['yaxis'+str(k+1)].update(title='Normalized breeding site counts by area')
            k+=1

    fig['layout'].update(
#         title='Multiple Subplots, where province code 80 refers to Krabi, and 81 refers to Nakhon Si Thammarat<br>',
#         width=1600,
        height=1200,
#         font=dict(size=14),
        hovermode='closest'
    )

    iplot(fig)


def get_best_param(X, y, grids):
    m = 0
    len_X, len_param = X.shape[0], len(grids)
    ratio = int(np.ceil(len_X/len_param))
    max_corr = -1
    best_param = {}
    for param in tqdm(grids):
        indices = [(n+m)%len_X for n in range(ratio)]

        y_true, y_pred = evaluation.eval_leave_one_out(
            np.delete(deepcopy(X), indices, axis=0),
            np.delete(deepcopy(y), indices, axis=0),
            ensemble.GradientBoostingRegressor(**param)
        )
        m += 1
        corr = pearsonr(y_true, y_pred)[0]
        if corr > max_corr:
            max_corr, best_param = deepcopy(corr), deepcopy(param)

    print(round(max_corr, 4), best_param)
    print('len_X:',len_X, ', len_param:', len_param, ', ratio:', ratio)
    return best_param

def get_best_param_mse(X, y, grids):
    m = 0
    len_X, len_param = X.shape[0], len(grids)
    ratio = int(np.ceil(len_X/len_param))
    min_mse = 999999
    best_param = {}
    for param in tqdm(grids):
        indices = [(n+m)%len_X for n in range(ratio)]

        y_true, y_pred = evaluation.eval_leave_one_out(
            np.delete(deepcopy(X), indices, axis=0),
            np.delete(deepcopy(y), indices, axis=0),
            ensemble.GradientBoostingRegressor(**param)
        )
        m += 1
        mse = mean_squared_error(y_true, y_pred)
        if mse < min_mse:
            min_mse, best_param = deepcopy(mse), deepcopy(param)

    print(round(min_mse, 4), best_param)
    print('len_X:',len_X, ', len_param:', len_param, ', ratio:', ratio)
    return best_param

def get_best_param_r2(X, y, grids):
    m = 0
    len_X, len_param = X.shape[0], len(grids)
    ratio = int(np.ceil(len_X/len_param))
    max_r2 = -1
    best_param = {}
    for param in tqdm(grids):
        indices = [(n+m)%len_X for n in range(ratio)]

        y_true, y_pred = evaluation.eval_leave_one_out(
            np.delete(deepcopy(X), indices, axis=0),
            np.delete(deepcopy(y), indices, axis=0),
            ensemble.GradientBoostingRegressor(**param)
        )
        m += 1
        r2 = r2_score(y_true, y_pred)
        if r2 > max_r2:
            max_r2, best_param = deepcopy(r2), deepcopy(param)

    print(round(max_r2, 4), best_param)
    print('len_X:',len_X, ', len_param:', len_param, ', ratio:', ratio)
    return best_param

def get_prediction_features(df_survey, df_detect, df_population, df_area, df_gsv_coverage, brd_sites, detect_stat, dengue_season=False, is_image_area=False, province=''):
    if is_image_area:
        print('Using image area')
    X, y = [], []
    names = []
    for addrcode in df_survey['addrcode'].unique():

        province_id = str(addrcode)[:2]
        if province_id not in province: continue

        mean_det, std_det = detect_stat[province_id]

        detect = query.get_detect(df_detect, addrcode)
        survey, title = query.get_survey(df_survey, dengue_season, addrcode)
        population = query.get_population(df_population, addrcode)
        area = query.get_area(df_area, addrcode)
        image_area = query.get_gsv_coverage(df_gsv_coverage, addrcode)

        if np.isnan(detect) or np.isnan(survey): continue
        if detect > mean_det+1*std_det or detect < mean_det-1*std_det: continue

        # ======================== Prediction Features =====================
        month = query.get_gsv_month(df_detect, addrcode)

        detect = query.get_detect(df_detect, addrcode, brd_sites)

        # Normalize by area
        if is_image_area:
            detect = (100*np.squeeze(detect.values))/(image_area*area)
        else:
            detect = np.squeeze(detect.values)/area

        # Combine Features
        features = list(detect) + [month]

        X.append(features)
        y.append(survey)

        names.append(addrcode)

    return np.array(X), np.array(y), np.array(names)


def perform_correlation(regr, title, names, province, X, y):

    y_pred, y_true = [], []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y[train_index], y[test_index]

        _=regr.fit(X_train, Y_train)

        y_pred.append(np.squeeze(regr.predict(X_test)))
        y_true.append(np.squeeze(Y_test))

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    r2 = r2_score(y_true, y_pred)

    pearson_val = round(pearsonr(y_true, y_pred)[0],4)
    spearman_val = round(spearmanr(y_true, y_pred)[0],4)

    pearson_p_val = round(pearsonr(y_true, y_pred)[1],4)
    spearman_p_val = round(spearmanr(y_true, y_pred)[1],4)

    trace_1 = go.Scatter(
        x = y_true,
        y = y_pred,
        mode = 'markers',
        name='Scatter',
        text=names,
        marker = dict(size = 12, opacity = 0.7)
    )

    regr = linear_model.LinearRegression()
    regr.fit(y_true.reshape(-1, 1), y_pred.reshape(-1, 1))
    y_trend = np.squeeze(regr.predict(y_true.reshape(-1, 1)))

    trace_2 = go.Scatter(
        x = y_true,
        y = y_true,
        name='Trend',
        mode = 'lines', line = dict(width = 4)
    )

    province = ', '.join(province)

    title += ' ('+str(X.shape[0])+' data points)'+', province: '+province+ \
            '<br>R-squared: '+str(r2) + \
            '<br>Pearson: '+str(pearson_val)+ ', p-value: '+str(pearson_p_val)+ \
            '<br>Spearman: '+str(spearman_val)+', p-value: '+str(spearman_p_val)

    layout = go.Layout(
        title=title,
        width=550,
        height=450,
        yaxis= dict(title='Predicted'),
        xaxis= dict(title='Breteau index'),
        hovermode='closest'
    )
    fig = go.Figure(data=[trace_1, trace_2], layout=layout)
    iplot(fig)

    # df_residual = residual.get_residuals(y_trend, y_pred, names)
    df_residual = residual.get_residuals(y_true, y_pred, names)

    return df_residual
