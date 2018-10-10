import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc

from app import app

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
import os
from sklearn import *

from scipy.stats.stats import pearsonr, spearmanr
from shapely.geometry import Polygon
from collections import Counter

from app import app

month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
categories = np.array(['bin','bowl','bucket','cup','jar','pottedplant','tire','vase']).reshape(-1,1)

# Load survey data
df_survey = pd.read_csv('../data/breeding-sites/larval-survey.csv')
df_survey = df_survey.replace(0, np.nan)
df_survey = df_survey.dropna(axis=0, how='any')
df_survey = df_survey.reset_index(drop=True)
df_survey = df_survey.loc[df_survey['province'] == 'นครศรีธรรมราช']
df_survey = df_survey.drop('province', axis=1)
df_survey['date'] = pd.to_datetime(df_survey['date'], format='%Y-%m')
df_survey = df_survey.set_index('date')
df_survey = df_survey.sort_index()
df_survey = df_survey['2015':'2018']

# Exclude outliers from survey index
df_filtered = []
subdist_list = df_survey['subdist'].unique()
for subdist in subdist_list:
    tmp = df_survey.loc[df_survey['subdist'] == subdist].copy()
    if len(tmp) == 1 and tmp['bi'].mean() < 100:
        df_filtered.append(tmp.copy())
    df_filtered.append(tmp[np.abs(tmp['bi']-tmp['bi'].mean()) <= (2*tmp['bi'].std())].copy())

df_filtered = pd.concat(df_filtered, axis=0)

# Load detected breeding sites from street view
df_detect = pd.read_csv('../data/breeding-sites/sum-detection.csv')
df_detect['date'] = pd.to_datetime(df_detect['date'], format='%Y-%m')
df_detect = df_detect.set_index('date')
df_detect = df_detect.sort_index()

# Calculate subdistrict area
URL = 'https://raw.githubusercontent.com/pcrete/Mosquito_Breeding_Sites_Detector/master/geojson/province/%E0%B8%99%E0%B8%84%E0%B8%A3%E0%B8%A8%E0%B8%A3%E0%B8%B5%E0%B8%98%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%A3%E0%B8%B2%E0%B8%8A.geojson'
import urllib.request, json
with urllib.request.urlopen(URL) as url:
    data_polygon = json.loads(url.read().decode())

df_area = []
for feature in data_polygon['features']:
    poly = Polygon(feature['geometry']['coordinates'][0])

    prop = feature['properties']
    district = prop['AP_TN']
    subdist = prop['TB_TN']

    df_area.append([district, subdist, round(poly.area*111111,2)])

df_area = pd.DataFrame.from_records(df_area)
df_area.columns = ['district', 'subdist', 'area']

# ========================================================================================
# ====================================== Correlation 1 ===================================
# ========================================================================================

x_train, y_train = [], []

xs, ys = [], []

column = 'total'

mean_det, std_det = df_detect[column].mean(), df_detect[column].std()
mean_det, std_det

df_filtered['bi'].mean(),df_filtered['bi'].std()

subdist_list = df_filtered['subdist'].unique()
for subdist in subdist_list:
    detect = round(df_detect.loc[df_detect['subdist'] == subdist][column].mean(),2)
    area = round(df_area.loc[df_area['subdist'] == subdist]['area'].mean(),2)
    survey = round(df_filtered.loc[(df_filtered['subdist'] == subdist)
#                                    & (df_filtered.index.month.isin([2,3,4,5]))
                                  ]['bi'].mean(), 2)

#     if detect > mean_det+1*std_det: continue
#     if detect < mean_det-1*std_det: continue
    if np.isnan(detect) or np.isnan(survey): continue

    xs.append(survey)
    ys.append(detect/area)

    x = df_detect.loc[df_detect['subdist'] == subdist].copy()
    x = x[['bin','bowl','bucket','cup','jar','pottedplant','tire','vase']].copy()

    x_train.append(np.squeeze(x.values)/area)
    y_train.append(survey)

X = np.array(x_train)
y = np.array(y_train)


trace = go.Scatter(
    x = xs, y = ys, mode = 'markers', name='Scatter',
    marker = dict(size = 16, opacity = 0.5)
)

xs = np.array(xs)
ys = np.array(ys)

regr = linear_model.LinearRegression()
regr.fit(xs.reshape(-1, 1), ys.reshape(-1, 1))

ys_pred = regr.predict(xs.reshape(-1, 1))
trace_2 = go.Scatter(
    x = xs, y = np.squeeze(ys_pred),
    mode = 'lines',
    name = 'Regression',
    line = dict(width = 4)
)

layout = dict(
    title = 'All Breeding Sites<br>Pearson: '+ \
    str(round(pearsonr(xs, ys)[0],4))+', Spearman:'+ \
    str(round(spearmanr(xs, ys)[0],4))+'<br>('+ \
    str(len(xs))+' data points or subdistricts)',
    titlefont=dict(size=20),
    xaxis = dict(title = 'Breteau Index'),
    yaxis = dict(title = '(Total # of detected containers)/Area'),
    height=800
)
fig_all_breeding_sites_1 = go.Figure(data=[trace,trace_2], layout=layout)

# ========================================================================================
# ========================================================================================

# ========================================================================================
# ====================================== Correlation 2 ===================================
# ========================================================================================

x_train, y_train = [], []

xs, ys = [], []

column = 'total'

mean_det, std_det = df_detect[column].mean(), df_detect[column].std()
mean_det, std_det

df_filtered['bi'].mean(),df_filtered['bi'].std()

subdist_list = df_filtered['subdist'].unique()
for subdist in subdist_list:
    detect = round(df_detect.loc[df_detect['subdist'] == subdist][column].mean(),2)
    area = round(df_area.loc[df_area['subdist'] == subdist]['area'].mean(),2)
    survey = round(df_filtered.loc[(df_filtered['subdist'] == subdist)
#                                    & (df_filtered.index.month.isin([2,3,4,5]))
                                  ]['bi'].mean(), 2)

    if detect > mean_det+1*std_det: continue
    if detect < mean_det-1*std_det: continue
    if np.isnan(detect) or np.isnan(survey): continue

    xs.append(survey)
    ys.append(detect/area)

    x = df_detect.loc[df_detect['subdist'] == subdist].copy()
    x = x[['bin','bowl','bucket','cup','jar','pottedplant','tire','vase']].copy()

    x_train.append(np.squeeze(x.values)/area)
    y_train.append(survey)

X = np.array(x_train)
y = np.array(y_train)


trace = go.Scatter(
    x = xs, y = ys, mode = 'markers', name='Scatter',
    marker = dict(size = 16, opacity = 0.5)
)

xs = np.array(xs)
ys = np.array(ys)

regr = linear_model.LinearRegression()
regr.fit(xs.reshape(-1, 1), ys.reshape(-1, 1))

ys_pred = regr.predict(xs.reshape(-1, 1))
trace_2 = go.Scatter(
    x = xs, y = np.squeeze(ys_pred),
    mode = 'lines',
    name = 'Regression',
    line = dict(width = 4)
)

layout = dict(
    title = 'All Breeding Sites (excluded outliers)<br>Pearson: '+ \
    str(round(pearsonr(xs, ys)[0],4))+', Spearman:'+ \
    str(round(spearmanr(xs, ys)[0],4))+ \
    ', R-squared:' + str(round(metrics.r2_score(xs, ys),4)) + \
    '<br>('+ str(len(xs))+' data points or subdistricts)',
    titlefont=dict(size=20),
    xaxis = dict(title = 'Breteau Index'),
    yaxis = dict(title = '(Total # of detected containers)/Area'),
    height=800
)
fig_all_breeding_sites_2 = go.Figure(data=[trace,trace_2], layout=layout)

# ========================================================================================
# ========================================================================================

loo = model_selection.LeaveOneOut()

def plot_regression(regr, name):
    regr.fit(x_train, y_train)

    try: coef = regr.coef_.reshape(-1,1)
    except: pass

    try: coef = regr.feature_importances_.reshape(-1,1)
    except: pass

    df_coef = pd.DataFrame.from_records(np.concatenate((categories, coef), axis=1))
    df_coef.columns = ['class', 'coef']
    df_coef['coef'] = df_coef['coef'].astype(float).round(4)

    # Leave One Out Validation
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

#     print('R-squared:', metrics.r2_score(y_true, y_pred))
#     print('Person:', pearsonr(y_true, y_pred))
#     print(spearmanr(y_true, y_pred),'\n')

    # Correlation Visualization
    trace_1 = go.Scatter(
        x = y_true, y = y_pred, mode = 'markers', name='Scatter',
        marker = dict(size = 12, opacity = 0.5)
    )
    xs = np.array(y_true)
    ys = np.array(y_pred)

    regr = linear_model.LinearRegression()
    regr.fit(xs.reshape(-1, 1), ys.reshape(-1, 1))

    ys_pred = regr.predict(xs.reshape(-1, 1))
    trace_2 = go.Scatter(
        x = xs, y = np.squeeze(ys_pred), name='Regression',
        mode = 'lines',
        line = dict(
            width = 4,
        )
    )

    coef = np.squeeze(coef)
    coef = 100.0 * (coef / coef.max())
    sorted_idx = np.argsort(coef)[::-1]
    pos = np.arange(sorted_idx.shape[0]) + .5

    trace_3 = go.Bar(
        x = np.squeeze(categories[sorted_idx]),
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


    fig = tools.make_subplots(rows=2, cols=1,
                              subplot_titles=('',''),
                             vertical_spacing = 0.1)

    fig.append_trace(trace_1, 1, 1)
    fig.append_trace(trace_2, 1, 1)
    fig.append_trace(trace_3, 2, 1)

    fig['layout'].update(
        height=700,
        title = name+'<br>Pearson: '+ \
        str(round(pearsonr(y_true, y_pred)[0],4))+', Spearman:'+ \
        str(round(spearmanr(y_true, y_pred)[0],4)) + \
        ', R-squared:' + str(round(metrics.r2_score(y_true, y_pred),4)),
        titlefont=dict(size=20)
    )
    fig['layout']['xaxis1'].update(title='Breteau index', titlefont=dict(size=16))
    fig['layout']['xaxis2'].update(title='Breeding site', tickfont=dict(size=14), titlefont=dict(size=16))

    fig['layout']['yaxis1'].update(title='Predicted', titlefont=dict(size=16))
    fig['layout']['yaxis2'].update(title='Relative Importance', titlefont=dict(size=16))

    return dcc.Graph(id=name, figure=fig)

# ========================================================================================
# ========================================================================================


layout = html.Div([

    html.P(),

    html.Div([
        dcc.Graph(id='fig_all_breeding_sites_1', figure=fig_all_breeding_sites_1)
    ], style={'width': '49%', 'display':'inline-block', 'padding-left':'1%'}),

    html.Div([
        dcc.Dropdown(
            id='month-dropdown',
            options=[
                {'label': 'Jan', 'value': 1},
                {'label': 'Feb', 'value': 2},
                {'label': 'Mar', 'value': 3},
                {'label': 'Apr', 'value': 4},
                {'label': 'May', 'value': 5},
                {'label': 'Jun', 'value': 6},
                {'label': 'Jul', 'value': 7},
                {'label': 'Aug', 'value': 8},
                {'label': 'Sep', 'value': 9},
                {'label': 'Oct', 'value': 10},
                {'label': 'Nov', 'value': 11},
                {'label': 'Dec', 'value': 12},
            ],
            value=[1,2,3,4,5,6,7,8,9,10,11,12],
            multi=True
        ),
        dcc.Graph(id='fig_all_breeding_sites_2')
    ], style={'width': '49%', 'display':'inline-block', 'padding-left':'1%'}),

    html.P(),

    html.Div([
        plot_regression(linear_model.LinearRegression(),'Linear Regression')
    ], style={'width': '32%', 'display':'inline-block', 'padding-left':'1%'}),

    html.Div([
        plot_regression(svm.SVR(kernel='linear'),'Support Vector Regression')
    ], style={'width': '32%', 'display':'inline-block', 'padding-left':'1%'}),

    html.Div([
        plot_regression(linear_model.BayesianRidge(),'Bayesian Ridge')
    ], style={'width': '32%', 'display':'inline-block', 'padding-left':'1%'}),

    html.P(),

    html.Div([
        plot_regression(ensemble.RandomForestRegressor(),'Random Forest Regressor')
    ], style={'width': '32%', 'display':'inline-block', 'padding-left':'1%'}),

    html.Div([
        plot_regression(tree.DecisionTreeRegressor(), 'Decision Tree Regressor')
    ], style={'width': '32%', 'display':'inline-block', 'padding-left':'1%'}),

    html.Div([
        plot_regression(ensemble.GradientBoostingRegressor(),'Gradient Boosting Regressor')
    ], style={'width': '32%', 'display':'inline-block', 'padding-left':'1%'}),



],
style={
    'background-color':'#F5F5F5',
    'width':'100%'
})


@app.callback(
    dash.dependencies.Output('fig_all_breeding_sites_2', 'figure'),
    [dash.dependencies.Input('month-dropdown', 'value')])
def generate_boxplot(selected_months):

    x_train, y_train = [], []

    xs, ys = [], []

    column = 'total'

    mean_det, std_det = df_detect[column].mean(), df_detect[column].std()
    mean_det, std_det

    df_filtered['bi'].mean(),df_filtered['bi'].std()

    subdist_list = df_filtered['subdist'].unique()
    for subdist in subdist_list:
        detect = round(df_detect.loc[df_detect['subdist'] == subdist][column].mean(),2)
        area = round(df_area.loc[df_area['subdist'] == subdist]['area'].mean(),2)
        survey = round(df_filtered.loc[(df_filtered['subdist'] == subdist)
                                       & (df_filtered.index.month.isin(selected_months))
                                      ]['bi'].mean(), 2)

        if detect > mean_det+1*std_det: continue
        if detect < mean_det-1*std_det: continue
        if np.isnan(detect) or np.isnan(survey): continue

        xs.append(survey)
        ys.append(detect/area)

        x = df_detect.loc[df_detect['subdist'] == subdist].copy()
        x = x[['bin','bowl','bucket','cup','jar','pottedplant','tire','vase']].copy()

        x_train.append(np.squeeze(x.values)/area)
        y_train.append(survey)

    x_train = np.array(x_train)
    y_train = np.array(y_train)


    trace = go.Scatter(
        x = xs, y = ys, mode = 'markers', name='Scatter',
        marker = dict(size = 16, opacity = 0.5)
    )

    xs = np.array(xs)
    ys = np.array(ys)

    regr = linear_model.LinearRegression()
    regr.fit(xs.reshape(-1, 1), ys.reshape(-1, 1))

    ys_pred = regr.predict(xs.reshape(-1, 1))
    trace_2 = go.Scatter(
        x = xs, y = np.squeeze(ys_pred),
        mode = 'lines',
        name = 'Regression',
        line = dict(width = 4)
    )

    layout = dict(
        title = 'All Breeding Sites (excluded outliers)<br>Pearson: '+ \
        str(round(pearsonr(xs, ys)[0],4))+', Spearman:'+ \
        str(round(spearmanr(xs, ys)[0],4))+ \
        ', R-squared:' + str(round(metrics.r2_score(xs, ys),4)) + \
        '<br>('+ str(len(xs))+' data points or subdistricts)',
        titlefont=dict(size=20),
        xaxis = dict(title = 'Breteau Index'),
        yaxis = dict(title = '(Total # of detected containers)/Area'),
        height=800
    )
    fig_all_breeding_sites_2 = go.Figure(data=[trace,trace_2], layout=layout)

    return fig_all_breeding_sites_2
