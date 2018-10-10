import dash
import dash_core_components as dcc
import dash_html_components as html

import dash_dangerously_set_inner_html

import plotly.graph_objs as go
from plotly.graph_objs import *
from collections import Counter

import numpy as np
import pandas as pd
import os

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import urllib.request, json

from app import app

URL = 'https://raw.githubusercontent.com/pcrete/Mosquito_Breeding_Sites_Detector/master/geojson/province/%E0%B8%99%E0%B8%84%E0%B8%A3%E0%B8%A8%E0%B8%A3%E0%B8%B5%E0%B8%98%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%A3%E0%B8%B2%E0%B8%8A.geojson'
with urllib.request.urlopen(URL) as url:
    data_polygon = json.loads(url.read().decode())

mapbox_access_token = 'pk.eyJ1IjoiYWxpc2hvYmVpcmkiLCJhIjoiY2ozYnM3YTUxMDAxeDMzcGNjbmZyMmplZiJ9.ZjmQ0C2MNs1AzEBC_Syadg'

months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

df = pd.read_csv('../data/breeding-sites/larval-survey.csv')
df = df.replace(0, np.nan)
df = df.dropna(axis=0, how='any')
df = df.reset_index(drop=True)
df = df.loc[df['province'] == 'นครศรีธรรมราช']
df = df.drop('province', axis=1)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m')
df = df.set_index('date')
df = df.sort_index()

df_meta = pd.read_csv('../data/gsv_meta.csv')
df_meta['date'] = pd.to_datetime(df_meta['date'], format='%Y-%m')
df_meta = df_meta.set_index('date')
df_meta = df_meta.sort_index()
df_meta = df_meta['2014':]

mean_lat = df_meta.lat.mean()
mean_lng = df_meta.lng.mean()

def barplot(df_date_index):
    count = dict(Counter(df_date_index))
    key, val = [], []
    for k in count:
        key.append(k)
        val.append(count[k])

    trace_bar_actual = go.Bar(
        x = key,
        y = val,
        text = val,
        textposition = 'auto',
        marker=dict(
            color='#6BE59A',
            line=dict(
                color='#05B083',
                width=1.5),
        ),
        opacity=0.8
    )
    layout = go.Layout(
        title='Data points for each Year',
        height=550,
#         width=750,
        yaxis= dict(title='Frequency'),
        xaxis= dict(title='Year'),
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(0,0,0,0)',
        margin=go.Margin(
            l=100,
            r=50,
            b=140,
            t=100,
            pad=4
        ),
    )
    fig = go.Figure(data=[trace_bar_actual], layout=layout)

    return dcc.Graph(
        id='bar-year',
        figure=fig,
    )

def meta_barplot():
    count = dict(Counter(df_meta.index.year))
    key, val = [], []
    for k in count:
        key.append(k)
        val.append(count[k])

    trace_bar_actual = go.Bar(
        x = key,
        y = val,
        text = val,
        textposition = 'auto',
        marker=dict(
                    color='rgb(158,202,225)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        opacity=0.8
    )
    layout = go.Layout(
        title='Street View Images for each Year',
        height=550,
        yaxis= dict(title='Frequency'),
        xaxis= dict(title='Year')
    )
    fig = go.Figure(data=[trace_bar_actual], layout=layout)

    return dcc.Graph(
                id='meta-bar-year',
                figure=fig,
            )


# ===============================================================================================
# ============================================= App =============================================
# ===============================================================================================


layout = html.Div([
    html.Div([
        html.Div([
            barplot(df.index.year)
         ], style={'width': '28%', 'display':'inline-block', 'padding':'1%'}),
         html.Div([
            dcc.Graph(id='bar-month-with-slider-2'),
        ], style={'width': '68%', 'display':'inline-block', 'padding':'1%'}),
    ], style={'width':'100%'}),


    html.Div([
        html.Div([
             html.Div([
                dcc.Graph(id='bar-month-with-slider')
             ]),
            html.Div([
                dcc.Slider(
                    id='bar-year-slider',
                    min=df.index.year.min(),
                    max=df.index.year.max(),
                    value=2016,
                    step=None,
                    marks={year: str(year) for year in df.index.year}
                )
            ], style={'background-color':'white'}),
        ], style={'width': '28%', 'display': 'inline-block', 'padding':'1%', 'margin-bottom':'2%'}),

        html.Div([
            html.Div([
              dcc.Dropdown(
                    id='larval-surveys',
                    options=[
                        {'label':'House Index: percentage of houses infested with larvae', 'value':'hi'},
                        {'label':'Container Index: percentage of water-holding containers infested with larvae', 'value':'ci'},
                        {'label':'Breteau Index: number of positive containers per 100 houses inspected', 'value':'bi'}
                    ],
                    value='bi'
                ),
            ], style={'padding-bottom':'1%'}),
            html.Div([
                dcc.Graph(id='boxplot-with-slider'),
            ])

        ], style={'width': '68%', 'display': 'inline-block', 'padding':'1%'})
    ], style={'width':'100%'}),



    html.Div([
        html.Div([
            meta_barplot()
         ], style={'width': '28%', 'display':'inline-block', 'padding':'1%'}),
         html.Div([
            dcc.Graph(id='meta-bar-month-with-slider'),
        ], style={'width': '68%', 'display':'inline-block', 'padding':'1%'}),
    ], style={'width':'100%'}),




    html.Div([
        html.Div([
            dcc.Graph(id='choropleth-map-year')
        ]),
        html.Div([
            dcc.Slider(
                id='map-year-slider',
                min=df.index.year.min(),
                max=df.index.year.max(),
                value=2016,
                step=None,
                marks={year: str(year) for year in df.index.year}
            )
        ], style={'background-color':'white'})
    ], style={'width':'48%', 'display':'inline-block','padding':'1%'}),

    html.Div([
        html.Div([
            dcc.Graph(id='choropleth-map-month')
        ]),
        html.Div([
            dcc.Slider(
                id='map-month-slider',
                min=1,
                max=12,
                value=3,
                step=None,
                marks={i+1: month for i, month in enumerate(months)}
            )
        ], style={'background-color':'white'})
    ], style={'width':'48%', 'display':'inline-block','padding':'1%'})

],
style={
    'background-color':'#F5F5F5',
    'width':'100%'
})

# =====================================================================================================
# =========================================== Call Back ===============================================
# =====================================================================================================

@app.callback(
    dash.dependencies.Output('boxplot-with-slider', 'figure'),
    [dash.dependencies.Input('bar-year-slider', 'value'),
     dash.dependencies.Input('larval-surveys', 'value')])
def generate_boxplot(selected_year, index_type):
    data = []
    filtered_df = df[df.index.year == selected_year].copy()
    subdist_list = filtered_df['subdist'].unique()
    for subdist in subdist_list:
        tmp = filtered_df.loc[filtered_df['subdist'] == subdist].copy()
        trace = go.Box(
            y=tmp[index_type].values,
            name=subdist,
            boxmean=True,
        )
        data.append(trace)

    figure={
            'data': data,
            'layout': go.Layout(
                title='Boxplot of each Subdistrict',
                height=550,
                yaxis= dict(title='Values'),
#                 xaxis= dict(title=''),
                margin=go.Margin(
                    l=100,
                    r=50,
                    b=140,
                    t=100,
                    pad=4
                ),
            )
        }
    return figure


@app.callback(
    dash.dependencies.Output('bar-month-with-slider', 'figure'),
    [dash.dependencies.Input('bar-year-slider', 'value')])
def update_barplot(selected_year):
    filtered_df = df[df.index.year == selected_year].copy()
    count = dict(Counter(filtered_df.index.month))
    key, val = [], []
    for k in count:
        key.append(months[k-1])
        val.append(count[k])

    trace_bar_actual = go.Bar(
        x = key,
        y = val,
        text = val,
        textposition = 'auto',
        marker=dict(
            color='#6BE59A',
            line=dict(
                color='#05B083',
                width=1.5),
        ),
        opacity=0.8
    )
    layout = go.Layout(
        title='Data points for each Month',
        height=550,
        yaxis= dict(title='Frequency'),
        xaxis= dict(title='Year'),
#         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=go.Margin(
            l=100,
            r=50,
            b=75,
            t=100,
            pad=4
        ),
    )
    fig = go.Figure(data=[trace_bar_actual], layout=layout)
    return fig


@app.callback(
    dash.dependencies.Output('bar-month-with-slider-2', 'figure'),
    [dash.dependencies.Input('bar-year-slider', 'value')])
def update_barplot_2(selected_year):
    filtered_df = df[df.index.year == selected_year].copy()
    arr = []
    subdist_list = filtered_df['subdist'].unique()
    for subdist in subdist_list:
        tmp = filtered_df.loc[filtered_df['subdist'] == subdist].copy()
        arr.append([subdist, len(tmp['bi'])])

    arr = pd.DataFrame.from_records(arr)
    arr.columns = ['subdist', 'freq']
#     arr = arr.sort_values('freq', ascending=0)

    trace = go.Bar(
        x = arr['subdist'],
        y = arr['freq'],
        text = arr['freq'],
        textposition = 'auto',
        marker=dict(
                    color='#96DFF7',
                    line=dict(
                        color='#608E9E',
                        width=1.5),
                ),
        opacity=0.8
    )
    layout = go.Layout(
        title='Data Points for each Subdistrict',
        height=550,
        yaxis= dict(title='Frequency'),
#         xaxis= dict(title='Year'),
        margin=go.Margin(
            l=100,
            r=50,
            b=140,
            t=100,
            pad=4
        ),
    )
    fig = go.Figure(data=[trace], layout=layout)
    return fig

@app.callback(
    dash.dependencies.Output('meta-bar-month-with-slider', 'figure'),
    [dash.dependencies.Input('bar-year-slider', 'value')])
def update_meta_barplot(selected_year):
    filtered_df = df_meta[df_meta.index.year == selected_year].copy()
    count = dict(Counter(filtered_df.index.month))
    key, val = [], []
    for k in count:
        key.append(months[k-1])
        val.append(count[k])

    trace_bar_actual = go.Bar(
        x = key,
        y = val,
        text = val,
        textposition = 'auto',
        marker=dict(
            color='rgb(158,202,225)',
            line=dict(
                color='rgb(8,48,107)',
                width=1.5),
        ),
        opacity=0.8
    )
    layout = go.Layout(
        title='Street View Images for each Month',
        height=550,
        yaxis= dict(title='Frequency'),
        xaxis= dict(title='Year'),
#         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=go.Margin(
            l=100,
            r=50,
            b=75,
            t=100,
            pad=4
        ),
    )
    fig = go.Figure(data=[trace_bar_actual], layout=layout)
    return fig


@app.callback(
    dash.dependencies.Output('choropleth-map-year', 'figure'),
    [dash.dependencies.Input('map-year-slider', 'value'),
     dash.dependencies.Input('larval-surveys', 'value')])
def update_choropleth_map_year(selected_year,  index_type):

    mean, sd = df[index_type].mean(), df[index_type].std()

    filtered_df = df[(df.index.year == selected_year)].copy()

    norm = mpl.colors.Normalize(vmin=mean-sd, vmax=mean+sd)
    cmap = cm.Blues

    polygons = []
    for feature in data_polygon['features']:
        prop = feature['properties']
        district, subdist = prop['AP_TN'], prop['TB_TN']

        value = filtered_df[(filtered_df.district == district) &
                            (filtered_df.subdist == subdist)
                           ].bi.mean()

        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        r,g,b,a = m.to_rgba(value)
        r,g,b,a = str(int(r*255)), str(int(g*255)), str(int(b*255)), str(1.0)
        rgba = 'rgba('+r+','+g+','+b+','+a+')'

        polygons.append(
            dict(
                sourcetype = 'geojson',
                source = feature,
                type = 'fill',
                color = rgba
            )
        )

    data = Data([
        Scattermapbox(
            lat=[mean_lat],
            lon=[mean_lng],
            mode='markers',
            marker=Marker(
                size=0
            ),
            text=['Montreal'],
        )
    ])

    layout = Layout(
        autosize=True,
        hovermode='closest',
        height=600,
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=dict(
                lat=mean_lat+0.25,
                lon=mean_lng
            ),
            pitch=0,
            zoom=8,
            style='light', # dark,satellite,streets,light
            layers=polygons,
        ),
        margin=go.Margin(
            l=10,
            r=10,
            b=10,
            t=10,
            pad=2
        ),
    )
    fig = dict(data=data, layout=layout)
    return fig


@app.callback(
    dash.dependencies.Output('choropleth-map-month', 'figure'),
    [dash.dependencies.Input('map-year-slider', 'value'),
     dash.dependencies.Input('map-month-slider', 'value'),
     dash.dependencies.Input('larval-surveys', 'value')])
def update_choropleth_map_month(selected_year, selected_month, index_type):

    mean, sd = df[index_type].mean(), df[index_type].std()

    filtered_df = df[(df.index.year == selected_year) & (df.index.month == selected_month)].copy()

    norm = mpl.colors.Normalize(vmin=mean-sd, vmax=mean+sd)
    cmap = cm.Blues

    polygons = []
    for feature in data_polygon['features']:
        prop = feature['properties']
        district, subdist = prop['AP_TN'], prop['TB_TN']

        value = filtered_df[(filtered_df.district == district) &
                            (filtered_df.subdist == subdist)
                           ].bi.mean()

        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        r,g,b,a = m.to_rgba(value)
        r,g,b,a = str(int(r*255)), str(int(g*255)), str(int(b*255)), str(1.0)
        rgba = 'rgba('+r+','+g+','+b+','+a+')'

        polygons.append(
            dict(
                sourcetype = 'geojson',
                source = feature,
                type = 'fill',
                color = rgba
            )
        )

    data = Data([
        Scattermapbox(
            lat=[mean_lat],
            lon=[mean_lng],
            mode='markers',
            marker=Marker(
                size=0
            ),
            text=['Montreal'],
        )
    ])


    layout = Layout(
        autosize=True,
        hovermode='closest',
        height=600,
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=dict(
                lat=mean_lat+0.25,
                lon=mean_lng
            ),
            pitch=0,
            zoom=8,
            style='light', # dark,satellite,streets,light
            layers=polygons,
        ),
        margin=go.Margin(
            l=10,
            r=10,
            b=10,
            t=10,
            pad=2
        ),
    )
    fig = dict(data=data, layout=layout)
    return fig



@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/vector_surveillance':
        return vector_surveillance
    elif pathname == '/dengue':
        return dengue_dashboard
    elif pathname == '/':
        return index_page

# if __name__ == '__main__':
#     app.run_server(host='0.0.0.0', port=5000, debug=True)
