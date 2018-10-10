from plotly import tools
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from shapely.geometry import Polygon

import plotly.graph_objs as go
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

from modules import query, residual

data_dir = os.path.join('..','..','data')

def plot_importance(title, features_name, coef):

    coef = coef.reshape(-1,1)
    features_name = np.array(features_name).reshape(-1,1)

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
            line=dict(color='rgb(8,48,107)',width=1.5)
        ),
        opacity=0.8
    )

    layout = go.Layout(
        title=title,
        width=550,
        height=450,
        yaxis= dict(title='Relative Importance'),
        xaxis= dict(title='Breeding site'),
        # font=dict(size=16)
    )

    fig = go.Figure(data=[trace_3], layout=layout)
    iplot(fig)

def get_color_range(df_residual, addrcode, is_reverse, is_category, q1, q2):
    value = df_residual.loc[addrcode]['residual']

    if is_category and q1 != None and q2 != None:
        Max = df_residual.residual.max()
        Min = df_residual.residual.min()

        if value > q2: value = Max
        elif value > q1: value = (Max+Min)/2
        else: value = Min

    if is_reverse:
        return df_residual.residual.max()-value
    return value

def choropleth_plot(data_polygon, df_residual, df_dictionary, map_style='dark', cmap_name='Blues', none_data_rgba='rgba(0,0,0,0.1)',
                    opacity=1, is_reverse=False, is_category=False, q1=None, q2=None, save_file=False, filename=''):
    mapbox_access_token = 'pk.eyJ1IjoiYWxpc2hvYmVpcmkiLCJhIjoiY2ozYnM3YTUxMDAxeDMzcGNjbmZyMmplZiJ9.ZjmQ0C2MNs1AzEBC_Syadg'

    norm = mpl.colors.Normalize(vmin=df_residual.residual.min(), vmax=df_residual.residual.max())
    cmap = cm.get_cmap(cmap_name)

    lats, lngs, texts = [], [], []

    polygons = []
    for feature in data_polygon['features']:
        prop = feature['properties']
        addrcode = int(prop['addrcode'])

        if addrcode not in df_residual.index:
            rgba = none_data_rgba
            residual_text = 'no data'
        else:
            residual_text = str(np.round(df_residual.loc[addrcode]['residual'],2))
            value = round(get_color_range(df_residual, addrcode, is_reverse, is_category, q1, q2),2)

            m = cm.ScalarMappable(norm=norm, cmap=cmap)
            r,g,b,a = m.to_rgba(value)
            r,g,b,a = str(int(r*255)), str(int(g*255)), str(int(b*255)), str(1)
            rgba = 'rgba('+r+','+g+','+b+','+a+')'

        polygons.append(
            dict(
                sourcetype='geojson',
                source=feature,
                type='fill',
                color=rgba,
                opacity=opacity,
                line=dict(width=2)
            )
        )

        # Hover info
        poly = Polygon(np.squeeze(feature['geometry']['coordinates'][0]))
        lngs.append(poly.centroid.x)
        lats.append(poly.centroid.y)

        dict_info = query.get_dict_info(df_dictionary, addrcode)
        if len(dict_info) == 0:
            texts.append(
                'Residual error: '+residual_text+'<br>'+\
                'ADDRCODE: '+str(addrcode)
            )
        else:
            texts.append(
                'Residual error: '+residual_text+'<br>'+\
                'ADDRCODE: '+str(addrcode)+'<br>'+\
                'Province: '+dict_info.province_th + ' ('+dict_info.province_en+')<br>'+\
                'District: '+dict_info.district_th + ' ('+dict_info.district_en+')<br>'+\
                'Sub-district: '+dict_info.subdistrict_th + ' ('+dict_info.subdistrict_en+')'
            )

    data = Data([
        Scattermapbox(
            lat=lats,
            lon=lngs,
            mode='markers',
            marker=Marker(
                size=0
            ),
            text=texts
        )
    ])

    layout = Layout(
        autosize=True,
        hovermode='closest',
        width=1600,
        height=900,
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=dict(
                lat=8.4029487,
                lon=99.9210635
            ),
            pitch=0,
            zoom=8,
            style=map_style,
            layers=polygons,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig = dict(data=data, layout=layout)

    if not save_file:
        iplot(fig, filename='Mapbox')

    if save_file and filename != '':
        plot(fig, filename=os.path.join(data_dir,'maps', 'html', filename), auto_open=False)
