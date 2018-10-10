import numpy as np
import pandas as pd

from modules import query, residual
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

def get_residuals(y_trend, y_pred, names):
    residual_entry = []
    for i in range(len(y_trend)):
        diff = round(np.abs(y_trend[i]-y_pred[i]), 4)
        residual_entry.append([names[i], diff])

    df_residual = pd.DataFrame(residual_entry, columns=['addrcode','residual'])
    df_residual = df_residual.set_index('addrcode')

    return df_residual

def get_category(df_residual, addrcode, q1, q2):
    value = df_residual.loc[addrcode]['residual']

    if value > q2: return 'Bad'
    elif value > q1: return 'Average'
    else: return 'Good'


def get_residual_corr(df_detect, df_population, df_area, df_residual, df_gsv_coverage, q1, q2):
    addrcodes = df_residual.index.values
    populations, land_areas, detects, categories, image_areas = [], [], [], [], []
    for addrcode in addrcodes:
        detects.append(query.get_detect(df_detect, addrcode))
        populations.append(query.get_population(df_population, addrcode))
        land_areas.append(query.get_area(df_area, addrcode))
        categories.append(residual.get_category(df_residual, addrcode, q1, q2))
        image_areas.append(query.get_gsv_coverage(df_gsv_coverage, addrcode))

    df_residual_corr = df_residual.copy()
    df_residual_corr['bs_counts'] = detects
    df_residual_corr['category'] = categories
    df_residual_corr['population'] = populations
    df_residual_corr['land_area'] = land_areas
    df_residual_corr['image_area'] = image_areas
    df_residual_corr['gsv_coverage'] = df_residual_corr['image_area']/df_residual_corr['land_area']
    df_residual_corr['population_density'] = df_residual_corr['population']/df_residual_corr['land_area']
    df_residual_corr['bs_land_density'] = df_residual_corr['bs_counts']/df_residual_corr['land_area']
    df_residual_corr['bs_image_density'] = df_residual_corr['bs_counts']/df_residual_corr['image_area']

    return df_residual_corr
