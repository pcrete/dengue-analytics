import numpy as np
import pandas as pd

def get_detect(df_detect, addrcode, columns=None):
    if columns is None:
        detect = round(df_detect.loc[df_detect['addrcode'] == addrcode].mean()['total'], 2)
#         cup =round(df_detect.loc[df_detect['addrcode'] == addrcode].mean()['cup'], 2)
#         vase = round(df_detect.loc[df_detect['addrcode'] == addrcode].mean()['vase'], 2)
#         detect = detect-cup-vase
    else:
        # Breeding Site Feature
        detect = df_detect.loc[df_detect['addrcode'] == addrcode][columns].copy()
    return detect

def filter_survey(df_survey, index='bi'):
    df_filtered = []
    for addrcode in df_survey['addrcode'].unique():
        tmp = df_survey.loc[df_survey['addrcode'] == addrcode].copy()
        if len(tmp) == 1 and tmp[index].mean() < 100:
            df_filtered.append(tmp.copy())
        df_filtered.append(tmp[np.abs(tmp[index]-tmp[index].mean()) <= (1*tmp[index].std())].copy())
    df_filtered = pd.concat(df_filtered, axis=0)
    return df_filtered

def get_survey(df_survey, dengue_season, addrcode):
    if dengue_season:
        months = [6,7,8,9,10,11]
        title = 'Dengue Season'
    else:
        months = [1,2,3,4,5,6,7,8,9,10,11,12]
        title = 'Entire Year'

    survey = round(df_survey.loc[
        (df_survey['addrcode'] == addrcode) &
        (df_survey.index.month.isin(months))
    ]['bi'].mean(), 2)

    return survey, title

def get_area(df_area, addrcode):
    area = round(df_area.loc[df_area['addrcode'] == addrcode]['area'].mean(), 2)
    return area

def get_population(df_population, addrcode):
    population = round(df_population.loc[df_population['addrcode'] == addrcode]['population'].mean(), 2)
    return population

def get_gsv_month(df_detect, addrcode):
    month = df_detect.loc[df_detect['addrcode'] == addrcode].index.month[0]
    return month

def get_gsv_coverage(df_gsv_coverage, addrcode):
    coverage = df_gsv_coverage.loc[df_gsv_coverage['addrcode'] == addrcode]['image_area'].mean()
    return coverage

def get_dict_info(df_dictionary, addrcode):
    dict_info = df_dictionary.loc[df_dictionary['addrcode'] == addrcode]
    return dict_info
