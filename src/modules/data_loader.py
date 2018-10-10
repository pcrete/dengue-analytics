import os
import pandas as pd
import numpy as np
import urllib.request, json
from shapely.geometry import Polygon

class df_loader:
    
    def __init__(self):
        self.df_survey = None

    def load_survey(self):
        self.df_survey = pd.read_csv('../data/breeding-sites/larval-survey.csv') 
        self.df_survey = self.df_survey.replace(0, np.nan)
        self.df_survey = self.df_survey.dropna(axis=0, how='any')
        self.df_survey = self.df_survey.reset_index(drop=True)
        self.df_survey = self.df_survey.loc[self.df_survey['province'] == 'นครศรีธรรมราช']
        self.df_survey = self.df_survey.drop('province', axis=1)
        self.df_survey['date'] = pd.to_datetime(self.df_survey['date'], format='%Y-%m')
        self.df_survey = self.df_survey.set_index('date')
        self.df_survey = self.df_survey.sort_index()
        self.df_survey = self.df_survey['2015':'2018']
        return self.df_survey

    def load_filterd(self, index):
        df_filtered = []
        subdist_list = self.df_survey['subdist'].unique()
        for subdist in subdist_list:
            tmp = self.df_survey.loc[self.df_survey['subdist'] == subdist].copy()
            if len(tmp) == 1 and tmp[index].mean() < 100:
                df_filtered.append(tmp.copy())
            df_filtered.append(tmp[np.abs(tmp[index]-tmp[index].mean()) <= (1*tmp[index].std())].copy())

        df_filtered = pd.concat(df_filtered, axis=0)
        return df_filtered

    def load_area(self):
        URL='https://raw.githubusercontent.com/pcrete/Mosquito_Breeding_Sites_Detector/' + \
            'master/geojson/province/' + \
            '%E0%B8%99%E0%B8%84%E0%B8%A3%E0%B8%A8%E0%B8%A3%E0%B8%B5%E0%B8%98%E0%B8%A3%E0%B8' + \
            '%A3%E0%B8%A1%E0%B8%A3%E0%B8%B2%E0%B8%8A.geojson'

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
        return df_area

    def load_detect(self):
        df_detect = pd.read_csv('../data/breeding-sites/sum-detection.csv') 
        df_detect['date'] = pd.to_datetime(df_detect['date'], format='%Y-%m')
        df_detect = df_detect.set_index('date')
        df_detect = df_detect.sort_index()
        return df_detect

    def load_population(self):
        df_population = pd.read_csv('../data/population.csv') 
        return df_population

    def load_cases(self):
        df_dengue_cases = pd.read_csv('../data/dengue-cases/dengue_cases_2016.csv') 
        return df_dengue_cases
    
    
    