from math import gamma
from turtledemo.penrose import start

import numba
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.stats as stats
import plotly.graph_objects as go
import streamlit as st
import seaborn as sns
from django.db.models.expressions import result
from fitter import Fitter
from pyvis.network import Network
import streamlit.components.v1 as components
from collections import Counter, defaultdict
from datetime import datetime
from scipy.linalg import expm
from dateutil.relativedelta import relativedelta
import time
import os
import numexpr
import statsmodels.api as sm
import networkx as nx
from scipy.stats import beta, alpha
from numpy.random import dirichlet
from sklearn.metrics import mean_squared_error, mean_absolute_error
# from current_test import alpha_prior, beta_prior, transitions
from distribution_by_sector import expert_test
from sklearn.model_selection import train_test_split
import time

"""
    Helpful materials for my diploma 

    https://cyberleninka.ru/article/n/primenenie-matrits-migratsiy-v-zadachah-otsenki-i-upravleniya-kreditnym-riskom-bankov-v-ramkah-podhoda-na-osnove-vnutrennih-reytingov/viewer
    http://pe.cemi.rssi.ru/pe_2009_1_105-138.pdf
    https://sdo2.irgups.ru/pluginfile.php/281033/mod_resource/content/0/Примеры%20решения%20задач%20по%20теме%20«Марковские%20процессы».pdf
    https://docs.kanaries.net/topics/Streamlit/streamlit-plotly

"""
# expert = {'AAA': 1, 'AA': 2, 'A': 3, 'BBB': 4,
#           'BB': 5, 'B': 6, 'CCC': 7, 'CC': 8, 'C': 9, 'D': 10, 'Рейтинг отозван': 11}
# expert_test = {'AAA': 1, 'AA': 2, 'A': 3, 'BBB': 4,
#                'BB': 5, 'B': 6, 'CCC': 7, 'CC': 8, 'C': 9, 'D': 10}
# expert_test = {'AAA': 1, 'AA': 2, 'A': 3, 'BBB': 4,
#                'BB': 5, 'B': 6, 'C': 7, 'D': 8}
expert_test = {
    "AAA-AA": 1,
    "AA-A": 2,
    "A-BBB": 3,
    "BB": 4,
    "B": 5,
    "CCC-C": 6,
    "D": 7
}
# expert_test = {
#     "AAA-AA": 1,
#     "A-BBB": 2,
#     "D": 3
# }
group_expert = {1: 'AAA', 'AA': 'AA', 'A': 'A', 'BBB': 'BBB',
               'BB': 'BB', 'B': 'B', 'CCC': 'C', 'CC': 'C', 'C': 'C', 'D': 'D'}
# expert_test = {
#     'ruAAA': 1,
#     'ruAA+': 2,
#     'ruAA': 3,
#     'ruAA-': 4,
#     'ruA+': 5,
#     'ruA': 6,
#     'ruA-': 7,
#     'ruBBB+': 8,
#     'ruBBB': 9,
#     'ruBBB-': 10,
#     'ruBB+': 11,
#     'ruBB': 12,
#     'ruBB-': 13,
#     'ruB+': 14,
#     'ruB': 15,
#     'ruB-': 16,
#     'ruCCC': 17,
#     'ruCC': 18,
#     'ruC': 19,
#     'ruD': 20,
#     # 'Рейтинг отозван': 21}
# }
# NCR_test = {
#     'AAA.ru': 1,
#     'AA+.ru': 2,
#     'AA.ru': 3,
#     'AA-.ru': 4,
#     'A+.ru': 5,
#     'A.ru': 6,
#     'A-.ru': 7,
#     'BBB+.ru': 8,
#     'BBB.ru': 9,
#     'BBB-.ru': 10,
#     'BB+.ru': 11,
#     'BB.ru': 12,
#     'BB-.ru': 13,
#     'B+.ru': 14,
#     'B.ru': 15,
#     'B-.ru': 16,
#     'CCC.ru': 17,
#     'CC.ru': 18,
#     'C.ru': 19,
#     'D.ru': 20,
#     # 'Рейтинг отозван': 20
# }

NCR_test = {
    'AAA': 1,
    'AA': 2,
    'A': 3,
    'BBB': 4,
    'BB': 5,
    'B': 6,
    'C': 7,
    'D': 8,
}
# expert_test = {
#     'A' : 1,
#     'B' : 2,
#     'D' : 3
# }
akra = {
    'AAA': 1,
    'AA': 2,
    'A': 3,
    'BBB': 4,
    'BB': 5,
    'B': 6,
    'C': 7,
    'D': 8,
}
# akra = {
#     'AAA(RU)': 1,
#     'AA+(RU)': 2,
#     'AA(RU)': 3,
#     'AA-(RU)': 4,
#     'A+(RU)': 5,
#     'A(RU)': 6,
#     'A-(RU)': 7,
#     'BBB+(RU)': 8,
#     'BBB(RU)': 9,
#     'BBB-(RU)': 10,
#     'BB+(RU)': 11,
#     'BB(RU)': 12,
#     'BB-(RU)': 13,
#     'B+(RU)': 14,
#     'B(RU)': 15,
#     'B-(RU)': 16,
#     'CCC(RU)': 17,
#     'CC(RU)': 18,
#     'C(RU)': 19,
#     'D(RU)': 20,
#     # 'Рейтинг отозван': 20
# }

s_and_p = {
    'ruAAA': 1,
    'ruAA+': 2,
    'ruAA': 3,
    'ruAA-': 4,
    'ruA+': 5,
    'ruA': 6,
    'ruA-': 7,
    'ruBBB+': 8,
    'ruBBB': 9,
    'ruBBB-': 10,
    'ruBB+': 11,
    'ruBB': 12,
    'ruBB-': 13,
    'ruB+': 14,
    'ruB': 15,
    'ruB-': 16,
    'ruCCC+': 17,
    'ruCCC': 18,
    'ruCCC-': 19,
    'ruCC': 20,
    'ruC': 21,
    'ruD': 22
}
fitch = {
    'AAA(rus)': 1,
    'AA+(rus)': 2,
    'AA(rus)': 3,
    'AA-(rus)': 4,
    'A+(rus)': 5,
    'A(rus)': 6,
    'A-(rus)': 7,
    'BBB+(rus)': 8,
    'BBB(rus)': 9,
    'BBB-(rus)': 10,
    'BB+(rus)': 11,
    'BB(rus)': 12,
    'BB-(rus)': 13,
    'B+(rus)': 14,
    'B(rus)': 15,
    'B-(rus)': 16,
    'CCC+(rus)': 17,
    'CCC(rus)': 18,
    'CCC-(rus)': 19,
    'CC(rus)': 20,
    'C(rus)': 21,
    'D(rus)': 22
}
moodys = {'Aaa.ru': 1,
          'Aa1.ru': 2,
          'Aa2.ru': 3,
          'Aa3.ru': 4,
          'A1.ru': 5,
          'A2.ru': 6,
          'A3.ru': 7,
          'Baa1.ru': 8,
          'Baa2.ru': 9,
          'Baa3.ru': 10,
          'Ba1.ru': 11,
          'Ba2.ru': 12,
          'Ba3.ru': 13,
          'B1.ru': 14,
          'B2.ru': 15,
          'B3.ru': 16,
          'Caa1.ru': 17,
          'Caa2.ru': 18,
          'Caa3.ru': 19,
          'Ca.ru': 20,
          'C.ru': 21,
          }
nra = {
    'AAA|ru|': 1,
    'AA+|ru|': 2,
    'AA|ru|': 3,
    'AA-|ru|': 4,
    'A+|ru|': 5,
    'A|ru|': 6,
    'A-|ru|': 7,
    'BBB+|ru|': 8,
    'BBB|ru|': 9,
    'BBB-|ru|': 10,
    'BB+|ru|': 11,
    'BB|ru|': 12,
    'BB-|ru|': 13,
    'B+|ru|': 14,
    'B|ru|': 15,
    'B-|ru|': 16,
    'CCC|ru|': 17,
    'CC|ru|': 18,
    'C|ru|': 19,
    'D|ru|': 20,
    # 'Рейтинг отозван': 21
}

default = {'Expert RA': 'D', 'Fitch Ratings': 'D(rus)',
           'S&P Global Ratings': 'ruD', 'AKRA': 'D(RU)', "Moody's Interfax Rating Agency": 'C.ru', 'NRA': 'D|ru|',
           'NCR': 'D'}

upload = st.sidebar.file_uploader("Choose a XLSX file")


@st.cache_data
def load_data(upload):
    if upload is not None:
        df = pd.read_excel(upload)
        # df.columns = df.iloc[0]
        # df = df[1:]
        # df = df.reset_index().drop(columns=['index'])
        return df

def convert_ratings(data: pd.DataFrame, agency: str):
    correct_dict = {'A++': 'AAA', 'A+ (I)': 'AA', 'A+ (II)': 'A', 'A+ (III)': 'A',
                    'A+': 'A', 'A (I)': 'BBB', 'A (II)': 'BBB', 'A': 'BBB',
                    'A (III)': 'BB', 'B++': 'B', 'B++ (III)': 'B', 'B+': 'B', 'B': 'CCC', 'C++': 'CC', 'C+': 'CC',
                    'A++.mfi': 'BBB', 'A+.mfi': 'BB', 'A.mfi': 'BB', 'B++.mfi': 'B', 'B+.mfi': 'B', 'B.mfi': 'CCC',
                    'E': 'D'}
    correct_dict_2 = {
        'ruAAA': 'AAA', 'ruAA+': 'AA', 'ruAA': 'AA', 'ruAA-': 'AA',
        'ruA+': 'A', 'ruA': 'A', 'ruA-': 'A',
        'ruBBB+': 'BBB', 'ruBBB': 'BBB', 'ruBBB-': 'BBB',
        'ruBB+': 'BB', 'ruBB': 'BB', 'ruBB-': 'BB',
        'ruB+': 'B', 'ruB': 'B', 'ruB-': 'B',
        'ruCCC': 'CCC', 'ruCC': 'CC', 'ruC': 'C', 'ruRD': 'RD', 'ruD': 'D'
    }

    data = data.reset_index()
    for i in data.index:
        if data['agency'][i] == agency:
            if int(data['_date'][i].split('-')[0]) >= 2016:
                if data['rating'][i] in correct_dict_2.keys():
                    st.write(data['_name'][i], data['rating'][i], correct_dict_2[data['rating'][i]])
                    data.at[i, 'rating'] = correct_dict_2[data['rating'][i]]
            # if int(data['_date'][i].split('-')[0]) <= 2019:
            #     if data['rating'][i] in correct_dict.keys():
            #         st.write(data['_name'][i], data['rating'][i], correct_dict[data['rating'][i]])
            #         data.at[i, 'rating'] = correct_dict[data['rating'][i]]

    data.to_excel('TO_WORK_WITH_final_last.xlsx')
    stop = 'here'


def find_dubl(data):
    pass


def ranking_ratings(data):
    converted = {'AAA': 1, 'AA': 2, 'A': 3, 'BBB': 4,
                 'BB': 5, 'B': 6, 'CCC': 7, 'CC': 8, 'C': 9, 'D': 10, 'отозван': 10}

    for ind in data.index:
        for key, value in converted.items():
            if data['raitng'][ind] == key:
                data.at[ind, 'rating'] = value

    return data



def graph_matric_migration_stat(data):
    # network = nx.DiGraph(directed=True)
    # nodes = data.columns
    # network.add_nodes_from(nodes)
    # edges = []
    # for col in data.columns:
    #     for ind in data.index[data[col] > 0].tolist():
    #         edges.append((col, ind, data[col][ind]))
    #
    # edges_final = []
    # for tup in edges:
    #     edges_final.append((tup[0], tup[1]))
    # network.add_edges_from(edges_final)
    # nx.draw_circular(network, with_labels=True, font_weight='bold')
    network = nx.DiGraph(directed=True)

    # Добавляем узлы
    nodes = data.index.tolist()
    network.add_nodes_from(nodes)

    # Добавляем связи на основе строк
    edges = []
    for row in data.index:
        for col in data.columns[data.loc[row] > 0].tolist():
            edges.append((row, col))

    network.add_edges_from(edges)
    # Настраиваем параметры для узлов и подписей
    node_size = 1000  # Размер узлов
    font_color = 'red'  # Цвет подписей
    font_size = 20  # Размер текста подписей
    node_color = 'none'  # Без заливки узлов
    edge_color = 'white'  # Цвет контура узлов

    # Рисуем граф
    nx.draw_circular(
        network,
        with_labels=True,
        font_weight='bold',
        node_size=node_size,
        font_color=font_color,
        font_size=font_size,
        node_color=node_color,  # Узлы без заливки
        edgecolors=edge_color  # Цвет контура узлов
    )
    plt.show()


def graph_matric_migration(data):
    # data = data.T
    network = Network(height='710px',
                      width='100%',
                      bgcolor='#222222',
                      font_color='white',
                      directed=True,
                      notebook=True, )
    # G = nx.Graph()  # создаём объект графа
    nodes = data.columns
    network.add_nodes(nodes=nodes)
    edges = []
    for col in data.columns:
        for ind in data.index[data[col] > 0].tolist():
            edges.append((col, ind, data[col][ind]))

    for tup in edges:
        network.add_edge(tup[0], tup[1], title=tup[2])
    # network.add_edges(edges)
    name = 'Simple_Network_Graph.html'
    network.save_graph(name)
    return name


def moved_to(data: pd.DataFrame) -> dict:
    pass


def get_prev_date_raitng(data, ogrn, start_date, step, col_ogrn, col_date, col_rating) -> list:
    data_1 = data.sort_values(col_date).reset_index().drop(columns=['index'])
    time_counter = None
    if 'months' in step.keys():
        time_counter = 31

    if 'years' in step.keys():
        time_counter = 365

    if 'days' in step.keys():
        time_counter = 1

    if len(data_1) > 0:
        rating = data_1[col_rating][len(data_1) - 1]
        date = data_1[col_date][len(data_1) - 1]
        # if rating == default[data_1['agency'][0]]:
        #     rating = ''
        # years = (datetime.strptime(start_date, "%Y-%m-%d") - datetime.strptime(data_1[col_date][len(data_1) - 1],
        #                                                                        "%Y-%m-%d")).days // time_counter
        # if len(data_1[data_1[col_date] == date]) > 1 and 'отозван' in data_1[data_1[col_date] == date][col_rating].values:
        #     rating = ''
        # st.write(rating, date)
        # if len(data_1) == 1 and default[agency] in data_1[col_rating].values:
        #     rating = ''
        return [rating, date]
    else:
        return [None, None]


def sort_df(result_df: pd.DataFrame, agency_dict: dict) -> pd.DataFrame:
    if len(result_df.columns) > 1:
        # result_df = result_df.reset_index().drop(columns=['level_0'])
        for ind in result_df.index:
            for key, value in agency_dict.items():
                if result_df['index'][ind] == key:
                    result_df.at[ind, 'index'] = value

        result_df = result_df.sort_values('index')
        for ind in result_df.index:
            for key, value in agency_dict.items():
                if result_df['index'][ind] == value:
                    result_df.at[ind, 'index'] = key

        result_df = result_df.set_index('index').T
        result_df = result_df.reset_index()

        for ind in result_df.index:
            for key, value in agency_dict.items():
                if result_df['index'][ind] == key:
                    result_df.at[ind, 'index'] = value

        result_df = result_df.sort_values('index')
        for ind in result_df.index:
            for key, value in agency_dict.items():
                if result_df['index'][ind] == value:
                    result_df.at[ind, 'index'] = key

        result_df = result_df.set_index('index')

        if len(result_df.index) > len(result_df.columns):
            for ind in result_df.index:
                if ind not in result_df.columns:
                    result_df = pd.concat([result_df,
                                           pd.DataFrame([0 for _ in range(len(result_df.index))], columns=[ind],
                                                        index=result_df.index)], axis=1)
        elif len(result_df.index) < len(result_df.columns):
            for col in result_df.columns:
                if col not in result_df.index:
                    result_df = pd.concat([result_df, pd.DataFrame([0 for _ in range(len(result_df.index))],
                                                                   columns=result_df.columns, index=[col])], axis=0)
    else:
        result_df = result_df.reset_index()
        for ind in result_df.index:
            for key, value in agency_dict.items():
                if result_df['index'][ind] == key:
                    result_df.at[ind, 'index'] = value

        result_df = result_df.sort_values('index')
        for ind in result_df.index:
            for key, value in agency_dict.items():
                if result_df['index'][ind] == value:
                    result_df.at[ind, 'index'] = key

        result_df = result_df.set_index('index')

    return result_df


def fill_empty(dict_: dict, agency_dict: dict):
    for key, val in agency_dict.items():
        for key_, val_ in dict_.items():
            if key not in dict_:
                dict_[key] = {}
            if key_ not in dict_[key]:
                dict_[key][key_] = 0.0
    return dict_


def get_generator(result_df: pd.DataFrame) -> pd.DataFrame:
    for i, v in enumerate(result_df.sum(axis=1)):
        result_df.at[result_df.index[i], result_df.index[i]] = -v

    return result_df


def get_nan_df(agency_dict):
    result = {}
    for key in agency_dict.keys():
        if key not in result:
            result[key] = {}
        for key_doub in agency_dict.keys():
            if key_doub not in result[key]:
                result[key][key_doub] = 0.0

    return pd.DataFrame().from_dict(result)


@st.cache_data(experimental_allow_widgets=True)
def calculate_discrete_migr(data: pd.DataFrame, agency: str, start_date: str, end_dates: str, scale: list, step, col_ogrn: str, col_date: str, col_rating: str):
    agency_dict = {}

    full_step = None
    curent_step = None
    time_counter = None
    step_num = None

    if 'months' in step.keys():
        step_num = step['months']
        full_step = relativedelta(months=1)
        curent_step = relativedelta(months=step_num)
        time_counter = 30

    if 'years' in step.keys():
        step_num = step['years']
        full_step = relativedelta(years=1)
        curent_step = relativedelta(years=step_num)
        time_counter = 363

    if 'days' in step.keys():
        step_num = step['days']
        full_step = relativedelta(days=1)
        curent_step = relativedelta(days=step_num)
        time_counter = 1

    if agency == 'Expert RA':
        agency_dict = expert_test
    if agency == 'NCR':
        agency_dict = NCR_test
    if agency == 'AKRA':
        agency_dict = akra
    if agency == 'S&P Global Ratings':
        agency_dict = s_and_p
    if agency == 'Fitch Ratings':
        agency_dict = fitch
    if agency == "Moody's Interfax Rating Agency":
        agency_dict = moodys
    if agency == 'NRA':
        agency_dict = nra

    full_df = get_nan_df(agency_dict)
    counter_AA = 0
    counter_AAA = 0
    # while start_date <= (datetime.strptime(end_dates, "%Y-%m-%d") + full_step).strftime('%Y-%m-%d'):
    #     result = {}
    #     check_ogrn = []  # todo array of orn to cancel dubl
    #     result_migr = {}
    #     data_prev = data[(data['agency'] == agency) & (data['_date'] < start_date)]
    #     end_date = (datetime.strptime(start_date, "%Y-%m-%d") + curent_step).strftime('%Y-%m-%d')
    data_1 = data
    data_1 = data_1.sort_values(col_date)
    st.write(len(data_1[col_ogrn].unique()))
    counter = 0
    set_ogrn = (data_1[col_ogrn].unique())
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    result = {}
    result_migr = {}
    counter_CC_D = 0
    for ogrn in set_ogrn:
        if pd.isna(ogrn) != True:
            time.sleep(0.01)
            my_bar.progress(int(100 * counter / len(set_ogrn)), text=progress_text)
            counter += 1
            # result = {}
            # result_migr = {}
            pr = data_1.loc[data_1[col_ogrn] == ogrn].reset_index(drop=True).sort_values(col_date)
            # start_dates = pr[col_date][0]
            start_dates = start_date
            # end_dates = pr[col_date][len(pr) - 1]
            while start_dates < (datetime.strptime(end_dates, "%Y-%m-%d")).strftime('%Y-%m-%d'):
                data_prev = pr[(pr[col_date] < start_dates)]
                end_date = (datetime.strptime(start_dates, "%Y-%m-%d") + curent_step).strftime('%Y-%m-%d')
                temp_df = pr.loc[(pr[col_date] >= start_dates) & (pr[col_date] <= end_date)].reset_index(drop=True).sort_values(col_date)

                # st.write(start_date, end_date)
                first = ''
                first_date = None
                last = ''
                if len(temp_df) > 0:
                    # first = temp_df[col_rating][0]
                    if temp_df[col_date][0] == start_dates:
                        first = temp_df[col_rating][0]
                    else:
                        if len(get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)) > 0:
                            first, first_date = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)  # todo get previous rating on date = start_date
                            # first_date = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)[1]
                    temp_local_df = temp_df.loc[temp_df[col_date] == temp_df[col_date][len(temp_df) - 1]]
                    if len(temp_local_df) > 1 and 'Рейиинг отозван' in temp_local_df[col_rating].values:
                        last = 'Рейтинг отозван'
                    else:
                        last = temp_df[col_rating][len(temp_df) - 1]

                    if last == 'Рейтинг отозван':
                        temp_local_df = temp_df.loc[temp_df[col_date] == temp_df[col_date][len(temp_df) - 1]]
                        if len(temp_local_df) > 1 and default[agency] in temp_local_df[col_rating].values:
                            last = default[agency]
                else:
                    if len(get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)) > 0:
                        first, first_date = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)  # todo get previous rating on date = start_date
                        # first_date = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)[1]
                        last = first

                if first_date != None:
                    # st.write(first_date != None)
                    print(first_date, type(first_date))
                    years = (datetime.strptime(start_dates, "%Y-%m-%d") - datetime.strptime(first_date,
                                                                                            "%Y-%m-%d")).days // 365
                    if years > 1:
                        # st.write(first, ogrn, start_dates, end_date)
                        # st.write(temp_df['_name'][0], prev_, start_date, first)
                        first = ''

                if first != '' and first != default[agency]:
                    # if first != default[agency] and last != 'отозван':
                        if first in agency_dict and last in agency_dict:
                            if first not in result_migr:
                                result_migr[first] = []
                            result_migr[first].append(last)

                start_dates = (datetime.strptime(start_dates, "%Y-%m-%d") + full_step).strftime('%Y-%m-%d')

    for key, value in agency_dict.items():
        if key not in result:
            result[key] = {}
        if key in result_migr.keys():
            if key not in result:
                result[key] = {}
            value_1 = result_migr[key]
            temp_dict = {}
            for i in range(len(set(value_1))):
                temp_dict[list(Counter(value_1).keys())[i]] = list(Counter(value_1).values())[i]
            result[key] = temp_dict

    result_df = pd.DataFrame().from_dict(result).fillna(0).reset_index()
    result_df = sort_df(result_df, agency_dict)
    full_df += result_df

    sum_mat = full_df.sum(axis=1)
    # st.write(full_df)
    # st.write(sum_mat)

    for ind in full_df.index:
        sum_ = sum_mat[ind]
        if sum_ == 0.0:
            full_df.at[ind, ind] = 1.
        for col in full_df.columns:
            if sum_ > 0:
                full_df.at[ind, col] /= round(sum_, 3)

    time.sleep(1)
    my_bar.empty()

    st.write(counter_CC_D)
    return [full_df, sum_mat]


def get_state_by_time(data: pd.DataFrame, agency: str, start_date: str, step: dict, scale: list, col_ogrn, col_date, col_rating, n: int):
    agency_dict = {}
    state_from = {}
    state_to = {}
    check_ogrn = []
    if agency == 'Expert RA':
        agency_dict = expert_test
    if agency == 'NCR':
        agency_dict = NCR_test
    if agency == 'AKRA':
        agency_dict = akra
    if agency == 'S&P Global Ratings':
        agency_dict = s_and_p
    if agency == 'Fitch Ratings':
        agency_dict = fitch
    if agency == "Moody's Interfax Rating Agency":
        agency_dict = moodys
    if agency == 'NRA':
        agency_dict = nra




    if 'months' in step.keys():
        step_num = step['months']
        full_step = relativedelta(months=1)
        curent_step = relativedelta(months=step_num * n)
        time_counter = 30

    if 'years' in step.keys():
        step_num = step['years']
        full_step = relativedelta(years=1)
        curent_step = relativedelta(years=step_num * n)
        time_counter = 363

    if 'days' in step.keys():
        step_num = step['days']
        full_step = relativedelta(days=1)
        curent_step = relativedelta(days=step_num * n)
        time_counter = 1

    end_date = (datetime.strptime(start_date, "%Y-%m-%d") + curent_step).strftime('%Y-%m-%d')
    data_1 = pd.DataFrame()
    if len(scale) != 0:
        for scal in scale:
            data_1 = pd.concat([data_1, data[(data['agency'] == agency) & (data['scale'] == scal)].reset_index().drop(columns=['index'])], ignore_index=True)
    else:
        data_1 = pd.concat([data_1, data[(data['agency'] == agency)].reset_index().drop(columns=['index'])], ignore_index=True)

    set_ogrn = data_1[col_ogrn].unique()
    for ogrn in set_ogrn:
        first = ''
        last = ''
        date_prev = ''
        date_last = ''
        if pd.isnull(ogrn) != True:
                data_prev = data_1[(data_1[col_date] < start_date) & (data_1[col_ogrn] == ogrn)]
                data_prev_to = data_1[(data_1[col_date] < end_date) & (data_1[col_ogrn] == ogrn)]
                temp_df_from = data_1[(data_1[col_ogrn] == ogrn) & (data_1[col_date] == start_date)].reset_index().drop(columns=['index']).sort_values(col_date)
                temp_df_to = data_1[(data_1[col_ogrn] == ogrn) & (data_1[col_date] == end_date)].reset_index().drop(columns=['index']).sort_values(col_date)
                if len(temp_df_from) > 0:
                    first = temp_df_from[col_rating][0]
                else:
                    prev_df = get_prev_date_raitng(data_prev, ogrn, start_date, step, col_ogrn, col_date, col_rating)
                    if len(prev_df) > 0:
                        first = prev_df[0]
                        date_prev = prev_df[1]

                if len(temp_df_to) > 0:
                    last = temp_df_to[col_rating][0]
                else:
                    last_df = get_prev_date_raitng(data_prev_to, ogrn, end_date, step, col_ogrn, col_date, col_rating)
                    if len(last_df) > 0:
                        last = last_df[0]
                        date_last = last_df[1]

                if date_last != '':
                    years = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(date_last,
                                                                                            "%Y-%m-%d")).days // 365
                    if years > 1:
                        # st.write(first, ogrn, start_dates, end_date)
                        # st.write(temp_df['_name'][0], prev_, start_date, first)
                        last = ''

                if date_prev != '':
                    years = (datetime.strptime(start_date, "%Y-%m-%d") - datetime.strptime(date_prev,
                                                                                         "%Y-%m-%d")).days // 364
                    if years > 1:
                        # st.write(first, ogrn, start_dates, end_date)
                        # st.write(temp_df['_name'][0], prev_, start_date, first)
                        first = ''

                if first != '' and last != '':
                    if first in agency_dict and last in agency_dict:
                        if first not in state_from:
                            state_from[first] = []
                        if last not in state_to:
                            state_to[last] = []
                        state_from[first].append(1)
                        state_to[last].append(1)

    for key in agency_dict.keys():
        if key in state_from.keys():
            state_from.update({key: sum(state_from[key])})
        else:
            state_from.update({key: 0})

    for key in agency_dict.keys():
        if key in state_to.keys():
            state_to.update({key: sum(state_to[key])})
        else:
            state_to.update({key: 0})

    df_from = sort_df(pd.Series(state_from).to_frame('Rating'), agency_dict)
    df_to = sort_df(pd.Series(state_to).to_frame('Rating'), agency_dict)


    file = st.file_uploader('Choose xlsx file to upload (predict)')
    df_predict = pd.read_excel(file).set_index('Unnamed: 0')

    for row in df_predict.index:
        for col in df_predict.columns:
            # st.write(row, col, df_predict[row][col])
            # st.write(df_predict[row][col] * df_from['Rating'][row], row, col, df_predict[row][col], df_from['Rating'][row])
            df_predict.at[row, col] *= df_from['Rating'][row]


    df_predict = df_predict.sum(axis=0).to_frame('Rating')
    # st.write(df_to)
    # st.write(df_predict)
    MSE = 0.0
    R_2 = 0.0
    for row in df_to.index:
        for col in df_to.columns:
            MSE += (df_to['Rating'][row] - df_predict['Rating'][row]) ** 2
    #
    MSE = MSE / (len(df_to.index) * len(df_to.columns))
    RMSE = pow(MSE, 1 / 2)

    x = df_predict.values
    x = sm.add_constant(x)
    y = []
    for v in df_to.values:
        y.append(int(v[0]))

    st.write(y)
    # st.write(x)
    model = sm.OLS(y, x)
    results = model.fit()
    st.write(results.summary())

    st.write('MSE=', MSE, 'RMSE=', RMSE)
    return [df_from, df_to]


def matrix_migration(data: pd.DataFrame, agency: str, start_date: str, end_dates: str, scale: list,
                     step: dict, date_to_check: str, directory, type_ogrn, type_date, type_rating):
    st.title('Markov process with discrete time')

    delta = datetime.strptime(end_dates, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")
    agency_dict = {}
    if agency == 'Expert RA':
        agency_dict = expert_test
    if agency == 'NCR':
        agency_dict = NCR_test
    if agency == 'AKRA':
        agency_dict = akra
    if agency == 'S&P Global Ratings':
        agency_dict = s_and_p
    if agency == 'Fitch Ratings':
        agency_dict = fitch
    if agency == "Moody's Interfax Rating Agency":
        agency_dict = moodys
    if agency == 'NRA':
        agency_dict = nra
    check_dict = {}
    pie_cont = {}
    check_ = {}
    full_df = calculate_discrete_migr(data, agency, start_date, end_dates, scale, step, type_ogrn, type_date, type_rating)[0]

    # full_df.to_excel(f'{directory}/discrete_markov_step={step}.xlsx')
    n = 0
    name = ''
    # full_df.index = ['AAA','AA+', 'AA', 'AA-', 'A+', 'A','A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC', 'CC', 'C', 'D']
    # full_df.columns = ['AAA','AA+', 'AA', 'AA-', 'A+', 'A','A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC', 'CC', 'C', 'D']
    fig = plt.figure(figsize=(10, 10))
    plot = sns.heatmap(full_df, annot=True, fmt='.3f', linewidths=.5, annot_kws={"size":11})
    # plt.savefig(f'{directory}/discrete_markov_step={step}.jpg')
    plt.close()

    for col in full_df.columns:  # TODO redact by index
        if full_df[col].sum() > 0:
            fig_ = go.Figure(data=[go.Pie(labels=full_df.index[full_df[col] > 0].tolist(),
                                          values=full_df[col][full_df[col] > 0])])
            fig_.update_layout(
                legend_title=f"{col} rating moved to:",
                font=dict(
                    family="Courier New, monospace",
                    size=15,
                    color="White"
                )
            )
            pie_cont[col] = fig_

    if st.checkbox('Display Pie chart of migration discrete matrix'):
        pies = st.sidebar.multiselect('Choose ratings to see moves', pie_cont)
        for pie_diag in pies:
            st.plotly_chart(pie_cont[pie_diag], theme='streamlit')

    if st.checkbox('Display graph of migration discrete matrix'):
        HtmlFile = open(graph_matric_migration(full_df), 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height=650, width=650)

    if st.checkbox('Display static graph of migration discrete matrix'):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(graph_matric_migration_stat(full_df))

    if st.checkbox('Display migration discrete matrix'):
        # full_df.to_excel(f'discrete_time_step={step}.xlsx')
        st.pyplot(fig)
        fig.savefig((f'results/{agency}/images/discrete_step={step}_{datetime.now().strftime('%Y-%m-%d')}.jpg'))

    if st.checkbox('Display predict discr. migration matrix'):
        fig = plt.figure(figsize=(10, 10))
        n = st.number_input('Enter number', max_value=1000, min_value=2)
        check_1 = np.linalg.matrix_power(full_df, n)
        plot = sns.heatmap(check_1, annot=True, fmt='.3f', linewidths=.5, annot_kws={"size":11})
        # plt.savefig(f'{directory}/time_cont_step={step}_second_avar.jpg')
        st.pyplot(fig)
        fig.savefig((f'results/{agency}/images/predict_{n}_discrete_step={step}_{datetime.now().strftime('%Y-%m-%d')}.jpg'))
        plt.close()

    if st.checkbox('Get predict of migration discrete matrix'):
        # get_state_by_time(data, agency, start_date, scale)
        n = st.number_input('Enter number', max_value=100, min_value=2)
        check_1 = np.linalg.matrix_power(full_df, n)
        st.write(pd.DataFrame(check_1, columns=list(agency_dict.keys()), index=list(agency_dict.keys())))
        check_1 = pd.DataFrame(check_1, columns=list(agency_dict.keys()), index=list(agency_dict.keys()))
        name = f"results/{agency}/discrete_time/predict_discrete_step={step}_predict={n}_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
        check_1.to_excel(name)
        f = Fitter(check_1,
                   distributions=['cauchy', 'chi2', 'expon', 'exponpow', 'gamma', 'lognorm', 'norm', 'powerlaw',
                                  'rayleigh', 'uniform', 'beta'])
        f.fit()
        distr_fit = f.summary()
        st.write(distr_fit)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(f.plot_pdf())
        st.pyplot(f.plot_pdf())

    if st.checkbox('Display distribution of migration discrete matrix '):
        # full_df = get_state_by_time(data, agency, date_to_check, step, scale)[0]
        f = Fitter(full_df,
                   distributions=['cauchy', 'chi2', 'expon', 'exponpow', 'gamma', 'lognorm', 'norm', 'powerlaw',
                                  'rayleigh', 'uniform', 'beta'])
        f.fit()
        distr_fit = f.summary()
        st.write(distr_fit)
        st.pyplot(f.plot_pdf())
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(f.plot_pdf())

    if st.button('Download discrete matrix', key='download'):
        full_df.to_excel(f'results/{agency}/discrete_time/discrete_step={step}_{datetime.now().strftime('%Y-%m-%d')}.xlsx')


def recurs(data):
    # st.write(data)
    if len(data['rating']) == 2:
        return {data['rating'][0]: data['rating'][1]}
    else:
        st.write(data[1:])
        return {data[0]: recurs(data[1:])}


def count_moves(dict_: dict) -> dict:
    temp_ = {}
    counter_rat = []
    result = {}
    for key_, value_ in dict_.items():
        for val in value_:
            if len(val) == 3:
                if key_ not in temp_:
                    temp_[key_] = []
                temp_[key_].append(val[1])

    for key_, value_ in temp_.items():
        if key_ not in result:
            result[key_] = {}
        value_1 = temp_[key_]
        temp_dict = {}
        for i in range(len(set(value_1))):
            temp_dict[list(Counter(value_1).keys())[i]] = list(Counter(value_1).values())[i]
        result[key_] = temp_dict
    return result


def count_time(dict_: dict, start_date: str, end_date: str) -> dict:
    result = {}
    for key_, value_ in dict_.items():
        if key_ not in result:
            result[key_] = []
        for val in value_:
            # if val[1] != key_:
            if len(val) == 3:
                delta = datetime.strptime(val[2], "%Y-%m-%d") - datetime.strptime(val[0], "%Y-%m-%d")
                delta_full = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(
                    start_date, "%Y-%m-%d")).days
                num_days = delta.days
                result[key_].append(num_days / delta_full)
    return result


def count_last(dict_: dict, start_date: str, end_date: str) -> dict:
    result = {}
    for key, value in dict_.items():
        for val in value:
            if len(val) == 5:
                if key not in result:
                    result[key] = []
                delta = datetime.strptime(val[3], "%Y-%m-%d") - datetime.strptime(val[2], "%Y-%m-%d")
                delta_full = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(
                    start_date, "%Y-%m-%d")).days
                num_days = delta.days
                result[key].append(num_days / delta_full)
    return result


def count_pd(dict_: dict, start_date: str, end_date: str):
    result = {}
    for key, value in dict_.items():
        for val in value:
            if len(val) == 5:
                if key not in result:
                    result[key] = []
                delta = datetime.strptime(val[3], "%Y-%m-%d") - datetime.strptime(val[2], "%Y-%m-%d")
                delta_full = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days
                num_days = delta.days
                result[val[len(val) - 1]].append(num_days / delta_full)
    return result


def get_time_cont_avarag(dict_: dict, agency_dit: dict):
    result = {}
    for key in agency_dit.keys():
        if key in dict_.keys() or key in dict_.values():
            if key not in result:
                result[key] = []
            result[key].append(1)
    return result


@st.cache_data(experimental_allow_widgets=True, show_spinner=False)
def calculate_time_cont_migr(data: pd.DataFrame, agency: str, start_date: str, end_dates: str, step: dict, col_ogrn: str, col_date: str, col_rating: str):
    result_num = {}
    result_non_num = {}
    agency_dict = {}
    if agency == 'Expert RA':
        agency_dict = expert_test
    if agency == 'NCR':
        agency_dict = NCR_test
    if agency == 'AKRA':
        agency_dict = akra
    if agency == 'S&P Global Ratings':
        agency_dict = s_and_p
    if agency == 'Fitch Ratings':
        agency_dict = fitch
    if agency == "Moody's Interfax Rating Agency":
        agency_dict = moodys
    if agency == 'NRA':
        agency_dict = nra

    full_df = get_nan_df(agency_dict)
    counter_def = 0

    full_step = None
    curent_step = None
    time_counter = None
    step_num = None

    if 'months' in step.keys():
        step_num = step['months']
        full_step = relativedelta(months=1)
        curent_step = relativedelta(months=step['months'])
        time_counter = 31

    if 'years' in step.keys():
        step_num = step['years']
        full_step = relativedelta(years=1)
        curent_step = relativedelta(years=step['years'])
        time_counter = 365

    if 'days' in step.keys():
        step_num = step['days']
        full_step = relativedelta(days=1)
        curent_step = relativedelta(days=step_num)
        time_counter = 1

    # data_prev = data[(data['agency'] == agency) & (data[col_date] < start_date)]
    data_1 = pd.DataFrame()
    if len(scale) != 0:
        for scal in scale:
            data_1 = pd.concat([data_1, data[
                (data['agency'] == agency) & (data['scale'] == scal)].reset_index().drop(columns=['index'])], ignore_index=True)
    else:
        data_1 = pd.concat([data_1, data[(data['agency'] == agency)].reset_index().drop(columns=['index'])], ignore_index=True)

    data_1 = data_1.sort_values(col_date)
    counter = 0
    set_ogrn = (data_1[col_ogrn].unique())
    progress_text = "Calculate time cont. matrix. Please wait."
    my_bar_1 = st.progress(0, text=progress_text)
    counter_CC_D = 0
    for ogrn in set_ogrn:  # todo iterate over full df
        if pd.isna(ogrn) != True:  # todo if ogrn in not None
            time.sleep(0.01)
            my_bar_1.progress(int(100 * counter / len(set_ogrn)) , text=progress_text)
            counter += 1
            # if data_1['ogrn'] not in check_ogrn:  # todo check ogrn to not dubl
            # check_ogrn.append(data_1['ogrn'][ind])  # todo save the all ogrn which were found
            # start_date + step
            pr = data_1[data_1[col_ogrn] == ogrn].reset_index().drop(columns=['index']).sort_values(col_date)
            # start_dates = start_date # pr[col_date][0]
            start_dates = pr[col_date].iloc[0]
            end_dates = pr[col_date][len(pr) - 1]
            while start_dates < (datetime.strptime(end_dates, "%Y-%m-%d")).strftime('%Y-%m-%d'):

                result = {}
                result_check_full = {}
                result_state_in = {}
                container_state_in = []
                data_prev = pr[(pr[col_date] < start_dates)]
                end_date = (datetime.strptime(start_dates, "%Y-%m-%d") + curent_step).strftime('%Y-%m-%d')
                temp_df = pr[(pr[col_date] >= start_dates) & (pr[col_date] <= end_date)].reset_index().drop(columns=['index']).sort_values(col_date)
                first = ''
                date_start = ''
                # last = ''
                # temp_df = data_1[(data_1['ogrn'] == ogrn)].reset_index().drop(columns=['index']).sort_values('_date')  # todo get df by ogrn
                ind_to_drop_otozv = []
                if len(temp_df) > 0:
                    if temp_df[col_date][0] == start_dates:
                        first = temp_df[col_rating][0]
                        date_start = start_dates
                    else:
                        if len(get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)) > 0:
                            first, date_start = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)  # todo get previous rating on date = start_date
                            # date_start = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)[1]
                else:
                    if len(get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)) > 0:
                        first, date_start = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)  # todo get previous rating on date = start_date
                        # date_start = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)[1]

                if date_start != '':
                    years = (datetime.strptime(start_dates, "%Y-%m-%d") - datetime.strptime(date_start,
                                                                                           "%Y-%m-%d")).days // 365
                    if years > 1:
                        # st.write(temp_df['_name'][0], date_start, start_date, first)
                        first = ''
                # if ogrn == 17:
                #     st.write(first, date_start)
                #     st.write(temp_df, data_prev, start_dates, end_date)
                if first != '' and first != default[agency]:
                    if first not in result_state_in and first in agency_dict:  # todo remove result_state_in from final version(do not forget)
                        result_state_in[first] = []
                    if len(temp_df) > 0:  # todo iterate over exist temp_df with unique ogrn, because in previous step save all unique ogrn
                            first_df = temp_df.loc[[0]]
                            first_df.at[0, col_rating] = first
                            first_df.at[0, col_date] = start_dates
                            temp_df = pd.concat([first_df, temp_df], axis=0).sort_values(col_date).reset_index().drop(columns=['index']).drop_duplicates()
                            if len(set(temp_df[col_rating].values)) == 1:
                                    container_state_in.append(True)
                            elif len(set(temp_df[col_rating].values)) > 1 and first == default[agency] and 'отозван' in temp_df[col_rating].values:
                                    container_state_in.append(True)
                            else:
                                    container_state_in.append(False)
                                    if 'отозван' in temp_df[col_rating].values:
                                        for indx in temp_df.index:
                                            if len(temp_df[temp_df[col_date] == temp_df[col_date][indx]]) > 1 and 'отозван' in \
                                                    temp_df[temp_df[col_date] == temp_df[col_date][indx]][col_rating].values \
                                                    and default[agency] not in temp_df.loc[temp_df[col_date] == temp_df[col_date][indx]][col_rating].values:
                                                        df_otozv = temp_df[temp_df[col_date] == temp_df[col_date][indx]].reset_index()  # todo df to check if any отозван in temp_df by _date
                                                        for k_indx in df_otozv.index:
                                                            if df_otozv[col_rating][k_indx] != 'отозван':
                                                                ind_to_drop_otozv.append(df_otozv['index'][k_indx])

                                            elif len(temp_df.loc[temp_df[col_date] == temp_df[col_date][indx]]) > 1 and 'отозван' in \
                                                    temp_df.loc[temp_df[col_date] == temp_df[col_date][indx]][col_rating].values and \
                                                    default[agency] in temp_df.loc[temp_df[col_date] == temp_df[col_date][indx]][col_rating].values:

                                                    df_otozv = temp_df.loc[temp_df[col_date] == temp_df[col_date][indx]].reset_index()  # todo df to check if any отозван in temp_df by _date
                                                    for k_indx in df_otozv.index:
                                                        if df_otozv[col_rating][k_indx] == 'отозван':
                                                            ind_to_drop_otozv.append(df_otozv['index'][k_indx])
                    else:
                        container_state_in.append(True)

                    if False not in container_state_in and first in agency_dict:
                        result_state_in[first].append(first)  # todo upd each first state with new information
                    # todo if error with 0 index, reset index there in temp_df
                    if False in container_state_in and first in agency_dict:
                        temp_df = temp_df.drop(index=ind_to_drop_otozv).reset_index().drop(columns=['index'])
                        test_rat = temp_df[col_rating].values
                        test_date = temp_df[col_date].values
                        # test_name = temp_df['_name'].values

                        curent_rating = test_rat[0]
                        curent_date = test_date[0]
                        for i, rat in enumerate(test_rat):
                            if rat not in result_check_full:
                                result_check_full[rat] = []
                                if curent_rating in agency_dict and rat in agency_dict:
                                    if curent_rating != rat:
                                        result_check_full[curent_rating].append((curent_date, rat, test_date[i]))
                                        if curent_rating == 'CC.ru' and rat == 'D.ru':
                                            # st.write(curent_date, test_date[i], test_name[i])
                                            counter_CC_D += 1
                                        if curent_rating == "CCC-C" and rat == "CCC-C":
                                            counter_def += 1
                                        curent_rating = rat
                                        curent_date = test_date[i]

                        for i, rat in enumerate(test_rat):
                            if rat not in result_check_full:
                                result_check_full[rat] = []
                            if i + 1 >= len(test_rat):
                                if rat in agency_dict:
                                    last_rat = rat
                                    last_dat = test_date[i]
                                    test_rat_rev = test_rat[::-1]
                                    test_date_rev = test_date[::-1]
                                    # test_name_rev = test_name[::-1]
                                    num = 0
                                    for j, rat_rev in enumerate(test_rat_rev):
                                        if rat_rev != last_rat:
                                            num = j - 1
                                            last_dat = test_date_rev[j]
                                            if test_rat_rev[num] in agency_dict:
                                                break

                                    # st.write(test_rat_rev[num], test_name_rev[num], test_date_rev[num], start_dates, end_date)
                                    if test_rat_rev[num] in agency_dict:
                                        result_check_full[test_rat_rev[num]].append((last_dat, test_rat[len(test_rat) - 1], test_date_rev[num], end_date, rat))

                    for key_ in agency_dict.keys():
                        if key_ not in result_num:
                            result_num[key_] = {}
                        if key_ not in result_non_num:
                            result_non_num[key_] = []
                        count_moves_time = 0.0
                        last_stat = 0.0
                        state_in = 0.0
                        if key_ in count_last(result_check_full, start_dates, end_date):
                            last_stat = sum(count_last(result_check_full, start_dates, end_date)[key_])
                        if key_ in result_state_in.keys():
                            state_in = len(result_state_in[key_])
                        if key_ in count_moves(result_check_full).keys():
                            count_moves_time = sum(count_time(result_check_full, start_dates, end_date)[key_])
                            for k, v in count_moves(result_check_full)[key_].items():
                                if k not in result_num[key_]:
                                    result_num[key_][k] = []
                                # todo use this ti make seocnd method
                                #     result[key_][k] = count_moves(result_check_full)[key_][k] / (state_in + sum(count_time(result_check_full, start_date, end_date)[key_]) + last_stat)
                                result_num[key_][k].append(count_moves(result_check_full)[key_][k])

                        result_non_num[key_].append(state_in + count_moves_time + last_stat)
                start_dates = (datetime.strptime(start_dates, "%Y-%m-%d") + full_step).strftime('%Y-%m-%d')
    # TODO realize method of time-continous process + think about NR rating for first method     07_02_2024
    # st.write(counter_CC_D)
    # st.write(counter_def)
    # st.write(result_num)
    result_full = {}
    for key, val in result_num.items():
        if key not in result_full:
            result_full[key] = {}
        if key in result_non_num.keys():
            for k, v in val.items():
                if k not in result_full[key]:
                        if key == "CCC-C" and k == "CCC-C":
                            st.write(result_num[key][k])
                        result_full[key][k] = (sum(result_num[key][k])) / (sum(result_non_num[key]))

    time.sleep(1)
    my_bar_1.empty()

    # todo use this to make second method
    # st.write(result_full)
    # for key_, val_ in agency_dict.items():
    #     if key_ not in result_full:
    #         result_full[key_] = {}
    #     for key, val in result_full.items():
    #         if key_ == key:
    #             for k, v in val.items():
    #                 if k not in result_full[key_]:
    #                     result_full[key_][k] = 0.0
    # for ind in full_df_2.index:
    #     for col in full_df_2.columns:
    #             if ind in res_avar.keys():
    #                         full_df_2[ind][col] = full_df_2[ind][col] / len(res_avar[ind])

    # or return [result_full, full_df_2]
    return result_full




@st.cache_data(experimental_allow_widgets=True, show_spinner=False)
def calculate_time_cont_migr_new(data: pd.DataFrame, agency: str, start_date: str, end_dates: str, step: dict, col_ogrn: str, col_date: str, col_rating: str):
    result_num = {}
    result_non_num = {}
    agency_dict = {}
    if agency == 'Expert RA':
        agency_dict = expert_test
    if agency == 'NCR':
        agency_dict = NCR_test
    if agency == 'AKRA':
        agency_dict = akra
    if agency == 'S&P Global Ratings':
        agency_dict = s_and_p
    if agency == 'Fitch Ratings':
        agency_dict = fitch
    if agency == "Moody's Interfax Rating Agency":
        agency_dict = moodys
    if agency == 'NRA':
        agency_dict = nra

    full_df = get_nan_df(agency_dict)
    counter_def = 0

    full_step = None
    curent_step = None
    time_counter = None
    step_num = None
    step_ = None
    st.write(step)

    if 'months' in step.keys():
        step_num = step['months']
        full_step = relativedelta(months=1)
        curent_step = relativedelta(months=step['months'])
        step_ = 31

    if 'years' in step.keys():
        step_num = step['years']
        full_step = relativedelta(years=1)
        curent_step = relativedelta(years=step['years'])
        step_ = 364

    if 'days' in step.keys():
        step_num = step['days']
        full_step = relativedelta(days=1)
        curent_step = relativedelta(days=step_num)
        step_ = 1

    data_1 = data

    data_1 = data_1.sort_values(col_date)
    counter = 0
    set_ogrn = (data_1[col_ogrn].unique())
    progress_text = "Calculate time cont. matrix. Please wait."
    my_bar_1 = st.progress(0, text=progress_text)

    full_df = get_nan_df(agency_dict)
    result_trans = {prev: {cur: 0 for cur in full_df.columns} for prev in full_df.columns}
    result_time = {cur: 0.00001 for cur in full_df.columns}  # Время в состоянии перед переходом

    counter_CC_D = 0
    checker_ccc = []
    for ogrn in set_ogrn:  # todo iterate over full df
        result_migr = defaultdict(list)
        result_migr_time = defaultdict(list)

        if pd.isna(ogrn) != True:  # todo if ogrn in not None
            time.sleep(0.01)
            my_bar_1.progress(int(100 * counter / len(set_ogrn)) , text=progress_text)
            counter += 1
            pr = data_1[data_1[col_ogrn] == ogrn].reset_index().drop(columns=['index']).sort_values(col_date)
            start_dates = pr[col_date].iloc[0]
            end_dates = f"{pd.to_datetime(pr[col_date][len(pr) - 1]).year + 1}-01-01"

            while start_dates < (datetime.strptime(end_dates, "%Y-%m-%d")).strftime('%Y-%m-%d'):
                data_prev = pr[(pr[col_date] < start_dates)]
                end_date = (datetime.strptime(start_dates, "%Y-%m-%d") + curent_step).strftime('%Y-%m-%d')
                temp_df = pr[(pr[col_date] >= start_dates) & (pr[col_date] <= end_date)].reset_index().drop(columns=['index']).sort_values(col_date)
                first = ''
                date_start = ''
                # last = ''
                # temp_df = data_1[(data_1['ogrn'] == ogrn)].reset_index().drop(columns=['index']).sort_values('_date')  # todo get df by ogrn
                ind_to_drop_otozv = []
                if len(temp_df) > 0:
                    if temp_df[col_date][0] == start_dates:
                        last_ , date_ = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn,col_date, col_rating)
                        first_ = temp_df[col_rating][0]
                        if first_ == last_ and last_ is not None:
                            first = first_

                        elif date_ is None and last_ is None:
                            first = first_

                        else:
                            first = last_

                        date_start = start_dates
                    else:
                        if len(get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)) > 0:
                            first, date_start = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)  # todo get previous rating on date = start_date
                else:
                    if len(get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)) > 0:
                        first, date_start = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)  # todo get previous rating on date = start_date

                if len(temp_df) > 0:

                    prev_rating = first
                    prev_year = pd.to_datetime(date_start).year
                    prev_date = date_start

                    for i in range(len(temp_df)):  # Проходим по записям начиная со второго элемента
                        cur_rating = temp_df[col_rating].iloc[i]
                        cur_year = pd.to_datetime(temp_df[col_date].iloc[i]).year
                            # print(temp_df)
                        if pd.to_datetime(temp_df[col_date].iloc[i]) != pd.to_datetime(prev_date):
                            if (pd.to_datetime(temp_df[col_date].iloc[i]) - pd.to_datetime(prev_date)).days / 364 > 1:
                                continue  # Пропускаем большие разрывы во времени
                            else:
                                if prev_rating == "CCC-C":
                                    print(ogrn, prev_rating)
                                if i != len(temp_df) - 1:
                                    if prev_rating != cur_rating:
                                        result_migr[prev_rating].append(cur_rating)

                                    result_migr_time[prev_rating].append((pd.to_datetime(temp_df[col_date].iloc[i]) - pd.to_datetime(prev_date)).days / step_)
                                    prev_date = temp_df[col_date].iloc[i]
                                    prev_rating = cur_rating  # Обновляем "предыдущее" значениеprev_rating = cur_rating  # Обновляем "предыдущее" значение
                                else:
                                    if prev_rating != cur_rating:
                                        result_migr[prev_rating].append(cur_rating)

                                    result_migr_time[prev_rating].append((pd.to_datetime(temp_df[col_date].iloc[i]) - pd.to_datetime(prev_date)).days / step_)
                                    prev_date = temp_df[col_date].iloc[len(temp_df) - 1]
                                    cur_rating = temp_df[col_rating].iloc[len(temp_df) - 1]
                                    # prev_rating = cur_rating
                                    result_migr_time[cur_rating].append((pd.to_datetime(end_date) - pd.to_datetime(prev_date)).days / step_)

                            prev_year = cur_year

                else:
                    if first in agency_dict:
                        result_migr_time[first].append((pd.to_datetime(end_date) - pd.to_datetime(start_dates)).days / step_)

                start_dates = (datetime.strptime(start_dates, "%Y-%m-%d") + full_step).strftime('%Y-%m-%d')

            for prev_rating, transitions in result_migr.items():
                    for cur_rating in transitions:
                        result_trans[prev_rating][cur_rating] += 1
            
            for prev_rating, time_ in result_migr_time.items():
                result_time[prev_rating] += sum(time_)

    result_full = {}
    # st.write(result_trans)
    # st.write(result_time)
    for key, val in result_trans.items():
        if key not in result_full:
            result_full[key] = {}
        if key in result_time.keys():
            for k, v in val.items():
                if k not in result_full[key]:
                        result_full[key][k] = (result_trans[key][k]) / (result_time[key])

    time.sleep(1)
    my_bar_1.empty()
    return result_full


# TODO check scale
def time_cont(data: pd.DataFrame, agency: str, start_date: str, end_dates: str, step: dict, directory, type_ogrn, type_date, type_rating):
    st.title('Markov process with continous time')
    delta = datetime.strptime(end_dates, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")
    agency_dict = {}
    # res_avar = {}
    pie_cont = {}
    if agency == 'Expert RA':
        agency_dict = expert_test
    if agency == 'NCR':
        agency_dict = NCR_test
    if agency == 'AKRA':
        agency_dict = akra
    if agency == 'S&P Global Ratings':
        agency_dict = s_and_p
    if agency == 'Fitch Ratings':
        agency_dict = fitch
    if agency == "Moody's Interfax Rating Agency":
        agency_dict = moodys
    if agency == 'NRA':
        agency_dict = nra
    full_df_2 = get_nan_df(agency_dict)

    # Second attampt to find the best avarage of final matrix throw the entire period
    start_time = time.perf_counter()
    result_full = calculate_time_cont_migr_new(data, agency, start_date, end_dates, step, type_ogrn, type_date, type_rating)
    # result_full = calculate_time_cont_migr(data, agency, start_date, end_dates, step, type_ogrn, type_date, type_rating)
    result_full = fill_empty(result_full, agency_dict)
    result_full_df = pd.DataFrame().from_dict(result_full).fillna(0).reset_index()
    result_full_df = get_generator(sort_df(result_full_df, agency_dict))
    result = expm(result_full_df.to_numpy())
    columns_ag = list(agency_dict.keys())
    result = pd.DataFrame(result, columns=columns_ag, index=columns_ag)
    end_time = time.perf_counter()

    st.write(f"Время выполнения: {end_time - start_time:.4f} секунд и размер датасета: {len(data)}")

    check_1 = pd.DataFrame()
    name = ''
    fig = plt.figure(figsize=(15, 15))
    plot = sns.heatmap(result, annot=True, fmt='.3f', linewidths=.5, annot_kws={"size":10})
    # plt.savefig(f'{directory}/time_cont_step={step}_second_avar.jpg')
    plt.close()
    # result.to_excel(f'{directory}/time_cont_step={step}_second_avar.xlsx')
    for col in result.columns:  # TODO redact by index
        if result[col].sum() > 0:
            fig_ = go.Figure(data=[go.Pie(labels=result.index[result[col] > 0].tolist(),
                                          values=result[col][result[col] > 0])])
            fig_.update_layout(
                legend_title=f"{col} rating moved to:",
                font=dict(
                    family="Courier New, monospace",
                    size=15,
                    color="White"
                )
            )
            pie_cont[col] = fig_
    if st.checkbox('Display Pie chart of migration time cont. matrix', key='time_cont_pie'):
        pies = st.sidebar.multiselect('Choose ratings to see moves', pie_cont)
        for pie_diag in pies:
            st.plotly_chart(pie_cont[pie_diag], theme='streamlit')

    if st.checkbox('Display graph of migration time cont. matrix', key='time_cont_graph'):
        HtmlFile = open(graph_matric_migration(result), 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height=650, width=650)

    if st.checkbox('Display static graph of migration time_cont matrix'):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(graph_matric_migration_stat(result))

    if st.checkbox('Display migration time cont.matrix', key='time_cont_migr'):
        # result.to_excel(f'cont_time_step={step}.xlsx')
        st.pyplot(fig)
        fig.savefig((f'results/{agency}/images/time_cont_step={step}_{datetime.now().strftime('%Y-%m-%d')}.jpg'))

    if st.checkbox('Get predict of time cont. matrix', key='time_cont_predict'):
        n = st.number_input('Enter number', max_value=1000, min_value=0, key='time_cont_get_pred')
        # check_1 = np.linalg.matrix_power(result, n)
        check_1 = expm(result_full_df.to_numpy() * 0.083)
        check_1 = pd.DataFrame(check_1, columns=columns_ag, index=columns_ag)
        st.write(pd.DataFrame(check_1, columns=list(agency_dict.keys()), index=list(agency_dict.keys())))
        name = f'results/{agency}/cont_time/predict_time_cont_step={step}_predict={n}_{datetime.now().strftime('%Y-%m-%d')}.xlsx'
        check_1.to_excel(name)
        f = Fitter(check_1,
                   distributions=['cauchy', 'chi2', 'expon', 'exponpow', 'gamma', 'lognorm', 'norm', 'powerlaw',
                                  'rayleigh', 'uniform', 'beta'])
        f.fit()
        distr_fit = f.summary()
        st.write(distr_fit)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(f.plot_pdf())
        st.pyplot(f.plot_pdf())

    if st.checkbox('Display predict time_cont. migration matrix', key='time_cont'):
        fig = plt.figure(figsize=(10, 10))
        n = st.number_input('Enter number', max_value=1000, min_value=2, key='time_cont_value')
        check_1 = np.linalg.matrix_power(result, n)
        plot = sns.heatmap(check_1, annot=True, fmt='.3f', linewidths=.5, annot_kws={"size":9})
        st.pyplot(fig)
        fig.savefig((f'results/{agency}/images/predict_{n}_time_cont_step={step}_{datetime.now().strftime('%Y-%m-%d')}.jpg'))
        # plt.savefig(f'{directory}/time_cont_step={step}_second_avar.jpg')
        plt.close()

    if st.checkbox('Display distribution of time cont. matrix', key='time_cont_dist'):
        f = Fitter(result,
                   distributions=['cauchy', 'chi2', 'expon', 'exponpow', 'gamma', 'lognorm', 'norm', 'powerlaw',
                                  'rayleigh', 'uniform', 'beta'])
        f.fit()
        distr_fit = f.summary()
        st.write(distr_fit)
        st.pyplot(f.plot_pdf())
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(f.plot_pdf())

    if st.button('Download time cont. matrix', key='donwload'):
        result.to_excel(f'results/{agency}/cont_time/time_cont_step={step}_{datetime.now().strftime('%Y-%m-%d')}.xlsx')

#TODO try to realize Bayesian method to calculate migration matrix

def get_first_bayes(agency_dict: dict):
    import random
    Bayes = get_nan_df(agency_dict)
    Check = pd.DataFrame(np.random.dirichlet(np.ones(len(Bayes.index)),size=len(Bayes.columns)), columns=Bayes.columns, index=Bayes.index)
    return Check
def get_bayesian_matrix(Bayes: pd.DataFrame, current_matrix: pd.DataFrame):
    # st.write(current_matrix.T['AAA.ru']['AA.ru'])
    sum_matrix = Bayes.copy()
    Bays_2 = Bayes.copy()
    for ind in Bayes.index:
        sum_a = 0.0
        for i, col in enumerate(Bayes.columns):
            sum_b = 0.0
            column_null = []
            for j, col_ in enumerate(Bayes.columns):
                if col != col_:
                    if current_matrix.loc[ind].at[col_] != 0.0:
                        sum_b += (1.001 - Bayes.loc[ind].at[col_]) * current_matrix.loc[ind].at[col_]
                    else:
                        sum_b += (1.001 - Bayes.loc[ind].at[col_]) * 0.001
                else:
                    if current_matrix.loc[ind].at[col_] != 0.0:
                        sum_b += (Bayes.loc[ind].at[col_]) * current_matrix.loc[ind].at[col_]
                    else:
                        sum_b += (Bayes.loc[ind].at[col_]) * 0.001
            # st.write(ind, col, Bayes.loc[ind].at[col], current_matrix.loc[ind].at[col], sum_b)
            # if Bayes.loc[ind].at[col] * current_matrix.loc[ind].at[col] == 0:
                # Bayes.at[ind, col] = Bayes.loc[ind].at[col] * 0.001 / sum_b
            # Bayes.at[ind, col] = Bayes.loc[ind].at[col] * current_matrix.loc[ind].at[col] / sum_b
            sum_matrix.at[ind, col] = sum_b
        for col in Bayes.columns:
            Bays_2.at[ind, col] = (current_matrix.loc[ind].at[col]) * Bayes.loc[ind].at[col] / sum_matrix.loc[ind].at[col]
            # st.write(ind, col, current_matrix.loc[ind].at[col], Bayes.loc[ind].at[col], sum_matrix.loc[ind].at[col])
            # if current_matrix.loc[ind].at[col] != 0.0 and Bayes.loc[ind].at[col] != 0.0:
            #     # st.write(sum_a,  current_matrix.loc[ind].at[col], Bayes.loc[ind].at[col])
            #     Bays_2.at[ind, col] = (current_matrix.loc[ind].at[col]) * Bayes.loc[ind].at[col] / sum_matrix.loc[ind].at[col]
            # if current_matrix.loc[ind].at[col] == 0.0 and Bayes.loc[ind].at[col] != 0.0:
            #     Bays_2.at[ind, col] = (0.001) * Bayes.loc[ind].at[col] / sum_matrix.loc[ind].at[col]
            # if current_matrix.loc[ind].at[col] != 0.0 and Bayes.loc[ind].at[col] == 0.0:
            #     Bays_2.at[ind, col] = (0.001) * current_matrix.loc[ind].at[col] / sum_matrix.loc[ind].at[col]
            # if current_matrix.loc[ind].at[col] == 0.0 and Bayes.loc[ind].at[col] == 0.0:
            #     Bays_2.at[ind, col] = 0.001
    #

    # Bays_2 = Bayes.copy().round(3)
    for ind in Bays_2.index:
        for col in Bays_2.columns:
            if Bays_2.loc[ind].at[col] == 0.:
                Bays_2.at[ind, col] = 0.

    sum_mat = Bays_2.sum(axis=1)
    for ind in Bays_2.index:
        sum_ = sum_mat[ind]
        if sum_ < 1.0:
            column_checker = []
            for col in Bays_2.columns:
                if Bays_2.loc[ind].at[col] != 0.:
                    column_checker.append(Bays_2.loc[ind].at[col])
            for col in Bays_2.columns:
                # if Bays_2.loc[ind].at[col] == 0.:
                    # st.write(ind, col, Bays_2.loc[ind].at[col] )
                    Bays_2.at[ind, col] = Bays_2.loc[ind].at[col] + (1 - sum(column_checker))/len(Bays_2.columns)
    return Bays_2
def get_bayesian_matrix_prev_2(Bayes: pd.DataFrame, current_matrix: pd.DataFrame):
    # st.write(current_matrix.T['AAA.ru']['AA.ru'])
    sum_matrix = Bayes.copy()
    Bays_2 = Bayes.copy()
    for ind in Bayes.index:
        sum_a = 0.0
        for i, col in enumerate(Bayes.columns):
            sum_b = 0.0
            column_null = []
            for j, col_ in enumerate(Bayes.columns):
                if col != col_:
                    if Bayes.loc[ind].at[col_] != 0.0:
                        if current_matrix.loc[ind].at[col_] != 0.:
                            sum_b += (current_matrix.loc[ind].at[col_]) * (1.001 - Bayes.loc[ind].at[col_])
                        else:
                            sum_b += (0.001) * (1.001 - Bayes.loc[ind].at[col_])
                    else:
                        if current_matrix.loc[ind].at[col_] != 0.:
                            sum_b += (current_matrix.loc[ind].at[col_]) * (1 - 0.001)
                        else:
                            sum_b += (0.001) * (1 - 0.001)
                else:
                        if Bayes.loc[ind].at[col_] == 0.0:
                            if current_matrix.loc[ind].at[col_] != 0.:
                                sum_b += (current_matrix.loc[ind].at[col_]) * (0.001)
                            else:
                                sum_b += (0.001) * (0.001)
                        else:
                            if current_matrix.loc[ind].at[col_] != 0.:
                                sum_b += current_matrix.loc[ind].at[col_] * (Bayes.loc[ind].at[col_])
                            else:
                                sum_b += (0.001) * (Bayes.loc[ind].at[col_])

            sum_matrix.at[ind, col] = sum_b

        for col in Bays_2.columns:
            # Bays_2.at[ind, col] = (current_matrix.loc[ind].at[col]) * Bayes.loc[ind].at[col] / sum_matrix.loc[ind].at[col]
            # st.write(ind, col, current_matrix.loc[ind].at[col], Bayes.loc[ind].at[col], sum_matrix.loc[ind].at[col])
            if current_matrix.loc[ind].at[col] != 0.0 and Bayes.loc[ind].at[col] != 0.0:
                # st.write(sum_a,  current_matrix.loc[ind].at[col], Bayes.loc[ind].at[col])
                Bays_2.at[ind, col] = (current_matrix.loc[ind].at[col]) * Bayes.loc[ind].at[col] / sum_matrix.loc[ind].at[col]
            if current_matrix.loc[ind].at[col] == 0.0 and Bayes.loc[ind].at[col] != 0.0:
                Bays_2.at[ind, col] = current_matrix.loc[ind].at[col] * Bayes.loc[ind].at[col] / sum_matrix.loc[ind].at[col]
            if current_matrix.loc[ind].at[col] != 0.0 and Bayes.loc[ind].at[col] == 0.0:
                Bays_2.at[ind, col] = 0.001 * current_matrix.loc[ind].at[col] / sum_matrix.loc[ind].at[col]
            # if current_matrix.loc[ind].at[col] == 0.0 and Bayes.loc[ind].at[col] == 0.0:
            #     Bays_2.at[ind, col] = 0.001
    #
    sum_mat = Bays_2.sum(axis=1)
    for ind in Bays_2.index:
        sum_ = sum_mat[ind]
        if sum_ != 1.0:
            column_checker = []
            for col in Bays_2.columns:
                if Bays_2.loc[ind].at[col] != 0.:
                    column_checker.append(Bays_2.loc[ind].at[col])
            for col in Bays_2.columns:
                if Bays_2.loc[ind].at[col] == 0.:
                    Bays_2.at[ind, col] = (1 - sum(column_checker))/(len(Bays_2.columns) - len(column_checker))
    return Bays_2

def get_bayesian_matrix_prev(Bayes: pd.DataFrame, current_matrix: pd.DataFrame):
    # st.write(current_matrix.T['AAA.ru']['AA.ru'])
    # st.write('Bayesian', Bayes, 'Current', current_matrix)
    sum_matrix = Bayes.copy()
    Bays_2 = Bayes.copy()
    for ind in Bayes.index:
        sum_a = 0.0
        for i, col in enumerate(Bayes.columns):
            if current_matrix.loc[ind].at[col] != 0.0 and Bayes.loc[ind].at[col] != 0.0:
                if current_matrix.loc[ind].at[col] == 1:
                    current_matrix.loc[ind].at[col] = 0.98
                sum_a += Bayes.loc[ind].at[col] * current_matrix.loc[ind].at[col]
            elif current_matrix.loc[ind].at[col] == 0.0 and Bayes.loc[ind].at[col] != 0.0:
                sum_a += Bayes.loc[ind].at[col] * Bayes.loc[ind].at[col]
            elif current_matrix.loc[ind].at[col] != 0.0 and Bayes.loc[ind].at[col] == 0.0:
                sum_a += current_matrix.loc[ind].at[col] * current_matrix.loc[ind].at[col]

        for col in Bays_2.columns:
            # st.write(ind, col, current_matrix.loc[ind].at[col], Bayes.loc[ind].at[col], sum_matrix.loc[ind].at[col])
            if current_matrix.loc[ind].at[col] != 0.0 and Bayes.loc[ind].at[col] != 0.0:
                # st.write(sum_a,  current_matrix.loc[ind].at[col], Bayes.loc[ind].at[col])
                Bays_2.at[ind, col] = (current_matrix.loc[ind].at[col]) * Bayes.loc[ind].at[col] / sum_a
            if current_matrix.loc[ind].at[col] == 0.0 and Bayes.loc[ind].at[col] != 0.0:
                Bays_2.at[ind, col] = (Bayes.loc[ind].at[col]) * Bayes.loc[ind].at[col] / sum_a
            if current_matrix.loc[ind].at[col] != 0.0 and Bayes.loc[ind].at[col] == 0.0:
                Bays_2.at[ind, col] = (current_matrix.loc[ind].at[col]) * current_matrix.loc[ind].at[col] / sum_a
            if current_matrix.loc[ind].at[col] == 0.0 and Bayes.loc[ind].at[col] == 0.0:
                Bays_2.at[ind, col] = 0.00001

    sum_mat = Bayes.sum(axis=1)
    for ind in Bayes.index:
        sum_ = sum_mat[ind]
        if sum_ == 0.0:
            Bayes.at[ind, ind] = 1.
    return Bays_2.fillna(0)

@st.cache_data(experimental_allow_widgets=True, show_spinner=False)
def bayesian_metric(data: pd.DataFrame, agency: str, start_date: str, end_dates: str, step: dict, col_date: str, col_ogrn:str, col_rating:str):
    agency_dict = {}
    if agency == 'Expert RA':
        agency_dict = expert_test
    if agency == 'NCR':
        agency_dict = NCR_test
    if agency == 'AKRA':
        agency_dict = akra
    if agency == 'S&P Global Ratings':
        agency_dict = s_and_p
    if agency == 'Fitch Ratings':
        agency_dict = fitch
    if agency == "Moody's Interfax Rating Agency":
        agency_dict = moodys
    if agency == 'NRA':
        agency_dict = nra

    Bayesint_trans_matrix = get_first_bayes(agency_dict)
    #initial conditions uniform distribution of each rating
    # for col in Bayesint_trans_matrix.columns:
    #     for ind in Bayesint_trans_matrix.index:
    #         Bayesint_trans_matrix.at[ind, col] = 1 / (len(Bayesint_trans_matrix.columns))

    st.write(Bayesint_trans_matrix)
    check_ = get_nan_df(agency_dict)
    full_step = None
    curent_step = None
    time_counter = None
    step_num = None

    if 'months' in step.keys():
        step_num = step['months']
        full_step = relativedelta(months=1)
        curent_step = relativedelta(months=step['months'])
        time_counter = 31

    if 'years' in step.keys():
        step_num = step['years']
        full_step = relativedelta(years=1)
        curent_step = relativedelta(years=step['years'])
        time_counter = 365

    if 'days' in step.keys():
        step_num = step['days']
        full_step = relativedelta(days=1)
        curent_step = relativedelta(days=step_num)
        time_counter = 1

    data_prev = data[(data['agency'] == agency) & (data[col_date] < start_date)]
    data_1 = pd.DataFrame()
    if len(scale) != 0:
        for scal in scale:
            data_1 = pd.concat([data_1, data[
                (data['agency'] == agency) & (data['scale'] == scal)].reset_index().drop(columns=['index'])], ignore_index=True)
    else:
        data_1 = pd.concat([data_1, data[(data['agency'] == agency)].reset_index().drop(columns=['index'])], ignore_index=True)

    data_1 = data_1.sort_values(col_date)
    set_ogrn = (data_1[col_ogrn].unique())
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    counter = 0

    start_dates = start_date
    prog = datetime.strptime(end_dates, "%Y-%m-%d").year - datetime.strptime(start_date, "%Y-%m-%d").year
    result = {}
    result_migr = {}
    while start_dates < (datetime.strptime(end_dates, "%Y-%m-%d")).strftime('%Y-%m-%d'):
        time.sleep(0.01)
        # my_bar.progress(int(100 * counter / prog), text=progress_text)
        for ogrn in set_ogrn:
            if pd.isna(ogrn) != True:
                # result = {}
                # result_migr = {}
                pr = data_1.loc[data_1[col_ogrn] == ogrn].reset_index().drop(columns=['index']).sort_values(col_date)
                # start_dates = pr[col_date][0]
                # end_dates = pr[col_date][len(pr) - 1]
                data_prev = pr[(pr[col_date] < start_dates)]
                end_date = (datetime.strptime(start_dates, "%Y-%m-%d") + curent_step).strftime('%Y-%m-%d')
                temp_df = pr.loc[(pr[col_date] >= start_dates) & (pr[col_date] <= end_date)].reset_index().drop(columns=['index']).sort_values(col_date)

                # st.write(start_date, end_date)
                first = ''
                first_date = ''
                last = ''
                if len(temp_df) > 0:
                    # first = temp_df[col_rating][0]
                    if temp_df[col_date][0] == start_dates:
                        first = temp_df[col_rating][0]
                    else:
                        if len(get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)) > 0:
                            first, first_date = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)  # todo get previous rating on date = start_date
                            # first_date = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)[1]


                    temp_local_df = temp_df.loc[temp_df[col_date] == temp_df[col_date][len(temp_df) - 1]]
                    if len(temp_local_df) > 1 and 'отозван' in temp_local_df[col_rating].values:
                        last = 'отозван'
                    else:
                        last = temp_df[col_rating][len(temp_df) - 1]

                    if last == 'отозван':
                        temp_local_df = temp_df.loc[temp_df[col_date] == temp_df[col_date][len(temp_df) - 1]]
                        if len(temp_local_df) > 1 and default[agency] in temp_local_df[col_rating].values:
                            last = default[agency]

                else:
                    if len(get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)) > 0:
                        first, first_date = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)  # todo get previous rating on date = start_date
                        # first_date = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)[1]
                        last = first

                #
                if first_date != '':
                    years = (datetime.strptime(start_dates, "%Y-%m-%d") - datetime.strptime(first_date,
                                                                                            "%Y-%m-%d")).days // 365
                    if years > 1:
                        # st.write(first, ogrn, start_dates, end_date)
                        # st.write(temp_df['_name'][0], prev_, start_date, first)
                        first = ''

                if first == default[agency]:
                    st.write(ogrn, first, last, start_dates, end_date)
                if first != '' and first != default[agency]:
                    # if first != default[agency] and last != 'отозван':
                        if first in agency_dict and last in agency_dict:
                            if first not in result_migr:
                                result_migr[first] = []
                            result_migr[first].append(last)

        for key, value in agency_dict.items():
            if key not in result:
                result[key] = {}
            if key in result_migr.keys():
                if key not in result:
                    result[key] = {}
                value_1 = result_migr[key]
                temp_dict = {}
                for i in range(len(set(value_1))):
                    temp_dict[list(Counter(value_1).keys())[i]] = list(Counter(value_1).values())[i]
                result[key] = temp_dict

        result_df = pd.DataFrame().from_dict(result).fillna(0).reset_index()
        result_df = sort_df(result_df, agency_dict)
        check_ = result_df
        # for ind in result_df.index:
        #     for col in result_df.columns:
        #         check_.at[ind, col] = result_df[ind][col]
        # st.write(full_df)
        # st.write(sum_mat)
        sum_mat = check_.sum(axis=1)
        for ind in check_.index:
            sum_ = sum_mat[ind]
            if sum_ == 0.0:
                check_.at[ind, ind] = 1.
            for col in check_.columns:
                if sum_ > 0:
                    check_.at[ind, col] /= round(sum_, 6)
                    # check_.at[ind, col] = result_df[ind][col] / round(sum_,3)

        # st.write('Bayesian', Bayesint_trans_matrix, 'Current', check_)
        for _ in range(20):
            # st.write('Bayes', Bayesint_trans_matrix, 'Curre', check_)
            Bayesint_trans_matrix = get_bayesian_matrix(Bayesint_trans_matrix, check_)
            # st.write('Bayesian', Bayesint_trans_matrix, 'Current', check_)
        counter += 1
        st.write(start_dates, end_date)
        # st.write('Bayesian', Bayesint_trans_matrix, 'Current', check_)
        start_dates = (datetime.strptime(start_dates, "%Y-%m-%d") + full_step).strftime('%Y-%m-%d')

    st.write(Bayesint_trans_matrix)
    time.sleep(1)
    my_bar.empty()
    return [Bayesint_trans_matrix, check_]

def Bayes_migration(data: pd.DataFrame, agency: str, start_date: str, end_dates: str,
                     step: dict, type_ogrn, type_date, type_rating):
    st.title('Markov process with discrete time')
    delta = datetime.strptime(end_dates, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")
    agency_dict = {}
    if agency == 'Expert RA':
        agency_dict = expert_test
    if agency == 'NCR':
        agency_dict = NCR_test
    if agency == 'AKRA':
        agency_dict = akra
    if agency == 'S&P Global Ratings':
        agency_dict = s_and_p
    if agency == 'Fitch Ratings':
        agency_dict = fitch
    if agency == "Moody's Interfax Rating Agency":
        agency_dict = moodys
    if agency == 'NRA':
        agency_dict = nra

    check_dict = {}
    pie_cont = {}
    check_ = {}
    full_df, current_df = bayesian_metric(data, agency, start_date, end_dates, step, type_date, type_ogrn, type_rating)
    # full_df.index = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B',
    #                  'B-', 'CCC', 'CC', 'C', 'D']
    # full_df.columns = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B',
    #                    'B-', 'CCC', 'CC', 'C', 'D']
    # full_df.to_excel(f'{directory}/discrete_markov_step={step}.xlsx')
    n = 0
    name = ''
    fig = plt.figure(figsize=(10, 10))
    plot = sns.heatmap(full_df, annot=True, fmt='.3f', linewidths=.5, annot_kws={"size":11})
    # plt.savefig(f'{directory}/discrete_markov_step={step}.jpg')
    plt.close()

    for col in full_df.columns:  # TODO redact by index
        if full_df[col].sum() > 0:
            fig_ = go.Figure(data=[go.Pie(labels=full_df.index[full_df[col] > 0].tolist(),
                                          values=full_df[col][full_df[col] > 0])])
            fig_.update_layout(
                legend_title=f"{col} rating moved to:",
                font=dict(
                    family="Courier New, monospace",
                    size=15,
                    color="White"
                )
            )
            pie_cont[col] = fig_

    if st.checkbox('Display Pie chart of Bayesian migration discrete matrix'):
        pies = st.sidebar.multiselect('Choose ratings to see moves', pie_cont)
        for pie_diag in pies:
            st.plotly_chart(pie_cont[pie_diag], theme='streamlit')

    if st.checkbox('Display graph of Bayesian discrete matrix'):
        HtmlFile = open(graph_matric_migration(full_df), 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height=650, width=650)

    if st.checkbox('Display static graph of migration Bayes matrix'):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(graph_matric_migration_stat(full_df))

    if st.checkbox('Display Bayesian migration discrete matrix'):
        # full_df.to_excel(f'discrete_time_step={step}.xlsx')
        st.pyplot(fig)
        fig.savefig((f'results/{agency}/images/Bayesian_matrix={step}_{datetime.now().strftime('%Y-%m-%d')}.jpg'))

    if st.checkbox('Display predict Bayesian. migration matrix'):
        fig = plt.figure(figsize=(10, 10))
        n = st.number_input('Enter number', max_value=1000, min_value=2, key='Bayes_disp')
        check_1 = np.linalg.matrix_power(full_df, n)
        plot = sns.heatmap(check_1, annot=True, fmt='.3f', linewidths=.5, annot_kws={"size":11 })
        # plt.savefig(f'{directory}/time_cont_step={step}_second_avar.jpg')
        st.pyplot(fig)
        fig.savefig((f'results/{agency}/images/predict_{n}_Bayesian_matrix={step}_{datetime.now().strftime('%Y-%m-%d')}.jpg'))
        plt.close()

    if st.checkbox('Get predict of Bayesian migration discrete matrix'):
        # get_state_by_time(data, agency, start_date, scale)
        n = st.number_input('Enter number', max_value=100, min_value=2, key='Bayes')
        check_1 = np.linalg.matrix_power(full_df, n)
        st.write(pd.DataFrame(check_1, columns=list(agency_dict.keys()), index=list(agency_dict.keys())))
        check_1 = pd.DataFrame(check_1, columns=list(agency_dict.keys()), index=list(agency_dict.keys()))
        name = f"results/{agency}/Bayes/predict_Bayes_step={step}_predict={n}_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
        check_1.to_excel(name)
        f = Fitter(check_1,
                   distributions=['cauchy', 'chi2', 'expon', 'exponpow', 'gamma', 'lognorm', 'norm', 'powerlaw',
                                  'rayleigh', 'uniform', 'beta'])
        f.fit()
        distr_fit = f.summary()
        st.write(distr_fit)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(f.plot_pdf())
        st.pyplot(f.plot_pdf())

    if st.checkbox('Display distribution of Bayesian migration discrete matrix '):
        # full_df = get_state_by_time(data, agency, date_to_check, step, scale)[0]
        f = Fitter(full_df,
                   distributions=['cauchy', 'chi2', 'expon', 'exponpow', 'gamma', 'lognorm', 'norm', 'powerlaw',
                                  'rayleigh', 'uniform', 'beta'])
        f.fit()
        distr_fit = f.summary()
        st.write(distr_fit)
        st.pyplot(f.plot_pdf())
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(f.plot_pdf())

    if st.button('Download Bayess'):
        full_df.to_excel(f'results/{agency}/Bayes/Bayes_step_{datetime.now().strftime('%Y-%m-%d')}.xlsx')
    if st.button('Download Current'):
        current_df.to_excel(f'results/{agency}/Bayes/current_step=Х{step}_{datetime.now().strftime('%Y-%m-%d')}.xlsx')

def aalen_johansen_metric(data: pd.DataFrame, agency: str, start_date: str, end_dates: str, step: int):
    delta = datetime.strptime(end_dates, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")
    agency_dict = {}
    result_num = {}
    result_non_num = {}
    res_avar = {}
    count_obs = {}
    full_df = pd.DataFrame()
    if agency == 'Expert RA':
        agency_dict = expert_test
    full_df_2 = get_nan_df(agency_dict)

    while start_date <= (datetime.strptime(end_dates, "%Y-%m-%d") + relativedelta(years=1)).strftime('%Y-%m-%d'):
        result = {}
        result_state_in = {}
        result_check_full = {}
        check_ogrn = []
        end_date = (datetime.strptime(start_date, "%Y-%m-%d") + relativedelta(years=step)).strftime('%Y-%m-%d')
        data_prev = data[(data['agency'] == agency) & (data['_date'] <= start_date)]
        data_1 = pd.DataFrame()
        if len(scale) != 0:
            for scal in scale:
                data_1 = pd.concat([data_1, data[
                    (data['agency'] == agency) & (data['_date'] >= start_date) & (data['_date'] <= end_date) & (
                            data['scale'] == scal)].reset_index().drop(columns=['index'])], ignore_index=True)
        else:
            data_1 = pd.concat([data_1, data[
                (data['agency'] == agency) & (data['_date'] >= start_date) & (
                        data['_date'] <= end_date)].reset_index().drop(columns=['index'])], ignore_index=True)

        for ind in data_1.index:  # todo iterate over full df
            if pd.isnull(data_1['ogrn'][ind]) != True:  # todo if ogrn in not None
                # start_date + step
                temp_df = data_1[(data_1['ogrn'] == data_1['ogrn'][ind])].reset_index().drop(
                    columns=['index']).sort_values('_date')  # todo get df by ogrn
                ind_to_drop_otozv = []
                if len(temp_df) > 0:
                    if data_1['ogrn'][ind] not in check_ogrn:  # todo check ogrn to not dubl
                        container_state_in = []
                        check_ogrn.append(data_1['ogrn'][ind])  # todo save the all ogrn which were found
                        first = ''  # todo if rating on previous date wasn't exist
                        date_start = start_date
                        if len(get_prev_date_raitng(data_prev, data_1['ogrn'][ind], start_date)) > 0:
                            first = get_prev_date_raitng(data_prev, data_1['ogrn'][ind], start_date)[0]  # todo get previous rating on date = start_date
                            date_start = get_prev_date_raitng(data_prev, data_1['ogrn'][ind], start_date)[1]
                            if date_start <= start_date:
                                date_start = start_date
                        else:
                            if temp_df['_date'][0] == start_date:  # todo if rating exist on date = start_date , then use it
                                first = temp_df['rating'][0]
                        if first != '':
                            if first not in result_state_in:  # todo remove result_state_in from final version(do not forget)
                                result_state_in[first] = []

                            for indx in temp_df.index:  # todo iterate over exist temp_df with unique ogrn, because in previous step save all unique ogrn
                                if first in agency_dict and temp_df['rating'][indx] in agency_dict:
                                    if first == temp_df['rating'][indx]:  # todo save infromation about not moveble companies
                                        container_state_in.append(True)
                                    else:
                                        container_state_in.append(False)
                                        if len(temp_df[temp_df['_date'] == temp_df['_date'][indx]]) > 1 and 'отозван' in \
                                                temp_df[temp_df['_date'] == temp_df['_date'][indx]]['rating'].values:
                                            # result_move_to[first].update(
                                            #     {temp_df['ogrn'][indx]: {'отозван': num_days / delta_full}})
                                            df_otozv = temp_df[temp_df['_date'] == temp_df['_date'][indx]].reset_index()  # todo df to check if any отозван in temp_df by _date
                                            for k_indx in df_otozv.index:
                                                if df_otozv['rating'][k_indx] != 'отозван':
                                                    ind_to_drop_otozv.append(df_otozv['index'][k_indx])

                            if False not in container_state_in and temp_df['rating'][0] in agency_dict and first in agency_dict:
                                result_state_in[first].append(temp_df['rating'][0])  # todo upd each first state with new information
                            # todo if error with 0 index, reset index there in temp_df
                            if False in container_state_in and temp_df['rating'][0] in agency_dict and first in agency_dict:
                                temp_df = temp_df.drop(index=ind_to_drop_otozv).reset_index().drop(columns=['index'])
                                last_df = temp_df.loc[[0]]
                                last = temp_df['rating'][len(temp_df) - 1]
                                last_df.at[0, 'rating'] = last
                                last_df.at[0, '_date'] = end_date
                                first_df = temp_df.loc[[0]]
                                first_df.at[0, 'rating'] = first
                                first_df.at[0, '_date'] = date_start
                                temp_df = pd.concat([temp_df, first_df], axis=0).sort_values(
                                    '_date').reset_index().drop(columns=['index'])
                                test_rat = temp_df['rating'].values
                                test_date = temp_df['_date'].values
                                test_name = temp_df['_name'].values
                                # first_rat = test_rat[0]
                                # first_date = test_date[0]
                                for i, rat in enumerate(test_rat):
                                    if rat not in result_check_full:
                                        result_check_full[rat] = []
                                    if i + 1 >= len(test_rat):
                                        last_rat = rat
                                        last_dat = test_date[i]
                                        test_rat_rev = test_rat[::-1]
                                        test_date_rev = test_date[::-1]
                                        num = 0
                                        for j, rat_rev in enumerate(test_rat_rev):
                                            if rat_rev != last_rat:
                                                num = j - 1
                                                last_dat = test_date_rev[j]
                                                break
                                        result_check_full[test_rat_rev[num]].append(
                                            (last_dat, test_rat[len(test_rat) - 1],
                                             test_date_rev[num], end_date, rat))
                                    else:
                                        if test_rat[i] != test_rat[i + 1]:
                                            result_check_full[rat].append(
                                                (test_date[i], test_rat[i + 1], test_date[i + 1]))

        # st.write(count_moves(result_check_full))
        for key in agency_dict.keys():
            if key not in res_avar:
                res_avar[key] = []
            if key in result_state_in.values() or key in count_moves(result_check_full).keys():
                res_avar[key].append(1)

        for key_, value_ in agency.items():
            counter = 0
            if key_ not in result and key_ in agency_dict:
                result[key_] = {}
            if key_ not in count_obs:
                count_obs[key_] = []
            for k, v in value_.items():
                if k not in result[key_]:
                    result[key_][k] = []
                if key_ in result_state_in.keys():
                    counter = len(result_state_in[key_])
                # st.write(key_, k, count_moves(result_check_full)[key_][k], state_in, sum(count_time(result_check_full, start_date, end_date)[key_]), last_stat)
                result[key_][k] = count_moves(result_check_full)[key_][k]
                counter += count_moves(result_check_full)[key_][k]
            count_obs[key_].append(counter)

        for key_, val_ in agency_dict.items():
            if key_ not in result:
                result[key_] = {}
            for key, val in result.items():
                if key_ == key:
                    for k, v in val.items():
                        if k not in result[key_]:
                            result[key_][k] = 0.0

        result = fill_empty(result, agency)
        result_df = pd.DataFrame().from_dict(result).fillna(0).reset_index()
        result_df = sort_df(result_df, agency_dict)
        full_df_2 += result_df
        start_date = (datetime.strptime(start_date, "%Y-%m-%d") + relativedelta(years=1)).strftime('%Y-%m-%d')
        # TODO realize method of time-continous process + think about NR rating for first method     07_02_2024
    result_full = {}
    for key_, val_ in agency_dict.items():
        if key_ not in result_full:
            result_full[key_] = {}
        for key, val in result_full.items():
            if key_ == key:
                for k, v in val.items():
                    if k not in result_full[key_]:
                        result_full[key_][k] = 0.0

    st.write(full_df_2)
    for ind in full_df_2.index:
        if ind in count_obs.keys():
            sum_ = sum(count_obs[ind])
            for col in full_df_2.columns:
                if sum_ > 0:
                    full_df.at[ind, col] /= round(sum_, 3)

    full_df_2 = get_generator(full_df_2.fillna(0))
    result_2 = expm(full_df_2.fillna(0).to_numpy())
    result_2 = pd.DataFrame(result_2, columns=list(agency_dict.keys()), index=list(agency_dict.keys()))
    # result_2.to_excel('output_test_cont_36_m.xlsx')
    fig = plt.figure(figsize=(15, 15))
    plot = sns.heatmap(result_2, annot=True, fmt='.3f')
    st.write(fig)
    plt.savefig(f'aalen_johansen_{step}.jpg')
    plt.close()

def get_conf_int_wald(df, sum_ratings):
    z = 1.96

    # Инициализируем матрицу для хранения доверительных интервалов
    ci_lower = df.copy()
    ci_upper = df.copy()

    # Расчёт доверительных интервалов для каждого элемента
    for i in range(df.shape[0]):  # По строкам
        for j in range(df.shape[1]):  # По столбцам
            p_ij = df.iloc[i, j]  # Вероятность P_{i->j}
            n_i = sum_ratings[i]  # Общее число N_i для строки i

            # Проверка на допустимость расчёта
            if n_i > 0:
                se_ij = np.sqrt(p_ij * (1 - p_ij) / n_i)  # Стандартная ошибка
                ci_lower.iloc[i, j] = max(0, p_ij - z * se_ij)  # Нижняя граница
                ci_upper.iloc[i, j] = min(1, p_ij + z * se_ij)  # Верхняя граница
            else:
                # Если N_i = 0, установить NaN
                ci_lower.iloc[i, j] = np.nan
                ci_upper.iloc[i, j] = np.nan

    # Результаты: оригинальная матрица и доверительные интервалы
    print("Матрица вероятностей:")
    print(df)

    print("\nНижние границы доверительных интервалов:")
    print(ci_lower)

    print("\nВерхние границы доверительных интервалов:")
    print(ci_upper)

    return (df, ci_lower, ci_upper)


def deviation_matrix(df, ci_lower, ci_upper):
    cv_matrix = (ci_upper - ci_lower) / df
    # deviation_matrix = df.copy()
    #
    # # Рассчитываем отклонения
    # for i in range(df.shape[0]):  # По строкам
    #     for j in range(df.shape[1]):  # По столбцам
    #         p_ij = df.iloc[i, j]  # Вероятность P_{i->j}
    #         ci_l = ci_lower.iloc[i, j]  # Нижняя граница
    #         ci_u = ci_upper.iloc[i, j]  # Верхняя граница
    #         st.write(p_ij, ci_u, ci_l)
    #         if pd.notna(p_ij) and pd.notna(ci_l) and pd.notna(ci_u):  # Проверка на NaN
    #             st.write(True)
    #             if p_ij < ci_l:  # Отклонение ниже интервала
    #                 deviation_matrix.iloc[i, j] = (ci_l - p_ij) / ci_l
    #                 st.write((ci_l - p_ij) / ci_l)
    #             elif p_ij > ci_u:  # Отклонение выше интервала
    #                 deviation_matrix.iloc[i, j] = (p_ij - ci_u) / ci_u
    #                 st.write((ci_l - p_ij) / ci_l)
    #             else:  # Если внутри интервала, отклонение = 0
    #                 deviation_matrix.iloc[i, j] = 0
    #                 st.write((ci_l - p_ij) / ci_l)
    #         else:
    #             deviation_matrix.iloc[i, j] = np.nan  # Если данные отсутствуют
    #
    # # Результат
    # print("Отклонения вероятностей от доверительных интервалов:")
    # print(deviation_matrix)
    print(cv_matrix)
    return cv_matrix

def wald_migration(data: pd.DataFrame, agency: str, start_date: str, end_dates: str, scale: list,
                     step: dict, type_ogrn, type_date, type_rating):

    st.title('Markov process with discrete time and Wald method')
    agency_dict = {}
    if agency == 'Expert RA':
        agency_dict = expert_test
    if agency == 'NCR':
        agency_dict = NCR_test
    if agency == 'AKRA':
        agency_dict = akra
    if agency == 'S&P Global Ratings':
        agency_dict = s_and_p
    if agency == 'Fitch Ratings':
        agency_dict = fitch
    if agency == "Moody's Interfax Rating Agency":
        agency_dict = moodys
    if agency == 'NRA':
        agency_dict = nra

    full_df, sum_rating = calculate_discrete_migr(data, agency, start_date, end_dates, scale, step, type_ogrn, type_date, type_rating)
    print('Summ of each row: ', sum_rating)
    pie_cont = {}
    check_ = {}
    df, ci_lower, ci_upper = get_conf_int_wald(full_df, sum_rating)

    dev_matrix = deviation_matrix(full_df, ci_lower, ci_upper)
    # full_df = \
    # calculate_discrete_migr(data, agency, start_date, end_dates, scale, step, type_ogrn, type_date, type_rating)[0]

    # full_df.to_excel(f'{directory}/discrete_markov_step={step}.xlsx')
    n = 0
    name = ''
    # full_df.index = ['AAA','AA+', 'AA', 'AA-', 'A+', 'A','A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC', 'CC', 'C', 'D']
    # full_df.columns = ['AAA','AA+', 'AA', 'AA-', 'A+', 'A','A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC', 'CC', 'C', 'D']
    fig = plt.figure(figsize=(10, 10))
    plot = sns.heatmap(ci_upper, annot=True, fmt='.3f', linewidths=.5, annot_kws={"size": 11})
    # plt.savefig(f'{directory}/discrete_markov_step={step}.jpg')
    plt.close()

    for col in full_df.columns:  # TODO redact by index
        if full_df[col].sum() > 0:
            fig_ = go.Figure(data=[go.Pie(labels=full_df.index[full_df[col] > 0].tolist(),
                                          values=full_df[col][full_df[col] > 0])])
            fig_.update_layout(
                legend_title=f"{col} rating moved to:",
                font=dict(
                    family="Courier New, monospace",
                    size=15,
                    color="White"
                )
            )
            pie_cont[col] = fig_

    if st.checkbox('Display Pie chart of migration discrete matrix with Wald'):
        pies = st.sidebar.multiselect('Choose ratings to see moves', pie_cont)
        for pie_diag in pies:
            st.plotly_chart(pie_cont[pie_diag], theme='streamlit')

    if st.checkbox('Display graph of migration discrete matrix with Wald'):
        HtmlFile = open(graph_matric_migration(full_df), 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height=650, width=650)

    if st.checkbox('Display static graph of migration discrete matrix with Wald'):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(graph_matric_migration_stat(full_df))

    if st.checkbox('Display migration discrete matrix with Wald'):
        # full_df.to_excel(f'discrete_time_step={step}.xlsx')
        st.pyplot(fig)
        fig.savefig((f'results/{agency}/images/discrete_step={step}_{datetime.now().strftime('%Y-%m-%d')}_with_Wald.jpg'))

    if st.checkbox('Display predict discr. migration matrix with Wald'):
        fig = plt.figure(figsize=(10, 10))
        n = st.number_input('Enter number', max_value=1000, min_value=2)
        check_1 = np.linalg.matrix_power(full_df, n)
        plot = sns.heatmap(check_1, annot=True, fmt='.3f', linewidths=.5, annot_kws={"size": 11})
        # plt.savefig(f'{directory}/time_cont_step={step}_second_avar.jpg')
        st.pyplot(fig)
        fig.savefig(
            (f'results/{agency}/images/predict_{n}_discrete_step={step}_{datetime.now().strftime('%Y-%m-%d')}_with_Wald.jpg'))
        plt.close()

    if st.checkbox('Get predict of migration discrete matrix with Wald'):
        # get_state_by_time(data, agency, start_date, scale)
        n = st.number_input('Enter number', max_value=100, min_value=2)
        check_1 = np.linalg.matrix_power(full_df, n)
        st.write(pd.DataFrame(check_1, columns=list(agency_dict.keys()), index=list(agency_dict.keys())))
        check_1 = pd.DataFrame(check_1, columns=list(agency_dict.keys()), index=list(agency_dict.keys()))
        name = f"results/{agency}/discrete_time/predict_discrete_step={step}_predict={n}_{datetime.now().strftime('%Y-%m-%d')}_with_Wald.xlsx"
        check_1.to_excel(name)
        f = Fitter(check_1,
                   distributions=['cauchy', 'chi2', 'expon', 'exponpow', 'gamma', 'lognorm', 'norm', 'powerlaw',
                                  'rayleigh', 'uniform', 'beta'])
        f.fit()
        distr_fit = f.summary()
        st.write(distr_fit)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(f.plot_pdf())
        st.pyplot(f.plot_pdf())

    if st.checkbox('Display distribution of migration discrete matrix with Wald'):
        # full_df = get_state_by_time(data, agency, date_to_check, step, scale)[0]
        f = Fitter(full_df,
                   distributions=['cauchy', 'chi2', 'expon', 'exponpow', 'gamma', 'lognorm', 'norm', 'powerlaw',
                                  'rayleigh', 'uniform', 'beta'])
        f.fit()
        distr_fit = f.summary()
        st.write(distr_fit)
        st.pyplot(f.plot_pdf())
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(f.plot_pdf())

    if st.button('Download discrete matrix with Wald', key='download'):
        full_df.to_excel(
            f'results/{agency}/discrete_time/discrete_step={step}_{datetime.now().strftime('%Y-%m-%d')}_with_Wald.xlsx')

@st.cache_data(experimental_allow_widgets=True)
def calculate_discrete_migr_beta(data: pd.DataFrame, agency: str, start_date: str, end_dates: str, scale: list, step, col_ogrn: str, col_date: str, col_rating: str):
    def calculate_state_weights(result_migr):
        # Словарь для хранения весов состояний
        state_weights = {}

        for state, transitions in result_migr.items():
            # Если все переходы только в себя
            if (transitions.count(state) / len(transitions)) > 0.7:  # Все переходы в себя
                # state_weights[state] = 0.1  # Маленький вес для таких состояний
                state_weights[state] = 1 / (len(transitions) + 1e-5)
            else:
                # Веса для состояний с переходами в другие состояния
                # state_weights[state] = 1 / (len(transitions) + 1e-5)
                state_weights[state] = 0.1

        return state_weights

    agency_dict = {}
    alpha_prior = None
    if alpha_prior is None:
        alpha_prior = np.ones(len(agency_dict))  # Равномерное априорное распределение
    if agency == 'Expert RA':
        agency_dict = expert_test
    if agency == 'NCR':
        agency_dict = NCR_test
    if agency == 'AKRA':
        agency_dict = akra
    if agency == 'S&P Global Ratings':
        agency_dict = s_and_p
    if agency == 'Fitch Ratings':
        agency_dict = fitch
    if agency == "Moody's Interfax Rating Agency":
        agency_dict = moodys
    if agency == 'NRA':
        agency_dict = nra

    full_step = None
    curent_step = None
    time_counter = None
    step_num = None

    if 'months' in step.keys():
        step_num = step['months']
        full_step = relativedelta(months=1)
        curent_step = relativedelta(months=step_num)
        time_counter = 30

    if 'years' in step.keys():
        step_num = step['years']
        full_step = relativedelta(years=1)
        curent_step = relativedelta(years=step_num)
        time_counter = 363

    if 'days' in step.keys():
        step_num = step['days']
        full_step = relativedelta(days=1)
        curent_step = relativedelta(days=step_num)
        time_counter = 1

    full_df = pd.DataFrame()
    data_1 = pd.DataFrame()
    result = {}
    if len(scale) != 0:
        for scal in scale:
            data_1 = pd.concat([data_1, data[(data['agency'] == agency) & (data['scale'] == scal)].reset_index().drop(columns=['index'])], ignore_index=True)
    else:
        data_1 = pd.concat([data_1, data[(data['agency'] == agency)].reset_index().drop(columns=['index'])],
                           ignore_index=True)

    data_1 = data_1.sort_values(col_date)
    st.write(len(data_1[col_ogrn].unique()))
    counter = 0
    set_ogrn = (data_1[col_ogrn].unique())
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    result_migr = {}

    for ogrn in set_ogrn:
        if pd.notna(ogrn):
            time.sleep(0.01)
            counter += 1
            my_bar.progress(int(100 * counter / len(set_ogrn)), text=progress_text)
            pr = data_1.loc[data_1[col_ogrn] == ogrn].reset_index(drop=True).sort_values(col_date)
            start_dates = start_date

            while start_dates < (datetime.strptime(end_dates, "%Y-%m-%d")).strftime('%Y-%m-%d'):
                data_prev = pr[(pr[col_date] < start_dates)]
                end_date = (datetime.strptime(start_dates, "%Y-%m-%d") + curent_step).strftime('%Y-%m-%d')
                temp_df = pr.loc[(pr[col_date] >= start_dates) & (pr[col_date] <= end_date)].reset_index(drop=True).sort_values(col_date)

                first = ''
                first_date = ''
                last = ''
                if len(temp_df) > 0:
                    if temp_df[col_date][0] == start_dates:
                        first = temp_df[col_rating][0]
                    else:
                        if len(get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)) > 0:
                            first, first_date = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)

                    temp_local_df = temp_df.loc[temp_df[col_date] == temp_df[col_date][len(temp_df) - 1]]
                    if len(temp_local_df) > 1 and 'Рейиинг отозван' in temp_local_df[col_rating].values:
                        last = 'Рейтинг отозван'
                    else:
                        last = temp_df[col_rating][len(temp_df) - 1]

                    if last == 'Рейтинг отозван':
                        temp_local_df = temp_df.loc[temp_df[col_date] == temp_df[col_date][len(temp_df) - 1]]
                        if len(temp_local_df) > 1 and default[agency] in temp_local_df[col_rating].values:
                            last = default[agency]

                else:
                    if len(get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)) > 0:
                        first, first_date = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)
                        last = first

                if first_date != None:
                    years = (datetime.strptime(start_dates, "%Y-%m-%d") - datetime.strptime(first_date, "%Y-%m-%d")).days // 365
                    if years > 1:
                        first = ''

                if first != '' and first != default[agency]:
                    if first in agency_dict and last in agency_dict:
                        if first not in result_migr:
                            result_migr[first] = []
                        result_migr[first].append(last)

                start_dates = (datetime.strptime(start_dates, "%Y-%m-%d") + full_step).strftime('%Y-%m-%d')

    transition_matrix = pd.DataFrame(0, index=agency_dict.keys(), columns=agency_dict.keys())

    for from_state, to_states in result_migr.items():
        for to_state, count in Counter(to_states).items():
            transition_matrix.loc[from_state, to_state] = count

    for state in transition_matrix.index:
        total_transitions = transition_matrix.loc[state].sum()
        if total_transitions > 0:
            for target_state in transition_matrix.columns:
                N_ij = transition_matrix.loc[state, target_state]
                alpha_post = 1 + N_ij
                beta_post = 1 + total_transitions - N_ij
                mean_posterior = beta.mean(alpha_post, beta_post)
                transition_matrix.loc[state, target_state] = mean_posterior
        else:
            # Если нет переходов, оставляем вероятность остаться в этом состоянии
            transition_matrix.loc[state, state] = 1.0

    # Нормализация строк матрицы, чтобы суммы были равны 1
    transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).fillna(0)

    st.write("Posterior Transition Matrix:")
    st.dataframe(transition_matrix)

    # Применение бутстрепа
    transition_matrix_boost, ci_matrix = bootstrap_transition_matrix(result_migr, agency_dict)
    st.write("Transition Matrix (Bootstrap Average):")
    st.dataframe(transition_matrix_boost)
    st.write("Confidence Intervals from Bootstrap:")
    st.dataframe(ci_matrix)

    my_bar.empty()
    return transition_matrix

@st.cache_data(experimental_allow_widgets=True, show_spinner=False)
def calculate_time_cont_migr_beta(data: pd.DataFrame, agency: str, start_date: str, end_dates: str, step: dict, col_ogrn: str, col_date: str, col_rating: str):
    result_num = {}
    result_non_num = {}
    result_full = {}
    agency_dict = {}
    if agency == 'Expert RA':
        agency_dict = expert_test
    if agency == 'NCR':
        agency_dict = NCR_test
    if agency == 'AKRA':
        agency_dict = akra
    if agency == 'S&P Global Ratings':
        agency_dict = s_and_p
    if agency == 'Fitch Ratings':
        agency_dict = fitch
    if agency == "Moody's Interfax Rating Agency":
        agency_dict = moodys
    if agency == 'NRA':
        agency_dict = nra

    full_df = get_nan_df(agency_dict)

    full_step = None
    curent_step = None
    time_counter = None
    step_num = None

    if 'months' in step.keys():
        step_num = step['months']
        full_step = relativedelta(months=1)
        curent_step = relativedelta(months=step['months'])
        time_counter = 31

    if 'years' in step.keys():
        step_num = step['years']
        full_step = relativedelta(years=1)
        curent_step = relativedelta(years=step['years'])
        time_counter = 365

    if 'days' in step.keys():
        step_num = step['days']
        full_step = relativedelta(days=1)
        curent_step = relativedelta(days=step_num)
        time_counter = 1

    # data_prev = data[(data['agency'] == agency) & (data[col_date] < start_date)]
    data_1 = pd.DataFrame()
    if len(scale) != 0:
        for scal in scale:
            data_1 = pd.concat([data_1, data[
                (data['agency'] == agency) & (data['scale'] == scal)].reset_index().drop(columns=['index'])], ignore_index=True)
    else:
        data_1 = pd.concat([data_1, data[(data['agency'] == agency)].reset_index().drop(columns=['index'])], ignore_index=True)

    data_1 = data_1.sort_values(col_date)
    counter = 0
    set_ogrn = (data_1[col_ogrn].unique())
    progress_text = "Calculate time cont. matrix. Please wait."
    my_bar_1 = st.progress(0, text=progress_text)
    counter_CC_D = 0
    for ogrn in set_ogrn:  # todo iterate over full df
        if pd.isna(ogrn) != True:  # todo if ogrn in not None
            time.sleep(0.01)
            my_bar_1.progress(int(100 * counter / len(set_ogrn)) , text=progress_text)
            counter += 1
            pr = data_1[data_1[col_ogrn] == ogrn].reset_index().drop(columns=['index']).sort_values(col_date)
            start_dates = start_date # pr[col_date][0]
            while start_dates < (datetime.strptime(end_dates, "%Y-%m-%d")).strftime('%Y-%m-%d'):

                result = {}
                result_check_full = {}
                result_state_in = {}
                container_state_in = []
                data_prev = pr[(pr[col_date] < start_dates)]
                end_date = (datetime.strptime(start_dates, "%Y-%m-%d") + curent_step).strftime('%Y-%m-%d')
                temp_df = pr[(pr[col_date] >= start_dates) & (pr[col_date] <= end_date)].reset_index().drop(columns=['index']).sort_values(col_date)
                first = ''
                date_start = ''
                # last = ''
                # temp_df = data_1[(data_1['ogrn'] == ogrn)].reset_index().drop(columns=['index']).sort_values('_date')  # todo get df by ogrn
                ind_to_drop_otozv = []
                if len(temp_df) > 0:
                    if temp_df[col_date][0] == start_dates:
                        first = temp_df[col_rating][0]
                        date_start = start_dates
                    else:
                        if len(get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)) > 0:
                            first, date_start = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)  # todo get previous rating on date = start_date
                            # date_start = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)[1]
                else:
                    if len(get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)) > 0:
                        first, date_start = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)  # todo get previous rating on date = start_date
                        # date_start = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)[1]

                if date_start != '':
                    years = (datetime.strptime(start_dates, "%Y-%m-%d") - datetime.strptime(date_start,
                                                                                           "%Y-%m-%d")).days // 365
                    if years >= 1:
                        # st.write(temp_df['_name'][0], date_start, start_date, first)
                        first = ''
                if first != '' and first != default[agency]:
                    if first not in result_state_in and first in agency_dict:  # todo remove result_state_in from final version(do not forget)
                        result_state_in[first] = []
                    if len(temp_df) > 0:  # todo iterate over exist temp_df with unique ogrn, because in previous step save all unique ogrn
                            first_df = temp_df.loc[[0]]
                            first_df.at[0, col_rating] = first
                            first_df.at[0, col_date] = start_dates
                            temp_df = pd.concat([first_df, temp_df], axis=0).sort_values(col_date).reset_index().drop(columns=['index']).drop_duplicates()
                            if len(set(temp_df[col_rating].values)) == 1:
                                    container_state_in.append(True)
                            elif len(set(temp_df[col_rating].values)) > 1 and first == default[agency] and 'отозван' in temp_df[col_rating].values:
                                    container_state_in.append(True)
                            else:
                                    container_state_in.append(False)
                                    if 'отозван' in temp_df[col_rating].values:
                                        for indx in temp_df.index:
                                            if len(temp_df[temp_df[col_date] == temp_df[col_date][indx]]) > 1 and 'отозван' in \
                                                    temp_df[temp_df[col_date] == temp_df[col_date][indx]][col_rating].values \
                                                    and default[agency] not in temp_df.loc[temp_df[col_date] == temp_df[col_date][indx]][col_rating].values:
                                                        df_otozv = temp_df[temp_df[col_date] == temp_df[col_date][indx]].reset_index()  # todo df to check if any отозван in temp_df by _date
                                                        for k_indx in df_otozv.index:
                                                            if df_otozv[col_rating][k_indx] != 'отозван':
                                                                ind_to_drop_otozv.append(df_otozv['index'][k_indx])

                                            elif len(temp_df.loc[temp_df[col_date] == temp_df[col_date][indx]]) > 1 and 'отозван' in \
                                                    temp_df.loc[temp_df[col_date] == temp_df[col_date][indx]][col_rating].values and \
                                                    default[agency] in temp_df.loc[temp_df[col_date] == temp_df[col_date][indx]][col_rating].values:

                                                    df_otozv = temp_df.loc[temp_df[col_date] == temp_df[col_date][indx]].reset_index()  # todo df to check if any отозван in temp_df by _date
                                                    for k_indx in df_otozv.index:
                                                        if df_otozv[col_rating][k_indx] == 'отозван':
                                                            ind_to_drop_otozv.append(df_otozv['index'][k_indx])
                    else:
                        container_state_in.append(True)

                    if False not in container_state_in and first in agency_dict:
                        result_state_in[first].append(first)  # todo upd each first state with new information
                    # todo if error with 0 index, reset index there in temp_df
                    if False in container_state_in and first in agency_dict:
                        temp_df = temp_df.drop(index=ind_to_drop_otozv).reset_index().drop(columns=['index'])
                        test_rat = temp_df[col_rating].values
                        test_date = temp_df[col_date].values
                        test_name = temp_df['_name'].values

                        curent_rating = test_rat[0]
                        curent_date = test_date[0]
                        for i, rat in enumerate(test_rat):
                            if rat not in result_check_full:
                                result_check_full[rat] = []
                                if curent_rating in agency_dict and rat in agency_dict:
                                    if curent_rating != rat:
                                        result_check_full[curent_rating].append((curent_date, rat, test_date[i]))
                                        if curent_rating == 'CC.ru' and rat == 'D.ru':
                                            st.write(curent_date, test_date[i], test_name[i])
                                            counter_CC_D += 1
                                        curent_rating = rat
                                        curent_date = test_date[i]

                        for i, rat in enumerate(test_rat):
                            if rat not in result_check_full:
                                result_check_full[rat] = []
                            if i + 1 >= len(test_rat):
                                if rat in agency_dict:
                                    last_rat = rat
                                    last_dat = test_date[i]
                                    test_rat_rev = test_rat[::-1]
                                    test_date_rev = test_date[::-1]
                                    # test_name_rev = test_name[::-1]
                                    num = 0
                                    for j, rat_rev in enumerate(test_rat_rev):
                                        if rat_rev != last_rat:
                                            num = j - 1
                                            last_dat = test_date_rev[j]
                                            if test_rat_rev[num] in agency_dict:
                                                break

                                    # st.write(test_rat_rev[num], test_name_rev[num], test_date_rev[num], start_dates, end_date)
                                    if test_rat_rev[num] in agency_dict:
                                        result_check_full[test_rat_rev[num]].append((last_dat, test_rat[len(test_rat) - 1], test_date_rev[num], end_date, rat))

                    for key_ in agency_dict.keys():
                        if key_ not in result_num:
                            result_num[key_] = {}
                        if key_ not in result_non_num:
                            result_non_num[key_] = []
                        count_moves_time = 0.0
                        last_stat = 0.0
                        state_in = 0.0
                        if key_ in count_last(result_check_full, start_dates, end_date):
                            last_stat = sum(count_last(result_check_full, start_dates, end_date)[key_])
                        if key_ in result_state_in.keys():
                            state_in = len(result_state_in[key_])
                        if key_ in count_moves(result_check_full).keys():
                            count_moves_time = sum(count_time(result_check_full, start_dates, end_date)[key_])
                            for k, v in count_moves(result_check_full)[key_].items():
                                if k not in result_num[key_]:
                                    result_num[key_][k] = []
                                # todo use this ti make seocnd method
                                #     result[key_][k] = count_moves(result_check_full)[key_][k] / (state_in + sum(count_time(result_check_full, start_date, end_date)[key_]) + last_stat)
                                result_num[key_][k].append(count_moves(result_check_full)[key_][k])
                        result_non_num[key_].append(state_in + count_moves_time + last_stat)
                start_dates = (datetime.strptime(start_dates, "%Y-%m-%d") + full_step).strftime('%Y-%m-%d')

    alpha_prior = 1
    beta_prior = 1

    # TODO realize method of time-continous process + think about NR rating for first method     07_02_2024
    # Updateing probability with beta - distribution
    for key, val in result_num.items():
        if key not in result_full:
            result_full[key] = {}

        if key in result_non_num.keys():
            for k, v in val.items():
                # parameters of beta - distr
                alpha_posterior = alpha_prior + sum(result_num[key][k])  # success movements
                beta_posterior = beta_prior + (sum(result_non_num[key]) - sum(result_num[key][k]))  # unsuccessful

                # Probability assessment
                transition_probability = beta.mean(alpha_posterior, beta_posterior)
                result_full[key][k] = transition_probability

    time.sleep(1)
    my_bar_1.empty()

    result_full_df = pd.DataFrame().from_dict(result_full).fillna(0).reset_index()
    result_full_df = get_generator(sort_df(result_full_df, agency_dict))
    result = expm(result_full_df.to_numpy())
    columns_ag = list(agency_dict.keys())
    result_df = pd.DataFrame(result, columns=columns_ag, index=columns_ag)

    # todo use this to make second method
    return result_df

# Функция бутстрепа
def bootstrap_transition_matrix(result_migr, agency_dict, n_samples=1000):
    states = list(agency_dict.keys())
    n_states = len(states)

    # Хранение вероятностей для каждой ячейки матрицы
    probabilities_samples = {state_from: {state_to: [] for state_to in states} for state_from in states}

    for _ in range(n_samples):
        # Создаем новую выборку и заполняем временную матрицу
        temp_matrix = pd.DataFrame(0, index=states, columns=states)

        for from_state in states:
            # Переходы из состояния `from_state`
            transitions = result_migr.get(from_state, [])
            if len(transitions) > 0:
                # Бутстреп-выборка с возвращением
                sampled_transitions = np.random.choice(transitions, size=len(transitions), replace=True)

                for to_state in states:
                    temp_matrix.loc[from_state, to_state] = np.sum(np.array(sampled_transitions) == to_state)

        # Нормализация и сохранение вероятностей
        for from_state in states:
            row_sum = temp_matrix.loc[from_state].sum()
            if row_sum > 0:
                normalized_row = temp_matrix.loc[from_state] / row_sum
            else:
                normalized_row = pd.Series(0, index=states)
                normalized_row[from_state] = 1  # Самопереход для пустых случаев

            # Сохранение вероятностей для каждой ячейки
            for to_state in states:
                probabilities_samples[from_state][to_state].append(normalized_row[to_state])

    # Формирование итоговой матрицы и доверительных интервалов
    transition_matrix = pd.DataFrame(0.0, index=states, columns=states)
    ci_matrix = pd.DataFrame(index=states, columns=states)

    for from_state in states:
        for to_state in states:
            probs = probabilities_samples[from_state][to_state]
            # Среднее значение
            transition_matrix.loc[from_state, to_state] = np.mean(probs)
            # Доверительный интервал (например, 2.5% и 97.5%)
            ci_lower = np.percentile(probs, 2.5)
            ci_upper = np.percentile(probs, 97.5)
            ci_matrix.loc[from_state, to_state] = f"({ci_lower:.3f}, {ci_upper:.3f})"

    return transition_matrix, ci_matrix

def bootstrap_transition_matrix_beta(beta_params, n_samples=10000):
    states = list(beta_params.keys())

    # Хранение вероятностей для каждой ячейки матрицы
    probabilities_samples = {state_from: {state_to: [] for state_to in states} for state_from in states}

    for _ in range(n_samples):
        # Генерация вероятностей переходов на основе бета-распределения
        temp_matrix = pd.DataFrame(0.0, index=states, columns=states)

        for from_state in states:
            alpha_beta_dict = beta_params.get(from_state, {})

            total_sample = 0
            sampled_probs = {}

            for to_state in states:
                if to_state in alpha_beta_dict:
                    alpha_ij = alpha_beta_dict[to_state]["alpha"]
                    beta_ij = alpha_beta_dict[to_state]["beta"]
                    sampled_probs[to_state] = beta.rvs(alpha_ij, beta_ij)  # Генерация из Beta(alpha, beta)
                    total_sample += sampled_probs[to_state]

            # Нормализация, чтобы суммы по строкам давали 1
            if total_sample > 0:
                for to_state in states:
                    temp_matrix.loc[from_state, to_state] = sampled_probs.get(to_state, 0) / total_sample

        # Сохранение вероятностей в список
        for from_state in states:
            for to_state in states:
                probabilities_samples[from_state][to_state].append(temp_matrix.loc[from_state, to_state])

    # Формирование итоговой матрицы и доверительных интервалов
    transition_matrix = pd.DataFrame(0.0, index=states, columns=states)
    ci_matrix = pd.DataFrame(index=states, columns=states)

    for from_state in states:
        for to_state in states:
            probs = probabilities_samples[from_state][to_state]
            transition_matrix.loc[from_state, to_state] = np.mean(probs)  # Среднее значение
            ci_lower = np.percentile(probs, 2.5)  # Нижняя граница 95% интервала
            ci_upper = np.percentile(probs, 97.5)  # Верхняя граница 95% интервала
            ci_matrix.loc[from_state, to_state] = f"({ci_lower:.3f}, {ci_upper:.3f})"

    return transition_matrix, ci_matrix


@st.cache_data(experimental_allow_widgets=True)
def calculate_discrete_migr_bayesian(data: pd.DataFrame, agency: str, start_date: str, end_dates: str, scale: list, step, col_ogrn: str, col_date: str, col_rating: str):
    agency_dict = {}

    full_step = None
    curent_step = None
    time_counter = None
    step_num = None

    if 'months' in step.keys():
        step_num = step['months']
        full_step = relativedelta(months=1)
        curent_step = relativedelta(months=step_num)
        time_counter = 30

    if 'years' in step.keys():
        step_num = step['years']
        full_step = relativedelta(years=1)
        curent_step = relativedelta(years=step_num)
        time_counter = 363

    if 'days' in step.keys():
        step_num = step['days']
        full_step = relativedelta(days=1)
        curent_step = relativedelta(days=step_num)
        time_counter = 1

    if agency == 'Expert RA':
        agency_dict = expert_test
    if agency == 'NCR':
        agency_dict = NCR_test
    if agency == 'AKRA':
        agency_dict = akra
    if agency == 'S&P Global Ratings':
        agency_dict = s_and_p
    if agency == 'Fitch Ratings':
        agency_dict = fitch
    if agency == "Moody's Interfax Rating Agency":
        agency_dict = moodys
    if agency == 'NRA':
        agency_dict = nra

    full_df = get_nan_df(agency_dict)
    data_1 = data
    data_1 = data_1.sort_values(col_date)
    st.write(len(data_1[col_ogrn].unique()))
    counter = 0
    set_ogrn = (data_1[col_ogrn].unique())
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    alpha, beta = 1, 1
    gamma = 0.5
    lamda = 10
    tau = 0.5
    n0 = 5
    counter_def = 0
    eps = 10 ** (-2)
    df_sum = pd.DataFrame(0, index=full_df.index, columns=["Total"])
    full_ = defaultdict(Counter)
    full_dict = {
        index: {
            col: {'alpha': alpha, 'beta': beta} for col in full_df.columns
        } for index in full_df.index
    }
    st.write(full_dict.keys())
    counter_CC_D = 0
    for ogrn in set_ogrn:
        if pd.isna(ogrn) != True:
            result_migr = defaultdict(list)
            time.sleep(0.01)
            my_bar.progress(int(100 * counter / len(set_ogrn)), text=progress_text)
            counter += 1
            pr = data_1.loc[data_1[col_ogrn] == ogrn].reset_index(drop=True).sort_values(col_date)
            start_dates = pr[col_date].iloc[0]
            end_dates = f"{pd.to_datetime(pr[col_date][len(pr) - 1]).year}-12-31"
            while start_dates < (datetime.strptime(end_dates, "%Y-%m-%d")).strftime('%Y-%m-%d'):
                data_prev = pr[(pr[col_date] < start_dates)]
                end_date = (datetime.strptime(start_dates, "%Y-%m-%d") + curent_step).strftime('%Y-%m-%d')
                temp_df = pr[(pr[col_date] >= start_dates) & (pr[col_date] <= end_date)].reset_index().drop(columns=['index']).sort_values(col_date)
                # first = pr[col_rating].iloc[0]  # Если только одна запись, берем ее
                first = ''
                date_start = ''
                if len(temp_df) > 0:
                    if temp_df[col_date].iloc[0] == start_dates:
                        first = temp_df[col_rating].iloc[0]
                        date_start = start_dates
                    else:
                        if len(get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)) > 0:
                            first, date_start = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)  # todo get previous rating on date = start_date
                            # date_start = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)[1]
                else:
                    if len(get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)) > 0:
                        first, date_start = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)  # todo get previous rating on date = start_date
                        # date_start = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)[1]
                if len(temp_df) > 1:
                    transitions = []
                    prev_rating = first
                    prev_year = pd.to_datetime(date_start).year
                    for i in range(len(temp_df)):  # Проходим по записям начиная со второго элемента
                        cur_rating = temp_df[col_rating].iloc[i]
                        cur_year = pd.to_datetime(temp_df[col_date].iloc[i]).year

                        if pd.to_datetime(temp_df[col_date].iloc[i]) != pd.to_datetime(date_start):
                            if cur_year - prev_year > 1:
                                continue  # Пропускаем большие разрывы во времени
                            else:
                                result_migr[prev_rating].append(cur_rating)
                                # transitions.append((prev_rating, cur_rating))  # Запоминаем переход

                                prev_rating = cur_rating  # Обновляем "предыдущее" значение

                            prev_year = pd.to_datetime(temp_df[col_date].iloc[i - 1]).year
                else:
                    if first in agency_dict:
                        result_migr[first].append(first)

                transition_counts = defaultdict(Counter)

                for start_rating in full_dict.keys():
                    if start_rating in result_migr:
                        for end_rating in result_migr[start_rating]:
                            if start_rating == "CCC-C" and end_rating == "CCC-C":
                                print(ogrn)
                            transition_counts[start_rating][end_rating] += 1
                            full_[start_rating][end_rating] += 1
                    else:
                        for end_rating_ in full_dict.keys():
                            transition_counts[start_rating][end_rating_] += 0
                            full_[start_rating][end_rating_] += 0

                transition_df = pd.DataFrame.from_dict(transition_counts, orient='index').fillna(0)
                transition_df["Total"] = transition_df.sum(axis=1)
                df_sum = pd.DataFrame.from_dict(full_, orient='index').fillna(0)
                df_sum["Total"] = df_sum.sum(axis=1)
                for key, value_d in full_dict.items():
                    for key_d, value_in_d in value_d.items():

                        if key_d in result_migr.values():
                            counter_CC_D += 1

                        n_ = df_sum.at[key, "Total"]
                        n_loc = df_sum.at[key, key_d]
                        if key in result_migr:
                            if key_d in result_migr[key]:
                                # if key_d == key:
                                #     # print(key_d, key)
                                old_rank = agency_dict[key]
                                new_rank = agency_dict[key_d]
                                penalty = abs((new_rank - old_rank))

                                if penalty == 0:
                                    penalty = 1

                                t_alpha = value_in_d["alpha"]
                                t_beta = value_in_d["beta"]

                                #todo add penalty for transition
                                full_dict[key][key_d]["alpha"] += transition_counts[key][key_d]
                                full_dict[key][key_d]["beta"] += (transition_df.at[key, "Total"] - transition_counts[key][key_d])

                        weight = 1 / (1 + np.exp(tau * (n_ - n0)))
                        k = 2
                        # if key != key_d:
                        if transition_counts[key][key_d] == 0:
                            old_rank = agency_dict[key]
                            new_rank = agency_dict[key_d]
                            # penalty = (new_rank - old_rank) ** 2
                            penalty = abs((new_rank ** 2 - old_rank ** 2))

                            # if penalty == 0:
                            #     penalty = 1

                            # full_dict[key][key_d]["beta"] += (penalty * (1 / (1 + n_loc / (n_ + 1)))) ** (penalty)
                            full_dict[key][key_d]["beta"] += penalty * (1 - weight)

                start_dates = (datetime.strptime(start_dates, "%Y-%m-%d") + full_step).strftime('%Y-%m-%d')
            # st.write(result_migr.keys())

    st.write(full_dict)
    # Пример использования
    transition_matrix, confidence_intervals = bootstrap_transition_matrix_beta(full_dict, n_samples=1000)

    # Вывод результатов
    st.write("Матрица переходных вероятностей:")
    st.write(transition_matrix)

    print("\nДоверительные интервалы:")
    print(confidence_intervals)

    index_labels = list(full_dict.keys())
    columns_labels = list(next(iter(full_dict.values())).keys())

    df = pd.DataFrame(index=index_labels, columns=columns_labels)

    for row_key, row_values in full_dict.items():
        for col_key, params in row_values.items():
            alpha, beta = params["alpha"], params["beta"]
            df.at[row_key, col_key] =  alpha / (alpha + beta)

    time.sleep(1)
    my_bar.empty()

    st.write("final version with beta", df.div(df.sum(axis=1), axis=0))

    return df.div(df.sum(axis=1), axis=0)


@st.cache_data(experimental_allow_widgets=True)
def calculate_discrete_migr_dirichlet(data: pd.DataFrame, agency: str, start_date: str, end_dates: str,
                                     scale: list, step, col_ogrn: str, col_date: str, col_rating: str):
    agency_dict = {}

    # Инициализация параметров шага времени
    full_step = None
    curent_step = None
    time_counter = None
    step_num = None

    if 'months' in step.keys():
        step_num = step['months']
        full_step = relativedelta(months=1)
        curent_step = relativedelta(months=step_num)
        time_counter = 30

    elif 'years' in step.keys():
        step_num = step['years']
        full_step = relativedelta(years=1)
        curent_step = relativedelta(years=step_num)
        time_counter = 363

    else:
        step_num = step['days']
        full_step = relativedelta(days=1)
        curent_step = relativedelta(days=step_num)
        time_counter = 1

    # Инициализация словаря рейтингов
    if agency == 'Expert RA':
        agency_dict = expert_test
    elif agency == 'NCR':
        agency_dict = NCR_test
    elif agency == 'AKRA':
        agency_dict = akra
    elif agency == 'S&P Global Ratings':
        agency_dict = s_and_p
    elif agency == 'Fitch Ratings':
        agency_dict = fitch
    elif agency == "Moody's Interfax Rating Agency":
        agency_dict = moodys
    elif agency == 'NRA':
        agency_dict = nra

    # Инициализация параметров Дирихле
    dirichlet_params = {
        rating: {k: 1.0 for k in agency_dict}  # Uniform prior
        for rating in agency_dict
    }
    # alpha_0 = 1  # Меньшее базовое значение
    # dirichlet_params = {
    #     rating: {k: (1.0 if k == rating else alpha_0) for k in agency_dict}
    #     for rating in agency_dict
    # }
    # Подготовка данных
    data_1 = data.sort_values(col_date)
    set_ogrn = data_1[col_ogrn].unique()
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    num_samples = 3000  # Количество выборок
    # 1. Сбор переходов по объектам
    obj_transitions = defaultdict(lambda: defaultdict(int))
    # Основной цикл обработки
    for counter, ogrn in enumerate(set_ogrn, 1):
        if pd.isna(ogrn):
            continue

        my_bar.progress(int(100 * counter / len(set_ogrn)),
                        text=f"Processing {counter}/{len(set_ogrn)} companies")

        pr = data_1.loc[data_1[col_ogrn] == ogrn].reset_index(drop=True).sort_values(col_date)
        start_dates = pr[col_date].iloc[0]
        end_dates = f"{pd.to_datetime(pr[col_date].iloc[-1]).year}-12-31"

        pr = data_1.loc[data_1[col_ogrn] == ogrn].reset_index(drop=True).sort_values(col_date)
        if len(pr) < 1:
            continue  # Пропускаем, если недостаточно данных для переходов

        prev_rating = pr[col_rating].iloc[0]  # Начальный рейтинг
        prev_date = pd.to_datetime(pr[col_date].iloc[0])

        for i in range(1, len(pr)):
            current_rating = pr[col_rating].iloc[i]
            current_date = pd.to_datetime(pr[col_date].iloc[i])

            # Пропускаем, если между датами разрыв > 1 года
            if (current_date - prev_date).days > 365:
                prev_rating = current_rating
                prev_date = current_date
                continue

            if prev_rating == 'AA-A' and current_rating == 'D':
                print(ogrn)
            obj_transitions[prev_rating][current_rating] += 1  # Фиксируем переход
            prev_rating = current_rating
            prev_date = current_date

    # Обработка для каждого состояния
    # Параметры
    transition_matrix = pd.DataFrame(
        index=agency_dict.keys(),
        columns=agency_dict.keys(),
        dtype=float
    )

    # 2. Формирование параметров Дирихле
    posterior_params = np.zeros((len(agency_dict), len(agency_dict)))
    for i, from_rating in enumerate(agency_dict):
        total_count = sum(obj_transitions[from_rating].values())  # Всего переходов из данного рейтинга

        for j, to_rating in enumerate(agency_dict):
            # Добавляем сглаживание, чтобы не было нулевых вероятностей
            old_rank = agency_dict[from_rating]
            new_rank = agency_dict[to_rating]
            penalty = abs((new_rank - old_rank))
            if penalty == 0:
                penalty = 0.5
            posterior_params[i, j] = (obj_transitions[from_rating][to_rating] + 1) / penalty
            # posterior_params[i, j] = obj_transitions[from_rating][to_rating] + 1
    # for i, from_rating in enumerate(agency_dict):
    #     total_count = sum(obj_transitions[from_rating].values())  # Всего переходов из данного рейтинга
    #
    #     for j, to_rating in enumerate(agency_dict):
    #         # Добавляем сглаживание, чтобы не было нулевых вероятностей
    #         posterior_params[i, j] = obj_transitions[from_rating][to_rating] + ALPHA_OTHER

    # st.write(posterior_params)
    # 3. Сэмплирование вероятностей переходов
    samples = np.array([stats.dirichlet.rvs(posterior_params[i], size=num_samples) for i in range(len(agency_dict))])

    # 4. Байесовская оценка матрицы миграции
    P_estimated = samples.mean(axis=1)

    # Обновляем transition_matrix
    for i, from_rating in enumerate(agency_dict):
        for j, to_rating in enumerate(agency_dict):
            transition_matrix.at[from_rating, to_rating] = round(P_estimated[i, j], 4)

    st.write("Байесовская оценка матрицы миграции рейтингов:")
    st.write(transition_matrix)

    return transition_matrix


@st.cache_data(experimental_allow_widgets=True)
def calculate_discrete_migr_dirichlet_2_0(data: pd.DataFrame, agency: str, start_date: str, end_dates: str,
                                      scale: list, step, col_ogrn: str, col_date: str, col_rating: str):
    # Инициализация параметров шага времени
    full_step = None
    curent_step = None
    time_counter = None
    step_num = None

    PENALTY_FACTOR = 1    # Коэффициент снижения для редких переходов
    MIN_OBSERVATIONS = 2    # Минимальное число наблюдений
    ALPHA_SELF = 0.05       # Базовый априор для самоперехода
    ALPHA_OTHER = 1       # Базовый априор для других переходов
    ALPHA_DEFAULT = 0.5     # Априор для дефолта

    if 'months' in step.keys():
        step_num = step['months']
        full_step = relativedelta(months=1)
        curent_step = relativedelta(months=step_num)
        time_counter = 30
    elif 'years' in step.keys():
        step_num = step['years']
        full_step = relativedelta(years=1)
        curent_step = relativedelta(years=step_num)
        time_counter = 364
    else:
        step_num = step['days']
        full_step = relativedelta(days=1)
        curent_step = relativedelta(days=step_num)
        time_counter = 1

        # Инициализация словаря рейтингов (без изменений)
    if agency == 'Expert RA':
        agency_dict = expert_test
    elif agency == 'NCR':
        agency_dict = NCR_test
    elif agency == 'AKRA':
        agency_dict = akra
    elif agency == 'S&P Global Ratings':
        agency_dict = s_and_p
    elif agency == 'Fitch Ratings':
        agency_dict = fitch
    elif agency == "Moody's Interfax Rating Agency":
        agency_dict = moodys
    elif agency == 'NRA':
        agency_dict = nra

    # Инициализация параметров Дирихле
    dirichlet_params = defaultdict(lambda: defaultdict(float))
    observed_transitions = defaultdict(set)  # Для трекинга наблюдаемых переходов

    for rating in agency_dict:
        for k in agency_dict:
            if k == 'D':
                dirichlet_params[rating][k] = ALPHA_DEFAULT
            else:
                dirichlet_params[rating][k] = ALPHA_OTHER if k != rating else ALPHA_SELF

    # Подготовка данных (без изменений)
    data_1 = data.sort_values(col_date)
    set_ogrn = data_1[col_ogrn].unique()
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    # Основной цикл обработки
    for counter, ogrn in enumerate(set_ogrn):
        if pd.isna(ogrn): continue

        my_bar.progress(int(100 * counter / len(set_ogrn)),
                        text=f"Обработка {counter}/{len(set_ogrn)} компаний")

        pr = data_1.loc[data_1[col_ogrn] == ogrn].reset_index(drop=True).sort_values(col_date)
        start_dates = pr[col_date].iloc[0]
        end_dates = f"{pd.to_datetime(pr[col_date].iloc[-1]).year}-12-31"
        current_date = start_dates
        while current_date <= end_dates:
            data_prev = pr[(pr[col_date] < current_date)]
            window_end = (pd.to_datetime(current_date) + curent_step).strftime('%Y-%m-%d')
            window_data = pr[(pr[col_date] >= current_date) & (pr[col_date] < window_end)].reset_index(drop=True)

            # Анализ переходов
            first = ''
            date_start = ''
            if len(window_data) > 0:
                if window_data[col_date].iloc[0] == current_date:
                    last_, date_ = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)
                    first_ = window_data[col_rating][0]

                    if first_ == last_ and last_ is not None:
                        first = first_

                    elif date_ is None and last_ is None:
                        first = first_

                    else:
                        first = last_

                    date_start = start_dates
                else:
                    if len(get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)) > 0:
                        first, date_start = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date,col_rating)  # todo get previous rating on date = start_date

            else:
                if len(get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)) > 0:
                    first, date_start = get_prev_date_raitng(data_prev, ogrn, start_dates, step, col_ogrn, col_date, col_rating)

            transitions = []

            if len(window_data) > 1:
                prev_rating = first
                prev_year = pd.to_datetime(date_start).year
                for _, row in window_data.iterrows():
                        current_rating = row[col_rating]
                        curr_year = pd.to_datetime(row[col_date]).year

                        if pd.to_datetime(row[col_date]) != pd.to_datetime(date_start):
                            if curr_year - prev_year > 1:
                                continue  # Пропускаем большие разрывы во времени
                            else:
                                if prev_rating is not None:
                                    transitions.append((prev_rating, current_rating))
                                    observed_transitions[prev_rating].add(current_rating)  # Фиксация факта перехода

                                prev_rating = current_rating
                            prev_year = curr_year

                # Обновление параметров
                for from_rating, to_rating in transitions:
                    dirichlet_params[from_rating][to_rating] += 1.0

            # Учет отсутствия переходов
            else:
            # if not transitions:
                for rating in agency_dict:
                    dirichlet_params[rating][rating] += ALPHA_SELF ** 2

            current_date = window_end

    # Применение штрафов для ненаблюдаемых переходов
    for from_rating in agency_dict:
        for to_rating in agency_dict:
            if (to_rating not in observed_transitions[from_rating] and
                    to_rating != from_rating and
                    to_rating != 'D'):
                old_rank = agency_dict[from_rating]
                new_rank = agency_dict[to_rating]
                penalty = abs((new_rank - old_rank))
                if penalty == 0:
                    penalty = 1
                dirichlet_params[from_rating][to_rating] *= PENALTY_FACTOR * 1 / penalty

    # Расчет матрицы переходов
    transition_matrix = pd.DataFrame(
        index=agency_dict.keys(),
        columns=agency_dict.keys(),
        dtype=float
    )

    for from_rating in agency_dict:
        row_total = sum(dirichlet_params[from_rating].values())
        if row_total == 0:
            row_total = 1.0  # Защита от деления на ноль

        # Расчет вероятностей
        prob = []
        for to_rating in agency_dict:
            raw_prob = dirichlet_params[from_rating][to_rating] / row_total

            # Дополнительный штраф для редких переходов
            if (dirichlet_params[from_rating][to_rating] < MIN_OBSERVATIONS and
                    to_rating != from_rating and
                    to_rating != 'D'):

                old_rank = agency_dict[from_rating]
                new_rank = agency_dict[to_rating]
                penalty = abs((new_rank - old_rank))

                if penalty == 0:
                    penalty = 1
                raw_prob *= PENALTY_FACTOR * 1 / penalty

            prob.append(raw_prob)

        # Нормализация
        sum_prob = sum(prob)
        if sum_prob > 0:
            prob = [p / sum_prob for p in prob]

        # Запись в матрицу
        for i, to_rating in enumerate(agency_dict):
            transition_matrix.at[from_rating, to_rating] = round(prob[i], 4)

        # Гарантия для поглощающего состояния
        if from_rating == 'D':
            transition_matrix.loc['D'] = 0.0
            transition_matrix.at['D', 'D'] = 1.0

    my_bar.empty()
    st.write("Итоговая матрица с штрафами:")
    st.write(transition_matrix)

    return transition_matrix

def matrix_migration_beta(data: pd.DataFrame, agency: str, start_date: str, end_dates: str, scale: list,
                     step: dict, type_ogrn, type_date, type_rating):
    st.title('Markov process with discrete time and Dirichlet-distr')

    delta = datetime.strptime(end_dates, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")
    agency_dict = {}
    if agency == 'Expert RA':
        agency_dict = expert_test
    if agency == 'NCR':
        agency_dict = NCR_test
    if agency == 'AKRA':
        agency_dict = akra
    if agency == 'S&P Global Ratings':
        agency_dict = s_and_p
    if agency == 'Fitch Ratings':
        agency_dict = fitch
    if agency == "Moody's Interfax Rating Agency":
        agency_dict = moodys
    if agency == 'NRA':
        agency_dict = nra
    check_dict = {}
    pie_cont = {}
    check_ = {}
    full_df = pd.DataFrame()
    method = st.sidebar.radio(
        "Choose discrete or time-cont methods:",
        ("disc", "time-cont")
    )
    if method == 'disc':
        # full_df = calculate_discrete_migr_beta(data, agency, start_date, end_dates, scale, step, type_ogrn, type_date, type_rating)
        start_time = time.perf_counter()

        full_df = calculate_discrete_migr_dirichlet(data, agency, start_date, end_dates, scale, step, type_ogrn, type_date, type_rating)

        end_time = time.perf_counter()

        st.write(f"Время выполнения: {end_time - start_time:.4f} секунд и размер датасета: {len(data)}")

        # full_df_ = calculate_discrete_migr_dirichlet_2_0(data, agency, start_date, end_dates, scale, step, type_ogrn,
        #                                             type_date, type_rating)
    else:
        full_df = calculate_time_cont_migr_beta(data, agency, start_date, end_dates, step, type_ogrn, type_date,
                                               type_rating)

    # full_df.index = ['AAA','AA+', 'AA', 'AA-', 'A+', 'A','A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC', 'CC', 'C', 'D']
    # full_df.columns = ['AAA','AA+', 'AA', 'AA-', 'A+', 'A','A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC', 'CC', 'C', 'D']
    fig = plt.figure(figsize=(10, 10))
    # plt.savefig(f'{directory}/discrete_markov_step={step}.jpg')
    full_df = full_df.apply(pd.to_numeric, errors='coerce')
    plot = sns.heatmap(full_df, annot=True, fmt='.3f', linewidths=.5, annot_kws={"size":9})
    plt.close()
    for col in full_df.columns:  # TODO redact by index
        if full_df[col].sum() > 0:
            fig_ = go.Figure(data=[go.Pie(labels=full_df.index[full_df[col] > 0].tolist(),
                                          values=full_df[col][full_df[col] > 0])])
            fig_.update_layout(
                legend_title=f"{col} rating moved to:",
                font=dict(
                    family="Courier New, monospace",
                    size=15,
                    color="White"
                )
            )
            pie_cont[col] = fig_

    if st.checkbox('Display Pie chart of migration discrete matrix - beta'):
        pies = st.sidebar.multiselect('Choose ratings to see moves', pie_cont)
        for pie_diag in pies:
            st.plotly_chart(pie_cont[pie_diag], theme='streamlit')

    if st.checkbox('Display graph of migration discrete matrix - beta'):
        HtmlFile = open(graph_matric_migration(full_df), 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height=650, width=650)

    if st.checkbox('Display static graph of migration discrete matrix - beta'):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(graph_matric_migration_stat(full_df))

    if st.checkbox('Display migration discrete matrix - beta'):
        # full_df.to_excel(f'discrete_time_step={step}.xlsx')
        st.pyplot(fig)
        fig.savefig((f'results/{agency}/images/discrete_step={step}_{datetime.now().strftime('%Y-%m-%d')}_beta.jpg'))

    if st.checkbox('Display predict discr. migration matrix - beta'):
        fig = plt.figure(figsize=(10, 10))
        n = st.number_input('Enter number', max_value=1000, min_value=2)
        check_1 = np.linalg.matrix_power(full_df, n)
        plot = sns.heatmap(check_1, annot=True, fmt='.3f', linewidths=.5, annot_kws={"size":9})
        # plt.savefig(f'{directory}/time_cont_step={step}_second_avar.jpg')
        st.pyplot(fig)
        fig.savefig((f'results/{agency}/images/predict_{n}_discrete_step={step}_{datetime.now().strftime('%Y-%m-%d')}_beta.jpg'))
        plt.close()

    if st.checkbox('Get predict of migration discrete matrix - beta'):
        # get_state_by_time(data, agency, start_date, scale)
        n = st.number_input('Enter number', max_value=100, min_value=2)
        check_1 = np.linalg.matrix_power(full_df, n)
        st.write(pd.DataFrame(check_1, columns=list(agency_dict.keys()), index=list(agency_dict.keys())))
        check_1 = pd.DataFrame(check_1, columns=list(agency_dict.keys()), index=list(agency_dict.keys()))
        name = f"results/{agency}/discrete_time/predict_discrete_step={step}_predict={n}_{datetime.now().strftime('%Y-%m-%d')}_beta.xlsx"
        check_1.to_excel(name)
        f = Fitter(check_1,
                   distributions=['cauchy', 'chi2', 'expon', 'exponpow', 'gamma', 'lognorm', 'norm', 'powerlaw',
                                  'rayleigh', 'uniform', 'beta'])
        f.fit()
        distr_fit = f.summary()
        st.write(distr_fit)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(f.plot_pdf())
        st.pyplot(f.plot_pdf())

    if st.checkbox('Display distribution of migration discrete matrix - beta'):
        # full_df = get_state_by_time(data, agency, date_to_check, step, scale)[0]
        f = Fitter(full_df,
                   distributions=['cauchy', 'chi2', 'expon', 'exponpow', 'gamma', 'lognorm', 'norm', 'powerlaw',
                                  'rayleigh', 'uniform', 'beta'])
        f.fit()
        distr_fit = f.summary()
        st.write(distr_fit)
        st.pyplot(f.plot_pdf())
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(f.plot_pdf())

    if st.button('Download discrete matrix', key='download_beta'):
        full_df.to_excel(f'results/{agency}/discrete_time/discrete_step={step}_{datetime.now().strftime('%Y-%m-%d')}_beta.xlsx')

@st.cache_data(experimental_allow_widgets=True)
def calculate_migration_matrix_by_series(data: pd.DataFrame, agency: str, start_date: str, end_dates: str, scale: list, step, col_ogrn: str, col_date: str, col_rating: str):
    def calculate_series(ratings):
        """
        Определяет серии изменений рейтинга (положительные, отрицательные и стабильные).
        """
        series = []
        current_series_type = None
        current_series_length = 0

        for i in range(1, len(ratings)):
            if ratings[i] >= ratings[i - 1]:
                change_type = "+"  # Положительная или стабильная серия
            else:
                change_type = "-"  # Отрицательная серия

            if change_type == current_series_type:
                current_series_length += 1
            else:
                if current_series_type is not None:
                    series.append((current_series_type, current_series_length, ratings[i - 1]))
                current_series_type = change_type
                current_series_length = 1

        if current_series_type is not None:
            series.append((current_series_type, current_series_length, ratings[-1]))

        return series

    # Подготовка данных
    data = data.sort_values(col_date)
    agency_data = data[data['agency'] == agency]

    if len(scale) != 0:
        agency_data = agency_data[agency_data['scale'].isin(scale)]

    agency_data = agency_data.sort_values([col_ogrn, col_date])

    unique_ogrns = agency_data[col_ogrn].unique()

    # Матрица переходов и счетчики
    state_transitions = pd.DataFrame(0, index=sorted(agency_data[col_rating].unique()), columns=sorted(agency_data[col_rating].unique()))

    for ogrn in unique_ogrns:
        ogrn_data = agency_data[agency_data[col_ogrn] == ogrn].reset_index(drop=True)
        if len(ogrn_data) < 2:
            continue

        # Рассчитываем серии изменений рейтинга
        series = calculate_series(ogrn_data[col_rating].values)

        # Заполняем матрицу переходов
        for i in range(len(series) - 1):
            _, _, from_state = series[i]
            _, _, to_state = series[i + 1]
            state_transitions.loc[from_state, to_state] += 1

    # Заполняем самопереходы, если нет других переходов
    for state in state_transitions.index:
        if state_transitions.loc[state].sum() == 0:
            state_transitions.loc[state, state] = 1

    # Нормализация строк матрицы, чтобы суммы были равны 1
    transition_matrix = state_transitions.div(state_transitions.sum(axis=1), axis=0).fillna(0)

    st.write("Матрица переходных вероятностей:")
    st.dataframe(transition_matrix)

    return transition_matrix

def migration_matrix_series(data: pd.DataFrame, agency: str, start_date: str, end_dates: str, scale: list,
                     step: dict, type_ogrn, type_date, type_rating):
    st.title('Markov process with discrete time and beta-distr')

    delta = datetime.strptime(end_dates, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")
    agency_dict = {}
    if agency == 'Expert RA':
        agency_dict = expert_test
    if agency == 'NCR':
        agency_dict = NCR_test
    if agency == 'AKRA':
        agency_dict = akra
    if agency == 'S&P Global Ratings':
        agency_dict = s_and_p
    if agency == 'Fitch Ratings':
        agency_dict = fitch
    if agency == "Moody's Interfax Rating Agency":
        agency_dict = moodys
    if agency == 'NRA':
        agency_dict = nra
    check_dict = {} 
    pie_cont = {}
    check_ = {}
    full_df = calculate_migration_matrix_by_series(data, agency, start_date, end_dates, scale, step, type_ogrn, type_date, type_rating)

    # full_df.to_excel(f'{directory}/discrete_markov_step={step}.xlsx')
    n = 0
    name = ''
    # full_df.index = ['AAA','AA+', 'AA', 'AA-', 'A+', 'A','A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC', 'CC', 'C', 'D']
    # full_df.columns = ['AAA','AA+', 'AA', 'AA-', 'A+', 'A','A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC', 'CC', 'C', 'D']
    fig = plt.figure(figsize=(10, 10))
    plot = sns.heatmap(full_df, annot=True, fmt='.3f', linewidths=.5, annot_kws={"size": 9})
    # plt.savefig(f'{directory}/discrete_markov_step={step}.jpg')
    plt.close()
    if st.checkbox('Display migration discrete matrix - series'):
        # full_df.to_excel(f'discrete_time_step={step}.xlsx')
        st.pyplot(fig)
        fig.savefig((f'results/{agency}/images/discrete_step={step}_{datetime.now().strftime('%Y-%m-%d')}_series.jpg'))


def predict_next_ratings(current_ratings, transition_matrix, mode='deterministic'):
    """
    Прогнозирование рейтингов на следующий период по текущим рейтингам и матрице переходов.

    Parameters:
      current_ratings : list-like
          Вектор текущих рейтингов (например, ['AAA', 'AA', 'A', ...]).
          Все значения должны присутствовать в индексах transition_matrix.
      transition_matrix : pandas.DataFrame
          Матрица переходных вероятностей, где строки и столбцы – рейтинги.
          Каждая строка должна суммироваться до 1.
      mode : str, optional
          'deterministic' (по умолчанию) – для каждого объекта выбирается рейтинг с максимальной вероятностью;
          'stochastic' – для каждого объекта выбирается следующий рейтинг случайным образом с учётом вероятностей.

    Returns:
      predicted_ratings : list
          Вектор прогнозируемых рейтингов на следующий период.
    """
    predicted_ratings = []
    # Проходим по каждому текущему рейтингу объекта
    for rating in current_ratings:
        if rating not in transition_matrix.index:
            raise ValueError(f"Rating '{rating}' отсутствует в матрице переходов.")

        # Извлекаем строку матрицы, соответствующую текущему рейтингу
        probs = transition_matrix.loc[rating].values

        if mode == 'deterministic':
            # Детерминированный вариант: выбираем рейтинг с максимальной вероятностью
            next_rating = transition_matrix.columns[np.argmax(probs)]
        elif mode == 'stochastic':
            # Стохастический вариант: сэмплируем следующий рейтинг по распределению probs
            next_rating = np.random.choice(transition_matrix.columns, p=probs)
        else:
            raise ValueError("mode должен быть либо 'deterministic', либо 'stochastic'.")

        predicted_ratings.append(next_rating)

    return predicted_ratings

def metric_quality(data: pd.DataFrame, data_test: pd.DataFrame, agency: str, start_date: str, end_dates: str, scale: list,
                     step: dict, type_ogrn, type_date, type_rating):
    agency_dict = {}
    if agency == 'Expert RA':
        agency_dict = expert_test
    if agency == 'NCR':
        agency_dict = NCR_test
    if agency == 'AKRA':
        agency_dict = akra
    if agency == 'S&P Global Ratings':
        agency_dict = s_and_p
    if agency == 'Fitch Ratings':
        agency_dict = fitch
    if agency == "Moody's Interfax Rating Agency":
        agency_dict = moodys
    if agency == 'NRA':
        agency_dict = nra

    transition_matrix = None
    method = st.radio("Choose method to compare results", ["Discrete", "Time-continous", "Dirichlet", "Basec Bayes"])

    if method == "Time-continous":
        result_full = calculate_time_cont_migr_new(data, agency, start_date, end_dates, step, type_ogrn, type_date, type_rating)
        result_full = fill_empty(result_full, agency_dict)
        result_full_df = pd.DataFrame().from_dict(result_full).fillna(0).reset_index()
        result_full_df = get_generator(sort_df(result_full_df, agency_dict))
        n = st.number_input("choose predicct num", min_value=1, max_value=100)
        result = expm(result_full_df.to_numpy() * n)
        columns_ag = list(agency_dict.keys())
        result = pd.DataFrame(result, columns=columns_ag, index=columns_ag)
        transition_matrix = result

    elif method == "Dirichlet":
        transition_matrix = calculate_discrete_migr_dirichlet_2_0(data, agency, start_date, end_dates, scale, step, type_ogrn,type_date, type_rating)

    # Пусть transition_matrix – DataFrame с матрицей переходов, а agency_dict.keys() – список состояний в том же порядке.

    date_to_check_pred = st.date_input('Choose date to check pred').strftime('%Y-%m-%d')
    date_to_check_fact = st.date_input('Choose date to check fact').strftime('%Y-%m-%d')

    states_for_K_objects_pred = []
    states_for_K_objects_fact = []

    for obj in data[type_ogrn].unique():
        temp_ = data[data[type_ogrn] == obj]
        data_prev = temp_[temp_[type_date] <= date_to_check].sort_values(by=type_ogrn)
        prev_rat, prev_date = get_prev_date_raitng(data_prev, obj, date_to_check_pred, step, type_ogrn, type_date, type_rating)
        if prev_rat is not None:
            states_for_K_objects_pred.append(prev_rat)

    states = list(agency_dict.keys())
    # Подсчитаем частоты
    counter_pred = Counter(states_for_K_objects_pred)
    empirical_distribution_pred = pd.Series({s: counter_pred[s] for s in states})
    empirical_distribution_pred = empirical_distribution_pred / empirical_distribution_pred.sum()

    for obj in data[type_ogrn].unique():
        temp_ = data[data[type_ogrn] == obj]
        data_prev = temp_[temp_[type_date] <= date_to_check].sort_values(by=type_ogrn)
        prev_rat, prev_date = get_prev_date_raitng(data_prev, obj, date_to_check_fact, step, type_ogrn, type_date, type_rating)
        if prev_rat is not None:
            states_for_K_objects_fact.append(prev_rat)

    counter_fact = Counter(states_for_K_objects_fact)
    empirical_distribution_fact = pd.Series({s: counter_fact[s] for s in states})
    empirical_distribution_fact = empirical_distribution_fact / empirical_distribution_fact.sum()

    st.write(transition_matrix)

    st.write("len fact: ", len(states_for_K_objects_fact), "leen pred: ", len(states_for_K_objects_pred))
    if transition_matrix is not None:

        # Прогнозируем распределение на следующий период
        predicted_distribution = empirical_distribution_pred.dot(transition_matrix)

        st.write("Эмпирическое распределение fact:")
        st.write(empirical_distribution_fact)

        st.write("\nПредсказанное распределение на следующий период:")
        st.write(predicted_distribution)

        # Прогнозируем следующие рейтинги (детерминированный подход)
        predicted = predict_next_ratings(states_for_K_objects_fact, transition_matrix, mode='deterministic')
        # st.write("Прогноз (deterministic):", predicted)

        # Прогнозируем следующие рейтинги (стохастический подход)
        predicted_stochastic = predict_next_ratings(states_for_K_objects_fact, transition_matrix, mode='stochastic')
        # st.write("Прогноз (stochastic):", predicted_stochastic)

        from sklearn.metrics import accuracy_score, confusion_matrix

        accuracy = accuracy_score(states_for_K_objects_pred, predicted)
        st.write(f"Accuracy: {accuracy:.4f}")

        cm = confusion_matrix(states_for_K_objects_fact, predicted)
        st.write(cm)

        from sklearn.metrics import accuracy_score, confusion_matrix

        accuracy = accuracy_score(states_for_K_objects_fact, predicted_stochastic)
        st.write(f"Accuracy: {accuracy:.4f}")

        cm = confusion_matrix(states_for_K_objects_fact, predicted_stochastic)
        st.write(cm)

        from sklearn.metrics import recall_score, precision_score
        recall_macro = recall_score(states_for_K_objects_fact, predicted, average='micro')
        st.write("Micr Recall predict:", recall_macro)

        precision_macro = precision_score(states_for_K_objects_fact, predicted_stochastic, average='micro')
        st.write("precision Recall stoch:", precision_macro)

        predicted_ = predict_next_ratings(data[type_rating].values, transition_matrix, mode='deterministic')

        cm = confusion_matrix(data_test[type_rating].values, predicted_)
        st.write(cm)
        recall_micro_ = recall_score(data_test[type_rating].values, predicted, average='micro')
        st.write("Micri Recall predict:", recall_micro_)

# def calculate_second_order(data: pd.DataFrame, agency: str, start_date: str, end_dates: str, step, col_ogrn: str, col_date: str, col_rating: str):
#
#     def create_transition_dataframe(ratings_list):
#         """Создает DataFrame переходов второго порядка с вероятностями."""
#         transitions = defaultdict(list)
#
#         # Формируем пары (S_{t-2}, S_{t-1}) -> S_t
#         for ratings in ratings_list:
#             for i in range(len(ratings) - 2):
#                 pair = (ratings[i], ratings[i + 1])
#                 next_state = ratings[i + 2]
#                 transitions[pair].append(next_state)
#
#         # Создаём список строк для DataFrame
#         data = []
#         for pair, next_states in transitions.items():
#             unique, counts = np.unique(next_states, return_counts=True)
#             total = counts.sum()
#             probabilities = {state: count / total for state, count in zip(unique, counts)}
#             row = {'state_t-2': pair[0], 'state_t-1': pair[1], **probabilities}
#             data.append(row)
#
#         # Создаём DataFrame
#         df = pd.DataFrame(data).fillna(0)
#         return df
#
#     full_step = None
#     curent_step = None
#     time_counter = None
#     step_num = None
#
#     if 'months' in step.keys():
#         step_num = step['months']
#         full_step = relativedelta(months=1)
#         curent_step = relativedelta(months=step_num)
#         time_counter = 30
#     elif 'years' in step.keys():
#         step_num = step['years']
#         full_step = relativedelta(years=1)
#         curent_step = relativedelta(years=step_num)
#         time_counter = 363
#     else:
#         step_num = step['days']
#         full_step = relativedelta(days=1)
#         curent_step = relativedelta(days=step_num)
#         time_counter = 1
#
#     if agency == 'Expert RA':
#         agency_dict = expert_test
#     elif agency == 'NCR':
#         agency_dict = NCR_test
#     elif agency == 'AKRA':
#         agency_dict = akra
#     elif agency == 'S&P Global Ratings':
#         agency_dict = s_and_p
#     elif agency == 'Fitch Ratings':
#         agency_dict = fitch
#     elif agency == "Moody's Interfax Rating Agency":
#         agency_dict = moodys
#     elif agency == 'NRA':
#         agency_dict = nra
#
#
#     # Подготовка данных (без изменений)
#     data_1 = data.sort_values(col_date)
#     set_identifier = data_1[col_ogrn].unique()
#     progress_text = "Operation in progress. Please wait."
#     my_bar = st.progress(0, text=progress_text)
#     values_ = []
#     for counter , obj in enumerate(set_identifier):
#         if pd.isna(obj): continue
#
#         my_bar.progress(int(100 * counter / len(set_identifier)),
#                         text=f"Обработка {counter}/{len(set_identifier)} компаний")
#
#         pr = data_1.loc[data_1[col_ogrn] == obj].reset_index(drop=True).sort_values(col_date)
#
#         values_.append(pr[col_rating].values)
#
#     transition_probs = create_transition_dataframe(values_)
#
#     my_bar.empty()
#
#     return transition_probs


def calculate_second_order(data: pd.DataFrame, agency: str, start_date: str, end_dates: str,
                        step: dict, type_ogrn, type_date, type_rating):
    # Преобразуем DataFrame и добавим уникальные имена колонок
    def make_unique_columns(df):
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [dup + f'_{i}' if i != 0 else dup for i in
                                                             range(sum(cols == dup))]
        df.columns = cols
        return df

    def add_spaces_to_pairs(index_or_columns, agency_dict):
        """Добавляет пробел между состояниями, используя agency_dict."""
        unique_states = sorted(agency_dict.keys())  # Берем все возможные состояния

        def split_pair(pair):
            for state in unique_states:
                if pair.startswith(state):
                    suffix = pair[len(state):]  # Оставшаяся часть после первого состояния
                    if suffix in unique_states:
                        return f"{state} {suffix}"
            return pair  # Если не нашли разделение, возвращаем без изменений

        return index_or_columns.map(split_pair)

    def add_spaces_and_sort(index_or_columns, agency_dict):
        """Добавляет пробел между состояниями и сортирует пары по убыванию, согласно agency_dict."""
        # Создаем список уникальных состояний из agency_dict
        unique_states = sorted(agency_dict.keys(), key=lambda x: agency_dict[x], reverse=True)

        def split_pair(pair):
            """Функция для добавления пробела между состояниями в паре."""
            for state in unique_states:
                if pair.startswith(state):
                    suffix = pair[len(state):]
                    if suffix in unique_states:
                        return f"{state} {suffix}"
            return pair

        # Добавляем пробелы между состояниями в парах
        formatted_pairs = index_or_columns.map(split_pair)

        # Создаем словарь для сортировки на основе порядка в agency_dict
        state_order = {state: i for i, state in enumerate(unique_states)}

        def sorting_key(pair):
            """Ключ для сортировки пар по порядку в agency_dict"""
            first, second = pair.split(" ")
            return (state_order.get(first, float('inf')), state_order.get(second, float('inf')))

        # Сортируем по убыванию, используя сортировку по порядку состояний
        sorted_pairs = sorted(formatted_pairs, key=sorting_key, reverse=True)

        return sorted_pairs


    """Формирует матрицу переходов второго порядка для заданного агентства."""

    if agency == 'Expert RA':
        agency_dict = expert_test
    elif agency == 'NCR':
        agency_dict = NCR_test
    elif agency == 'AKRA':
        agency_dict = akra
    elif agency == 'S&P Global Ratings':
        agency_dict = s_and_p
    elif agency == 'Fitch Ratings':
        agency_dict = fitch
    elif agency == "Moody's Interfax Rating Agency":
        agency_dict = moodys
    elif agency == 'NRA':
        agency_dict = nra

    # Извлекаем все уникальные состояния рейтингов
    # unique_states = sorted(data[type_rating].dropna().unique())
    unique_states = sorted(agency_dict.keys())

    # Генерируем все возможные пары (i, j)
    pairs = [x + y for x in unique_states for y in unique_states]
    pair_to_idx = {pair: i for i, pair in enumerate(pairs)}

    # Словарь для подсчета частот переходов
    transition_counts = defaultdict(lambda: defaultdict(int))

    # Обход по каждой компании (по ОГРН)
    data_1 = data.sort_values(type_date)

    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    set_identifier = data_1[type_ogrn].unique()
    counter = 0
    for ogrn in data[type_ogrn].dropna().unique():
        if pd.isna(ogrn): continue

        counter += 1
        my_bar.progress(int(100 * counter / len(set_identifier)),
                        text=f"Обработка {counter}/{len(set_identifier)} компаний")

        company_data = data[data[type_ogrn] == ogrn].sort_values(by=type_rating)
        ratings = company_data[type_rating].values

        # Подсчет частот переходов (i, j) → (j, z)
        for i in range(len(ratings) - 2):
            pair = ratings[i] + ratings[i + 1]  # (i, j)
            next_pair = ratings[i + 1] + ratings[i + 2]  # (j, z)
            transition_counts[pair][next_pair] += 1

    # Создаем DataFrame для матрицы переходов
    transition_matrix = pd.DataFrame(0.0, index=pairs, columns=pairs)

    # Заполняем матрицу относительными частотами
    for pair, next_states in transition_counts.items():
        total = sum(next_states.values())
        for next_pair, count in next_states.items():
            transition_matrix.loc[pair, next_pair] = count / total

    # Заполняем самопереходы (если строка пустая)
    for pair in pairs:
        # if transition_matrix.loc[pair].sum() == 0:
        if transition_matrix.loc[pair].sum().sum() == 0:
            transition_matrix.loc[pair, pair] = 1.0

        # Добавляем пробелы только между уникальными значениями
    transition_matrix.index = add_spaces_to_pairs(transition_matrix.index, agency_dict)
    transition_matrix.columns = add_spaces_to_pairs(transition_matrix.columns, agency_dict)

    # Переиндексируем DataFrame – это отсортирует и переставит значения в матрице
    transition_matrix = make_unique_columns(transition_matrix)

    # Сортируем индексы и колонки согласно порядку в agency_dict
    # sorted_columns = sorted(transition_matrix.columns, key=lambda x: agency_dict.get(x.split(" ")[0], float('inf')))
    # sorted_index = sorted(transition_matrix.index, key=lambda x: agency_dict.get(x.split(" ")[0], float('inf')))
    # Сортировка по первому и второму состоянию
    sorted_columns = sorted(transition_matrix.columns, key=lambda x: (
        agency_dict.get(x.split(" ")[0], float('inf')),
        agency_dict.get(x.split(" ")[1], float('inf'))
    ))
    sorted_index = sorted(transition_matrix.index, key=lambda x: (
        agency_dict.get(x.split(" ")[0], float('inf')),
        agency_dict.get(x.split(" ")[1], float('inf'))
    ))
    # Применяем сортировку индексов и колонок
    transition_matrix = transition_matrix.loc[sorted_index, sorted_columns]
    # Теперь сортируем индексы и колонки в порядке убывания
    # transition_matrix = transition_matrix.loc[transition_matrix.index].loc[:, transition_matrix.columns]

    # Сортируем индексы и колонки для симметрии
    # transition_matrix = make_unique_columns(transition_matrix)

    st.write(transition_matrix)
    print(transition_matrix)

def matrix_second_order(data: pd.DataFrame, agency: str, start_date: str, end_dates: str, scale: list,
                     step: dict, type_ogrn, type_date, type_rating):

    st.title('Markov process of second order for factual transitions')

    full_dict = calculate_second_order(data, agency, start_date, end_dates, step, type_ogrn, type_date, type_rating)

    st.write(full_dict)

def compare_methods(data: pd.DataFrame, agency: str, start_date: str, end_dates: str, scale: list,
                     step: dict, type_ogrn, type_date, type_rating):

    st.title('Markov process of second order for factual transitions')

    agency_dict = {}
    if agency == 'Expert RA':
        agency_dict = expert_test
    if agency == 'NCR':
        agency_dict = NCR_test
    if agency == 'AKRA':
        agency_dict = akra
    if agency == 'S&P Global Ratings':
        agency_dict = s_and_p
    if agency == 'Fitch Ratings':
        agency_dict = fitch
    if agency == "Moody's Interfax Rating Agency":
        agency_dict = moodys
    if agency == 'NRA':
        agency_dict = nra

    methods = ["Time-continous", "Dirichlet", "Discrete"]
    method1 = st.selectbox("Выберите первый метод", methods)
    method2 = st.selectbox("Выберите второй метод", methods)

    transitions_matrix1 = None
    transitions_matrix2 = None

    if method1 == "Time-continous":
        result_full = calculate_time_cont_migr_new(data, agency, start_date, end_dates, step, type_ogrn, type_date, type_rating)
        result_full = fill_empty(result_full, agency_dict)
        result_full_df = pd.DataFrame().from_dict(result_full).fillna(0).reset_index()
        result_full_df = get_generator(sort_df(result_full_df, agency_dict))
        result = expm(result_full_df.to_numpy())
        columns_ag = list(agency_dict.keys())
        result = pd.DataFrame(result, columns=columns_ag, index=columns_ag)
        transitions_matrix1 = result

    elif method1 == "Dirichlet":
        transitions_matrix1 = calculate_discrete_migr_dirichlet(data, agency, start_date, end_dates, scale, step,
                                                                  type_ogrn, type_date, type_rating)

    if method2 == "Time-continous":
        result_full = calculate_time_cont_migr_new(data, agency, start_date, end_dates, step, type_ogrn, type_date,type_rating)
        result_full = fill_empty(result_full, agency_dict)
        result_full_df = pd.DataFrame().from_dict(result_full).fillna(0).reset_index()
        result_full_df = get_generator(sort_df(result_full_df, agency_dict))
        result = expm(result_full_df.to_numpy())
        columns_ag = list(agency_dict.keys())
        result = pd.DataFrame(result, columns=columns_ag, index=columns_ag)
        transitions_matrix2 = result

    elif method2 == "Dirichlet":
        transitions_matrix2 = calculate_discrete_migr_dirichlet(data, agency, start_date, end_dates, scale, step,
                                                               type_ogrn, type_date, type_rating)

    if transitions_matrix1 is not None and transitions_matrix2 is not None:
        matrix1_values = transitions_matrix1.to_numpy().flatten() * 100
        matrix2_values = transitions_matrix2.to_numpy().flatten() * 100

        rmse = np.sqrt(mean_squared_error(matrix1_values, matrix2_values))
        mae = mean_absolute_error(matrix1_values, matrix2_values)

        st.write("RMSE: ", rmse, "MAE:", mae)

        # Извлекаем главные диагонали из обеих матриц переходных вероятностей
        matrix1_diag = np.diag(transitions_matrix1.to_numpy()) * 100
        matrix2_diag = np.diag(transitions_matrix2.to_numpy()) * 100

        # Вычисляем RMSE и MAE только для главной диагонали
        rmse_diag = np.sqrt(mean_squared_error(matrix1_diag, matrix2_diag))
        mae_diag = mean_absolute_error(matrix1_diag, matrix2_diag)

        # Выводим результаты
        st.write("RMSE для главной диагонали: ", rmse_diag)
        st.write("MAE для главной диагонали:", mae_diag)


if __name__ == '__main__':
    # Рейтинги кредитоспособности банков
    # input_data = pd.read_excel("merged data set.xlsx", sheet_name='otput')
    # _data = pd.read_excel("merged data set.xlsx", sheet_name='Лист1')
    # data_check = pd.concat([_data['Название региона'], _data['Код региона (Минфин РФ)']], axis=1)
    directory = f'temp_{datetime.now()}'  # todo use path to create local directory
    # os.mkdir(directory)
    # data_1 = pd.read_excel('Output_without_dubl.xlsx')
    data = load_data(upload)
    type_agency = st.sidebar.selectbox("Choose agency column", data.columns)
    agency_dict_group = {'Expert RA': group_expert, 'AKRA' : {}, 'NRA' : {}, 'NCR' : {}}
    type_date = st.sidebar.selectbox("Choose date column", data.columns)
    type_ogrn = st.sidebar.selectbox("Choose ogrn column", data.columns)
    type_rating = st.sidebar.selectbox("Choose rating column", data.columns)
    type_field = st.sidebar.selectbox("Choose type column", data.columns)
    scale = st.sidebar.selectbox("Choose scale column", data.columns)

    agency = st.sidebar.selectbox("Choose one agency to check", data[type_agency].unique())

    data = data[data[type_agency] == agency]

    _ro_type = st.sidebar.multiselect('Choose type of companies', data[type_field].unique())
    data['Groupped_ratings'] = data[type_rating].map(agency_dict_group[agency])
    type_rating_group = st.sidebar.selectbox("Choose grouped rating column (if needed)", data.columns)

    #TODO choose there to add group by method (column)
    if type_rating_group is not None:
        type_rating = type_rating_group

    #TODO choose there to add field of companies
    if len(_ro_type) != 0 and type_field is not None:
        data = data[data[type_field].isin(_ro_type)]

    # valid_keys = list(expert_test.keys())
    data[type_date] = pd.to_datetime(data[type_date]).dt.strftime('%Y-%m-%d')

    # data = data[data[type_rating].isin(valid_keys)]
    data = data.sort_values(type_date).reset_index().drop(columns=['index'])
    # data, data_test = train_test_split(data, test_size=0.3, random_state=42)
    data = data.reset_index(drop=True)
    # data_test = data_test.reset_index(drop=True)
    st.write(data)
    start_date = data[type_date][0]
    end_date = data[type_date][len(data[type_date]) - 1]
    st.write(start_date, end_date)
    scale = st.sidebar.multiselect('Choose scale (Be careful, do not choose different scales)',
                                   data[data[type_agency] == agency][scale].unique())
    # TODO here matrix_migration is year-matrix migration
    step_type = st.sidebar.selectbox('Choose type of step', ['months', 'years', 'days'])
    step_ = None
    if step_type == 'months':
        step_ = st.sidebar.number_input(f'Enter step in months', min_value=1, max_value=36)
    elif step_type == 'years':
        step_ = st.sidebar.number_input('Enter step in years', min_value=1, max_value=12)
    else:
        step_ = st.sidebar.number_input('Enter step in years', min_value=1, max_value=365)

    step = {step_type: step_}
    date_to_check = st.sidebar.date_input('Choose date to check').strftime('%Y-%m-%d')

    if st.sidebar.checkbox('Markov process with discrete time'):
        matrix_migration(data, agency, start_date, end_date, scale, step, date_to_check, directory, type_ogrn, type_date, type_rating)

    if st.sidebar.checkbox('Markov process with continous time'):
        time_cont(data, agency, start_date, end_date, step, directory, type_ogrn, type_date, type_rating)

    if st.sidebar.checkbox('Markov process with Wald method'):
        wald_migration(data, agency, start_date, end_date, scale, step, type_ogrn, type_date, type_rating)

    if st.sidebar.checkbox('Markov process with beta - distribution'):
        matrix_migration_beta(data, agency, start_date, end_date, scale, step, type_ogrn, type_date, type_rating)

    if st.sidebar.checkbox('Markov process with series - length'):
        migration_matrix_series(data, agency, start_date, end_date, scale, step, type_ogrn, type_date, type_rating)

    if st.sidebar.checkbox('Markov model second order'):
        matrix_second_order(data, agency, start_date, end_date, scale, step, type_ogrn, type_date, type_rating)

    if st.sidebar.checkbox('Compare results by quality'):
        data, data_test = train_test_split(data, test_size=0.3, random_state=42)
        data = data.reset_index(drop=True)
        data_test = data_test.reset_index(drop=True)
        metric_quality(data, data_test, agency, start_date, end_date, scale, step, type_ogrn, type_date, type_rating)

    if st.sidebar.checkbox('Compare results'):
        n = st.number_input('Write number of each predict you want to do', min_value= 1, max_value = 1000)
        get_state_by_time(data, agency, date_to_check, step, scale, type_ogrn, type_date, type_rating, n)

    if st.sidebar.checkbox('Compare methods'):
        compare_methods(data, agency, start_date, end_date, scale, step, type_ogrn, type_date, type_rating)

    if st.sidebar.checkbox('Get MSE, RMSE, R ^ 2'):
        file_fact = st.file_uploader('Choose file to verify with the same step and num')
        file_pred = st.file_uploader('Choose file to predict with the same step and num')
        # file_Bays = st.file_uploader('Choose file to predict with the same step and num (Bayes)')
        predict = pd.read_excel(file_pred)
        fact = pd.read_excel(file_fact)
        # Bays = pd.read_excel(file_Bays)
        if 'Unnamed: 0' in predict.columns:
            predict = predict.set_index('Unnamed: 0')
        if 'Unnamed: 0' in fact.columns:
            fact = fact.set_index('Unnamed: 0')
        # if 'Unnamed: 0' in Bays.columns:
        #     Bays = Bays.set_index('Unnamed: 0')
        MSE = 0.0
        R_square = 0.0
        RMSE = 0.0

        fact_d = {}
        pred_d = {}
        i = 1
        MSE_1 = {'Дискретный': 32.295, 'Непрерывный': 7.02582, 'Байесовский': 17.6962}   #for presentation
        # MSE_1 = {'Дискретный-Непрерывный': 7.69, 'Байес-Непрерывный': 6.02, 'Дискретный-Байес': 1.27}
        # MSE_2 = {'Дискретный-Непрерывный': 8.27, 'Байес-Непрерывный': 11, 'Дискретный-Байес': 1.329}
        # MSE_3 = {'Дискретный-Непрерывный': 1.78, 'Байес-Непрерывный': 1.49, 'Дискретный-Байес': 1.196}
        for row in fact.index:
            j = 1
            for col in fact.columns:
                # if f'{i}_{j}' not in fact_d:
                #     fact_d[f'{i}_{j}'] = []
                # if f'{i}_{j}' not in pred_d:
                #     pred_d[f'{i}_{j}'] = []
                # fact_d[f'{i}_{j}'].append(fact.loc[row].at[col])
                # pred_d[f'{i}_{j}'].append(predict.loc[row].at[col])
                MSE += (fact.loc[row].at[col] * 100 - predict.loc[row].at[col] * 100) ** 2
                j += 1
            i += 1

        MSE = MSE / (len(fact.index) * len(fact.columns))
        RMSE = pow(MSE, 1 / 2)

        st.write('MSE=',MSE, 'RMSE=',RMSE)
        # fact_ = fact.to_numpy()
        predict_ = predict.to_numpy()
        fig = plt.figure(figsize=(15, 15))
        # plot = sns.heatmap(r, annot=True, fmt='.3f', linewidths=.5, annot_kws={"size": 9})
        if st.button('Plot curve'):
            lists = sorted(fact_d.items())  # sorted by key, return a list of tuples

            x, y = zip(*lists)  # unpack a list of pairs into two tuples

            # plt.plot(x, y)
            plot_fact = plt.plot(x, y)

            lists_pred = sorted(pred_d.items())  # sorted by key, return a list of tuples

            x_pred, y_pred = zip(*lists_pred)  # unpack a list of pairs into two tuples

            # plt.plot(x, y)
            plot_pred = plt.plot(x_pred, y_pred)
            # predict.plot()
            plt.ylabel("Вероятность перехода", fontdict={'size':20})
            plt.xlabel("Перемещение по рейтингам", fontdict={'size':20})
            plt.legend(["Фактическое перемещение", "Спрогнозированное перемещение"], loc="upper right", fontsize="20")

        if st.button('Plot bar'):
            # df_1 = pd.DataFrame(MSE_1, index=MSE_1.keys())
            # df_2 = pd.DataFrame(MSE_2, index=MSE_2.keys())
            # df_3 = pd.DataFrame(MSE_3, index=MSE_3.keys())
            # df_full = pd.concat([df_1, df_2, df_3], axis=0)
            barWidth = 0.5
            br1 = np.arange(len(MSE_1))
            br2 = [x + barWidth for x in br1]
            br3 = [x + barWidth for x in br2]
            plt.bar(br1, list(MSE_1.values()), color='b', width=barWidth,
                    edgecolor='grey', label='КРА 1')
            # plt.bar(br2, list(MSE_2.values()), color='g', width=barWidth,
            #         edgecolor='grey', label='КРА 2')
            # plt.bar(br3, list(MSE_3.values()), color='b', width=barWidth,
            #         edgecolor='grey', label='КРА 3')
            plt.xlabel('методы', fontweight='bold', fontsize=15)
            plt.ylabel('MSE', fontweight='bold', fontsize=15)
            plt.xticks([r for r in range(len(MSE_1))],
                       ['Дискретный','Непрерывный', 'Байесовский'], fontsize=14)

            plt.yticks(fontsize=14)
            plt.legend(loc="upper right", fontsize="20")
            # plt.bar(*zip(*MSE_1.items()))
            # plt.bar(*zip(*MSE_2.items()))
            # plt.bar(*zip(*MSE_3.items()))
            # plt.bar(df_full)
            # plt.show()
            # fig.savefig(f'results/images/_{datetime.now().strftime('%Y-%m-%d')}.jpg')
            st.pyplot(fig)
        # plt.savefig(f'{directory}/time_cont_step={step}_second_avar.jpg')
        # plt.close()


    if st.sidebar.checkbox('Markov process with Aalen-Johansen metric'):
        aalen_johansen_metric(data, agency, start_date, end_date, step)

    if st.sidebar.checkbox('Markov process with Bayesian metric'):
        Bayes_migration(data, agency, start_date, end_date, step, type_ogrn, type_date, type_rating)
    # convert_ratings(data, agency)
    stop = 0

    # functions for correct data from input file
    #
    # print(input_data)
    # print(first_step_comp(input_data))    #first step of comparing data
    # print(second_step_comp(input_data))   #second step of comparing data
    # print(third_step_comp(input_data))    #third step of comapring data
    # first_try(data_trouble)       #searching unique objects of ogrn/_name
    # input_ogrn(data_trouble, data_['inn_'], data_['ogrn_'])
    # correct_data(input_data, data_check)  #function to updated regions and cities by official mapper
    # convert_ratings(data, 'Expert RA')    #function to convert old ratings to new