import pandas as pd
from tools import Executor,  SqlCreatorAlpha
import matplotlib.pyplot as plt



def get_num_comp(agency, test):
    # test = executor.execute(SqlCreatorAlpha().from_raw(sql)).as_dataframe
    # test = test.drop(index=[8882])
    print(test)
    stop = 0
    result = pd.DataFrame()
    test = test[test['agency'] == agency]
    test = test.reset_index(drop=True)
    test['_date'] = pd.to_datetime(test['_date'])
    test['year'] = test['_date'].dt.year
    comp_year = []
    for i in test['_date'].unique():
        print(len(test[test['_date'] == i]))
        comp_year.append(len(test[test['_date'] == i]['identifier'].unique()))

    pd.DataFrame(comp_year, columns=['Num of comp'], index=test['_date'].sort_values().unique()).plot()

if __name__ == '__main__':
    executor = Executor(host='10.1.8.38', database='dev', user='developer', password='developer', port=5432)
    sql = f"""
            select * from parser_cb.level_2
        """
    print(sql)
    test = executor.execute(SqlCreatorAlpha().from_raw(sql)).as_dataframe
    test = test.drop(index=[8882])
    print(test)
    stop = 0
    result = pd.DataFrame()
    test = test[test['agency'] == 'АО "Эксперт РА"']
    test = test.reset_index(drop=True)
    test['_date'] = pd.to_datetime(test['_date'])
    test['year'] = test['_date'].dt.year
    comp_year = []
    for i in range(2017, 2025):
        print(len(test[test['_date'] == i]))
        comp_year.append(len(test[test['_date'] == i]['identifier'].unique()))

    pd.DataFrame(comp_year, columns=['Num of comp'], index=range(2017, 2025)).plot()

    for i in test.index:
        if 'ru' in test['rating'][i] or 'Рейтинг отозван' in test['rating'][i] or 'D' in test['rating'][i]:
            result = pd.concat([result, test.iloc[[i]]], axis=0)

    result = result.reset_index(drop=True)
    ratings = ['ruAAA', 'ruAA+', 'ruAA', 'ruAA-', 'ruA+', 'ruA', 'ruA-', 'ruBBB+', 'ruBBB', 'ruBBB-', 'ruBB+', 'ruBB',
               'ruBB-', 'ruB+', 'ruB', 'ruB-', 'ruCCC', 'ruCC', 'ruC', 'ruD', 'Рейтинг отозван']

    result['_date'] = pd.to_datetime(result['_date'])
    result['year'] = result['_date'].dt.year
    pivot_df = result.pivot_table(index='rating', columns='year', aggfunc='size', fill_value=0).reindex(ratings,
                                                                                                        fill_value=0)
    pivot_df.plot()
    comp_year = []
    for i in range(2017, 2025):
        print(len(result[result['year'] == i]))
        comp_year.append(len(result[result['year'] == i]))
    pd.DataFrame(comp_year, columns=['Num of comp'], index=range(2017, 2025)).plot(kind='bar')
    stop = 0
    comp_year_def = []
    for i in range(2017, 2025):
        print(len(result[result['year'] == i]))
        year_df = result[result['year'] == i]
        if len(year_df[year_df['rating'] == 'D']) != 0:
            counter_D = year_df[year_df['rating'] == 'D']['rating'].values.tolist()
            print(counter_D)
            comp_year_def.append(counter_D.count('D'))
        else:
            comp_year_def.append(0)

    pd.DataFrame(comp_year_def, columns=['Num of rat D'], index=range(2017, 2025)).plot(kind='bar')
    stop = 0