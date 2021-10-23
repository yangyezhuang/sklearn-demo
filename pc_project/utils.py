import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    origin_data = pd.read_csv(path)
    data = origin_data[['开始时间', '风险值', '地区名', '销售额']]
    data['销售额'] = data['销售额'].astype(np.int32)
    data_del = pd.get_dummies(data, columns=['地区名'])
    data_del['month'] = data_del['开始时间'].map(lambda t: int(t.split('-')[1]))
    data_del = data_del.drop('开始时间', axis=1)
    return data_del


def get_risk_label(num):
    arr = pd.read_csv('type.csv', nrows=1, header=None)
    arr = arr.values[0]
    if num <= arr[0]:
        return {"risk": num, "label": "low risk"}
    elif num > arr[0] and num <= arr[1]:
        return {"risk": num, "label": "normal risk"}
    elif num > arr[1]:
        return {"risk": num, "label": "high risk"}


def get_score_label(num):
    arr = pd.read_csv('type.csv', nrows=2, header=None, skiprows=1)
    arr = arr.values[0]
    if num <= arr[0]:
        return {"risk": num, "label": "low score"}
    elif num > arr[0] and num <= arr[1]:
        return {"risk": num, "label": "normal score"}
    elif num > arr[1]:
        return {"risk": num, "label": "high score"}


if __name__ == '__main__':
    # r=load_data('../data/per_month_sale_and_risk.csv')
    # print(r)
    r = get_risk_label(0.5)
    print(r)
