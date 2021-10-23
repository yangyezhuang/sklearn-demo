# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def divide_x_y(x, y, cv, n):
    '''
    将训练集均匀的分开
    input:
        x 特征向量
        y label
        cv 几折，也是step
        n 几折在cv的范围内
    output：
        train_x
        test_x
        train_y 和原始格式一样
        test_y
    '''
    l = len(x)
    test_index = [j for j in range(n, l, cv)]
    all_index = [j for j in range(l)]
    train_index = list(set(all_index).difference(set(test_index)))  # 加了一个listQAQ,set返回的不是一个list
    test_x = x[test_index]
    test_y = y[test_index]
    train_x = x[train_index]
    train_y = y[train_index]
    return train_x, train_y, test_x, test_y


# load data
food_store = pd.read_csv(r'.\food_store.csv')
food_record = pd.read_csv(r'.\food_record.csv')
# 处理数据，把start_time和place编码
food = pd.merge(food_record, food_store)
start_time = pd.get_dummies(food.开始时间)
place = pd.get_dummies(food.地区名)

data = np.hstack([start_time, place])
month_sell = np.array(food['月售'])
month_sell = month_sell.reshape([len(month_sell), 1])
data = np.hstack([data, month_sell])
label = np.array(food.风险值)
# 最值归一化
data = (data - data.min(0)) / (data.max(0) - data.min(0))
# 分出训练集和测试集,其中索引1，6，11...的是测试集
train_x, train_y, test_x, test_y = divide_x_y(data, label, 5, 1)
# 预测结果
from sklearn import linear_model

reg = linear_model.LinearRegression()
model = reg.fit(train_x, train_y)
pre = model.predict(test_x)
MSE = np.sum(np.square(pre - test_y)) / len(test_y)
RMSE = MSE ** 0.5
