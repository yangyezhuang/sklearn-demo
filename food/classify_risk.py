# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import ensemble, tree
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier


def RiskClassisfication():
    '''
    风险预测，依据时间、月售、风险值
    '''
    # Step 1. 加载原始数据
    food = pd.read_csv(r'../data/per_month_sale_and_risk.csv')

    # Step 2. 数据预处理
    # Step 2.1 把对属性start_time和place进行编码
    start_time = pd.get_dummies(food.开始时间)
    place = pd.get_dummies(food.地区名)

    # 地区+月份+销售额
    data = pd.concat([place, start_time], axis=1)
    data = pd.concat([data, food['销售额']], axis=1)
    # 地区
    # data = place
    # 月份
    # data = start_time
    # 销售额
    # data = food['销售额'].to_frame()
    # 地区+月份
    # data = pd.concat([place,start_time],axis=1)
    # 地区+销售额
    # data = pd.concat([place,food['销售额']],axis=1)
    # 月份+销售额
    # data = pd.concat([start_time,food['销售额']],axis=1)

    # 分类 风险值大于0.7为高风险，数值为2，在0.7-0.4为中风险，数值为1，小于0.4为低风险，数值为0
    riskClass = pd.DataFrame(np.ones(len(food)), columns=['riskClass'])
    riskClass['riskClass'][food['风险值'] > 0.7] = 2
    riskClass['riskClass'][food['风险值'] < 0.3] = 0

    # Step 2.2 数据归一化处理
    data = (data - data.min(0)) / (data.max(0) - data.min(0))

    # Step 3.划分训练集和验证集

    treeDecision = tree.DecisionTreeClassifier()
    # AdaBoost训练与预测
    ada = OneVsRestClassifier(ensemble.AdaBoostClassifier())

    l = len(data)
    # 五折交叉
    for i in range(5):
        test_index = [j for j in range(i, l, 5)]
        all_index = [j for j in range(l)]
        train_index = list(set(all_index).difference(set(test_index)))

        # 验证集
        test_x = data.loc[test_index]
        test_y = riskClass.riskClass.loc[test_index]
        # 训练集
        train_x = data.loc[train_index]
        train_y = riskClass.riskClass.loc[train_index]

        # Step 4. 训练与预测
        # 4.1 决策树训练与预测 
        # 构建模型(决策树)
        tdModel = treeDecision.fit(train_x, train_y)  # 模型训练

        pre_tree = tdModel.predict(test_x)  # 预测，结果存入pre中
        print('使用DecisionTree模型的结果:')  # 给出统计结果
        print(classification_report(test_y, pre_tree, target_names=['calss0', 'calss1', 'calss2']))

        # 4.2 AdaBoost训练与预测
        # ada = OneVsRestClassifier(ensemble.AdaBoostClassifier())
        ada.fit(train_x, train_y)

        pre_ada = ada.predict(test_x)
        print('使用AdaBoost的模型:')
        print(classification_report(test_y, pre_ada, target_names=['calss0', 'calss1', 'calss2']))

    return None


if (__name__ == "__main__"):
    food_pre = RiskClassisfication()
    # print(food_pre)

    # Riskclassification()
