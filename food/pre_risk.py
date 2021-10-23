# -*- coding: utf-8 -*-

import pandas as pd
# import numpy as np
from sklearn import linear_model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# import seaborn as sns


def loadDataset():
    # Step 1. 加载原始数据
    food_record = pd.read_csv(r'.\per_month_sale_and_risk.csv', encoding='utf8')
    return food_record


def RiskPrediction(store_id):
    '''
    风险预测，依据时间、月售、风险值
        store_id: 店家ID
    '''
    # Step 1. 加载原始数据
    data_all = loadDataset()

    # 根据ID取指定店辅数据
    food = data_all.loc[data_all['店铺ID'] == store_id, ['店铺ID', '店铺名', '开始时间', '风险值', '地区名']]

    # Step 2. 数据预处理
    # Step 2.1 对属性start_time和place进行编码

    start_time = pd.get_dummies(food.开始时间)
    # place = pd.get_dummies( food.地区名 )

    # data = pd.concat([place,start_time,food['月售']],axis=1)    
    data = pd.concat([start_time], axis=1)

    # Step 2.2 数据归一化处理
    # data = ( data-data.min(0) )/(data.max(0)-data.min(0))

    # Step 3.划分训练集和验证集 
    l = len(data)
    test_index = [j for j in range(0, l, 5)]
    all_index = [j for j in range(l)]
    train_index = list(set(all_index).difference(set(test_index)))

    # 验证集
    test_x = data.loc[test_index]
    test_y = food.风险值.loc[test_index]
    # 训练集
    train_x = data.loc[train_index]
    train_y = food.风险值.loc[train_index]

    # Step 4. 训练与预测    
    reg = linear_model.LinearRegression()  # 构建模型(线性回归)
    model = reg.fit(train_x, train_y)  # 模型训练

    pre = model.predict(test_x)  # 预测，结果存入pre中

    # 保存在pre中的预测结果，进一步处理，以便理解预测结果。处理结果存入food_pre中。
    food_pre = food.loc[test_index]
    # food_pre = food_pre.drop(columns=['风险值'])               #去掉未编码前测试集
    food_pre = food_pre.reset_index(drop=True)  # 重置索引为0-14076
    food_pre = food_pre.join(pd.DataFrame({'预测风险值': pre}))  # 预测结果拼上去

    return food_pre


def Riskclassification():
    '''
    聚类(无监督, 依据风险和收入)
    '''
    # 1. 数据加载
    food = loadDataset()
    data = pd.DataFrame(food, columns=['风险值', '月售'])
    # 2. 归一化
    data = (data - data.min(0)) / (data.max(0) - data.min(0))

    # 2. 构建模型(KMeans)
    kmeans = KMeans(n_clusters=3)

    # 3. 模型训练
    kmeans.fit(data)

    # 4. 聚类(结果 0 1 2)
    y_kmeans = kmeans.predict(data)
    print(y_kmeans)

    # 5. 可视化－－画图
    # sns.set()  
    # cValue = ['r','y','g','b','r','y','g','b','r'] 

    # matplotlib画图中中文显示会有问题，需要这两行设置默认字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.scatter(data['风险值'], data['月售'], c=y_kmeans, s=0.5, alpha=1)
    plt.xlabel('risk valune')
    plt.ylabel('Monthly Sales')
    # plt.colorbar()
    plt.show()


if (__name__ == "__main__"):
    # 店铺ID
    store_test = "3d8ab846-7725-11ea-98c9-525400c74cae"
    # 风险预测
    food_pre = RiskPrediction(store_test)
    print(food_pre)  # 结果显示
    food_pre.to_csv("risk_pre.csv", index=False, sep=',')  # 写入文件

    # 绘图
    Riskclassification()
