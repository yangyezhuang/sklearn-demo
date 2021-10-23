# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入均方误差

"""
*******波士顿房价数据集解读及预测--回归模型应用的实例*******

  波士顿房价数据集（Boston House Price Dataset），可使用sklearn.datasets.load_boston()即可加载相关数据。
  该数据集是一个回归问题。每个类的观察值数量是均等的，共有 506 个观察，13 个输入变量和1个输出变量。

  每条数据包含房屋以及房屋周围的详细信息。其中包含城镇犯罪率，一氧化氮浓度，住宅平均房间数，
到中心区域的加权距离以及自住房平均房价等等，共13个特征：
1    CRIM：城镇人均犯罪率。
2    ZN：住宅用地超过 25000 sq.ft. 的比例。
3    INDUS：城镇非零售商用土地的比例。
4    CHAS：查理斯河空变量（如果边界是河流，则为1；否则为0）。
5    NOX：一氧化氮浓度。
6    RM：住宅平均房间数。
7    AGE：1940 年之前建成的自用房屋比例。
8    DIS：到波士顿五个中心区域的加权距离。
9    RAD：辐射性公路的接近指数。
10    TAX：每 10000 美元的全值财产税率。
11    PTRATIO：城镇师生比例。
12    B：1000（Bk-0.63）^ 2，其中 Bk 指代城镇中黑人的比例。
13    LSTAT：人口中地位低下者的比例。

14*    MEDV：自住房的平均房价，以千美元计。    
    预测平均值的基准性能的均方根误差（RMSE）是约 9.21 千美元。
    
load_boston()重要参数：
    return_X_y:表示是否返回target（即价格），默认为False，只返回data（即属性）。
    
    模型：采用AdaBoost回归、决策树回归、KNN回归、线性回归模型LinearRegression和SGDRegressor分析
"""

# 1、加载数据，输出特征等
# (1)测试1
boston = load_boston()
print(boston.data.shape)
# (506L, 13L)

# (2)测试2
data, target = load_boston(return_X_y=True)
print(data.shape)
# (506L, 13L)
print(target.shape)
# (506L,)

# 开始
data = load_boston()

print('feature_names:', data['feature_names'])
print(data['DESCR'])
# 从输出结果来看，该数据共有506条波士顿房价的数据，每条数据包括对指定房屋的13项数值型特征和目标房价
# 此外，该数据中没有缺失的属性/特征值。

# 2、将数据分割成训练集和测试集
# 准备训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(
    data.data, data.target, test_size=0.3, random_state=33)

# 分析回归目标值的差异
print('The max target value is ', np.max(data.target))
print('The min target value is ', np.min(data.target))
print('The average target value is ', np.mean(data.target))

# 3、训练数据和测试数据标准化处理
# 分别创建初始化对特征值和目标值的标准化器
ss_X = StandardScaler()
ss_y = StandardScaler()

# 因训练数据都是数值型，要标准化处理。
X_train = ss_X.fit_transform(train_x)
X_test = ss_X.transform(test_x)
# 目标数据（房价预测值）也是数值型，所以也要标准化处理。
# 注：fit_transform()与transform()要求操作2D数据，而此时的y_train与y_test都是1D的，
# 因此需要调用reshape(-1,1)，例如：[1,2,3]变成[[1],[2],[3]]
y_train = ss_y.fit_transform(train_y.reshape(-1, 1))
y_test = ss_y.transform(test_y.reshape(-1, 1))

# 3、创建AdaBoost回归模型，得到房价预测结果(数据没有标准化处理)

Ada_regressor = AdaBoostRegressor()
# fit(X_train,y_train)引发警告：“DataConversionWarning: A column-vector y was passed when a 1d array was expected.”问题解决：y_train.ravel().
Ada_regressor.fit(X_train, y_train.ravel())

pred_Ada_y = Ada_regressor.predict(X_test)
# print('房价预测结果',pred_Ada_y)

# 4、性能测评
# 主要是判断预测值与真实值之间的差距，比较直观的评价指标有：
# (1)平均绝对值误差MAE(mean absolute error)
# (2)均方误差MSE(mean squared error)：将预测结果与实际结果对比，求均方误差。
# (3)R-squared评价函数

# 使用AdaBoost模型自带的评估模块，并输出评估结果:
print('the value of default measurement of AdaBoost：',
      Ada_regressor.score(X_test, y_test))
print('the value of R-squared of AdaBoost is', r2_score(y_test, pred_Ada_y))

# 使用标准化器中的inverse_transform函数还原转换前的真实值,求MSE。
print('the MSE of AdaBoost is', mean_squared_error(
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(pred_Ada_y)))
print('the MAE of AdaBoost is', mean_absolute_error(
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(pred_Ada_y)))

# mse = mean_squared_error(y_test,pred_Ada_y)
# print('\nAdaBoost均方误差 = ',round(mse,2))   #返回浮点数的四舍五入值


# 5、使用决策树回归、KNN回归、线性回归模型LinearRegression和SGDRegressor分析数据集，对比结果
# (1) DecisionTree回归(数据使用原始数据，未标准化)
dec_regressor = DecisionTreeRegressor()
dec_regressor.fit(train_x, train_y)
pred_dec_y = dec_regressor.predict(test_x)

mse_dec = mean_squared_error(test_y, pred_dec_y)
# print('决策树均方误差=',round(mse_dec,2))
# 决策树均方误差= 24.13

# 使用决策树回归模型自带的评估模块，并输出评估结果
print('the value of default measurement of DecisionTree：',
      dec_regressor.score(test_x, test_y))
print('the value of R-squared of DecisionTree is', r2_score(test_y, pred_dec_y))

# 可以使用标准化器中的inverse_transform函数还原转换前的真实值，这里没有使用标准化后的数据。
print('the MSE of DecisionTree is', mean_squared_error(test_y, pred_dec_y))
print('the MAE of DecisionTree is', mean_absolute_error(y_test, pred_dec_y))

# (2)KNN回归
knn_regressor = KNeighborsRegressor()
knn_regressor.fit(train_x, train_y)
pred_knn_y = knn_regressor.predict(test_x)

# 使用KNN回归模型自带的评估模块，并输出评估结果
print('the value of default measurement of KNN：',
      dec_regressor.score(X_test, test_y))
print('the value of R-squared of KNN is', r2_score(test_y, pred_knn_y))

# 可以使用标准化器中的inverse_transform函数还原转换前的真实值，这里没有使用标准化后的数据。
# print( 'the MSE of KNN is',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(pred_knn_y)))
print('the MSE of KNN is', mean_squared_error(test_y, pred_knn_y))
print('the MAE of KNN is', mean_absolute_error(y_test, pred_knn_y))
# mse_knn = mean_squared_error(test_y,pred_knn_y)
# print('KNN均方误差=',round(mse_knn,2))
# KNN均方误差= 27.27

# (3) 线性回归模型LinearRegression
# 不要搞混了，这里用的是LinearRegression而不是线性分类的LogisticRegression

lr = LinearRegression()
lr.fit(X_train, y_train.ravel())
lr_y_predict = lr.predict(X_test)

# mse = mean_squared_error(y_test,lr_y_predict)
# print('\nLR均方误差 = ',round(mse,2))   #返回浮点数的四舍五入值

# (4) SGDRegressor

sgdr = SGDRegressor()
sgdr.fit(X_train, y_train.ravel())
sgdr_y_predict = sgdr.predict(X_test)

# mse = mean_squared_error(y_test,sgdr_y_predict)
# print('\nSGDR均方误差 = ',round(mse,2))   #返回浮点数的四舍五入值

# 6、性能测评
# 主要是判断预测值与真实值之间的差距，比较直观的评价指标有
# 平均绝对值误差MAE(mean absolute error)
# 均方误差MSE(mean squared error)
# R-squared评价函数
# 使用LinearRegression模型自带的评估模块，并输出评估结果
print('the value of default measurement of LR：', lr.score(X_test, y_test))
print('the value of R-squared of LR is', r2_score(y_test, lr_y_predict))

# 可以使用标准化器中的inverse_transform函数还原转换前的真实值
print('the MSE of LR is', mean_squared_error(
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))
print('the MAE of LR is', mean_absolute_error(
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))

# 使用SGDRegressor自带的评估模块，并输出评估结果
print('the value of default measurement of SGDR：', sgdr.score(X_test, y_test))

print('the value of R-squared of SGDR is', r2_score(y_test, sgdr_y_predict))
print('the MSE of SGDR is', mean_squared_error(
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict)))
print('the MAE of SGDR is', mean_absolute_error(
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict)))

# 总结：
# 从输出结果来看，AdaBoost、DecisionTree、回归模型自带、SGDR的评估结果与r2_score的值是一样的，推荐使用第一种方式。
# AdaBoost 的MSE性能最好
# SGDRegressor在性能上表现略逊于LinearRegression，前者是随机梯度下降的方式估计参数，后者是精确解析参数
# 在数据量十分庞大（10W+）的时候，推荐使用SGDRegressor
