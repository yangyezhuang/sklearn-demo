# Import Library
from sklearn import svm

"""
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object
model = svm.SVC()
# there is various option associated with it, this is simple for classification. You can refer link, for mo# re detail.
# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)

#Predict Output
predicted= model.predict(x_test)


sklearn svm基本使用
SVM基本使用　　
　　SVM在解决分类问题具有良好的效果，出名的软件包有libsvm(支持多种核函数),liblinear。此外python机器学习库scikit-learn也有svm相关算法，sklearn.svm.SVC和

sklearn.svm.LinearSVC 分别由libsvm和liblinear发展而来。

　　推荐使用SVM的步骤为：

将原始数据转化为SVM算法软件或包所能识别的数据格式；
将数据标准化；(防止样本中不同特征数值大小相差较大影响分类器性能)
不知使用什么核函数，考虑使用RBF；
利用交叉验证网格搜索寻找最优参数(C, γ)；（交叉验证防止过拟合，网格搜索在指定范围内寻找最优参数）
使用最优参数来训练模型；
测试。
下面利用scikit-learn说明上述步骤：
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from pandas.plotting import scatter_matrix

import pandas as pd


def load_data(filename):
    '''
    假设这是鸢尾花数据,csv数据格式为：
    0,5.1,3.5,1.4,0.2
    0,5.5,3.6,1.3,0.5
    1,2.5,3.4,1.0,0.5
    1,2.8,3.2,1.1,0.2
    每一行数据第一个数字(0,1...)是标签,也即数据的类别。
    '''
    # data = np.genfromtxt(filename, delimiter=',')
    dataSet = pd.read_csv(filename)
    print(dataSet)
    print(dataSet.groupby("class").size())

    array = dataSet.values
    x = array[:, 0:4]  # 数据特征
    y = array[:, 4]  # 标签

    # x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=7)
    # print(y)
    # y = data[:, 0].astype(int)  # 标签
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)  # 标准化
    # 将数据划分为训练集和测试集，test_size=.3表示30%的测试集
    x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=.3)
    return x_train, x_test, y_train, y_test


def KNN(x_train, x_test, y_train, y_test):
    # 在Python中实现并不困难，以KNN为例，仅需4行代码即可完成训练模型和评估
    model = KNeighborsClassifier()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print('KNN精度为: %s.' % accuracy_score(y_test, predictions))

    predic = model.predict([[5.9, 3.0, 5.1, 1.8]])
    print("结果：", predic[0])
    print(predic)


'''
model = RandomForestClassifier(n_estimators=5)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print(accuracy_score(y_test, predictions)

model = LogisticRegression()
'''


def svm_c(x_train, x_test, y_train, y_test):
    # rbf核函数，设置数据权重
    svc = SVC(kernel='rbf', class_weight='balanced', )
    c_range = np.logspace(-5, 15, 11, base=2)
    gamma_range = np.logspace(-9, 3, 13, base=2)
    # 网格搜索交叉验证的参数范围，cv=3,3折交叉
    param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
    grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
    # 训练模型
    clf = grid.fit(x_train, y_train)
    # 计算测试集精度
    score = grid.score(x_test, y_test)
    print('SVM精度为: %s.' % score)


if __name__ == '__main__':
    # svm_c(*load_data('./data/iris.csv'))

    KNN(*load_data('./data/iris.csv'))
