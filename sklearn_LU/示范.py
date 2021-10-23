# -*- coding: utf-8 -*-

import numpy as np
import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC  # 支持向量机
from sklearn.naive_bayes import MultinomialNB  # 朴素也贝斯
from sklearn.tree import DecisionTreeClassifier  # 决策树
from sklearn.ensemble import RandomForestClassifier  # 随机森铃
from sklearn.ensemble import GradientBoostingClassifier  # GBDT
from xgboost import XGBClassifier  # xgboost


def modelReturn(model, name):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    predict = model.predict(x_test)
    trueNum = 0
    for i in range(len(y_test)):
        if (y_test[i] == predict[i]):
            trueNum += 1
    print(name, ":", trueNum / len(y_test))


dataframe = pandas.read_csv("export.csv")
# 获取 CVS中的值
dataset = dataframe.values
# 本身数据有53列  下标0开始 取52列  53列是标签
# X = dataset[:,0:53].astype(np.float);

# Y =dataset[:,53];
X = dataset[:, 3:6].astype(np.float)

Y = dataset[:, 53]
x_train, x_test, y_train, y_test = train_test_split(X, Y)

# xgboost 46 50
model = XGBClassifier()
modelReturn(model, "xgboost")
# GBDT 40 -48
model = GradientBoostingClassifier()
modelReturn(model, "GBDT")

# 随机森林  44-46
model = RandomForestClassifier()
modelReturn(model, "随机森林")

# 决策树  36-39
model = DecisionTreeClassifier()
modelReturn(model, "决策树")

# 朴素也贝斯 44-51
model = MultinomialNB()
modelReturn(model, "朴素也贝斯")

# 支持向量机  45-48
model = LinearSVC()
modelReturn(model, "支持向量机")

# SVM  48-52
model = SVC()
modelReturn(model, "SVM")

# laoss   68-73%
model = Lasso(alpha=0.005)  # 调节aplha 可以实现对拟合的程度
modelReturn(model, "laoss")

"""
model.fit(x_train,y_train);

predict =model.predict(x_test);

trueNum =0;

print(predict)

for i  in range(len(y_test)):
    if ((abs(y_test[i])-abs(predict[i])< 0.5)):
        trueNum += 1;
        
        
print(trueNum/len(y_test));
"""
"""
pca = PCA(n_components=27);
xTrainPca = pca.fit_transform(x_train);
xTestPca = pca.fit_transform(x_test);


log =LogisticRegression();
log.fit(xTrainPca,y_train);

print("准确率:",log.score(xTestPca,y_test));
"""

"""
#降到10个维度
pca = PCA(n_components=50);

xTrainPca = pca.fit_transform(x_train);
xTestPca = pca.fit_transform(x_test);

knn = KNeighborsClassifier(n_neighbors=11);
knn.fit(xTrainPca,y_train);

print(knn.score(xTestPca,y_test))
"""
