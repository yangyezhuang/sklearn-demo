from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(X)

# Decision Tree Classifieras for estimator
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0)

# cross_validate
# X, y = wine.data, wine.target
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X, y, cv=5)

print("******cross_validate******")
print(scores)  # cv = number of splited data
print(scores.mean())

# pre_y= clf.predict(X)

print(clf)

# cross_validate
from sklearn.model_selection import cross_validate

scoring = ['precision_macro', 'recall_macro']
scores = cross_validate(clf, X, y, scoring=scoring, cv=5)
print(scores)

# ******************回归度量*********************
# TL; DR：在大多数情况下，我们使用R2或RMSE。

# Data Preparation
from sklearn.datasets import load_boston

boston = load_boston()
X, y = boston.data, boston.target
# Train data and Test data Splitting
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 机器学习模型评估指标示例
# 模型1：线性回归
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X_train, y_train)
y_pred1 = reg.predict(X_test)

# 模型2：决策树回归

from sklearn.tree import DecisionTreeRegressor

reg2 = DecisionTreeRegressor(max_depth=3)
reg2.fit(X_train, y_train)

y_pred2 = reg2.predict(X_test)

# R2：决定系数
from sklearn.metrics import r2_score

print(r2_score(y_test, y_pred1))  # Linear Regression
print(r2_score(y_test, y_pred2))  # Decision Tree Regressor

# MSE：均方误差
from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_test, y_pred1))
print(mean_squared_error(y_test, y_pred2))

# RMSE：均方根误差
import numpy as np

print(np.sqrt(mean_squared_error(y_test, y_pred1)))
print(np.sqrt(mean_squared_error(y_test, y_pred2)))

# MAE：平均绝对误差
from sklearn.metrics import mean_absolute_error

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print(mean_absolute_error(y_test, y_pred))

print("   ****************分类指标***********************")

# 一对一分类：例如付费用户或免费
# One vs. Rest分类：例如高级会员或付费或免费
# 使用Iris数据集作为多类分类问题。

# Data Preparation
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
# Train data and Test data Splitting
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 模型1：SVM

from sklearn.svm import SVC

clf1 = SVC(kernel='linear', C=0.01)
clf1.fit(X_train, y_train)
y_pred1 = clf1.predict(X_test)

# 模型2：朴素贝叶斯
from sklearn.naive_bayes import GaussianNB

clf2 = GaussianNB()
clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)

# 现在我们准备评估我们的两个模型并选择一个！
# 1.准确性：

from sklearn.metrics import accuracy_score

print('++++++++++++++++\n',y_test)
print(y_pred1)
print(accuracy_score(y_test, y_pred1))
print(accuracy_score(y_test, y_pred2))

# 2.精度：

from sklearn.metrics import precision_score

print(precision_score(y_test, y_pred1, average=None))
print(precision_score(y_test, y_pred2, average=None))

# 3.召回或灵敏度：

from sklearn.metrics import recall_score

print(recall_score(y_test, y_pred2, average=None))

# 4. F分数：

from sklearn.metrics import f1_score

print(f1_score(y_test, y_pred2, average=None))

# 5.混淆矩阵

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred2))

# 6. ROC

# 如果你不使用OneVsRest Classifier，它不起作用......

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

clf = OneVsRestClassifier(LinearSVC(random_state=0))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 通过ROC Curve进行检查。

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=2)
print("********ROC Curve********")
print(fpr, tpr, thresholds)

# 7. AUC：曲线下面积

from sklearn.metrics import auc

print(auc(fpr, tpr))

# 8.多类对数损失
# 这是一个概率。并且需要使用OneVsRestClassifier。

# clf = OneVsRestClassifier(LinearSVC(random_state=0))
# from sklearn.metrics import log_loss
# y_pred = clf.decision_function(X_test) # not .predict()
# # y_pred = clf.predict_proba(X_test)
# print(log_loss(y_test, y_pred))
