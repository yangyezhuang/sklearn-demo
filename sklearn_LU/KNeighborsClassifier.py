# -*- coding: utf-8 -*-

# 0. 导入相关模块
from sklearn.model_selection import train_test_split
from sklearn import datasets
# 导入K近邻分类器函数
from sklearn.neighbors import KNeighborsClassifier

# matplotlib是python专门用于画图的库
import matplotlib.pyplot as plt

# 1. 加载数据
iris = datasets.load_iris()

# 2. 数据预处理
# 导入数据和标签
iris_X = iris.data
iris_y = iris.target
# 划分为训练集和测试集数据
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)
# print(y_train)

# 3. 构建模型
# 创建KNN分类器
knn_model = KNeighborsClassifier()

# 4. 模型训练
knn_model.fit(X_train, y_train)

# 5. 使用训练好的KNN进行数据预测
pre_y = knn_model.predict(X_test)

# 6. 结果展示与模型评估
# 6.1结果展示
print(pre_y)
print(y_test)

# 解决matplotlib中的中文显示问题
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# 图形展示，采用点线图。
plt.plot(y_test, label='实际值')
plt.plot(pre_y, label='预测值')

plt.legend()
plt.show()

# 6.2 模型评估
# 使用均方误差对其进行打分，输出精确度，
# 即利用训练好的模型对X_test进行预测，得到预测后，和原本标签进行比较
print("MSE: %.4f" % knn_model.score(X_test, y_test))
# R2：决定系数
from sklearn.metrics import r2_score

r2_score(y_test, pre_y)  # KNN Regression

# 7. 模型参数
# 取出之前定义的模型的参数
print(knn_model.get_params())
