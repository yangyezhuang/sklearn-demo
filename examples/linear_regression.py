import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# 从sklearn中导入数据
diabetes = datasets.load_diabetes()

# 只用到数据集中的一种特征
diabetes_x = diabetes.data[:, np.newaxis, 2]

# 将数据分成训练和测试集
diabetes_x_train = diabetes_x[:-20]  # 训练样本
diabetes_x_test = diabetes_x[-20:]  # 测试样本 后20行

# 将标签分为测试和训练集合
diabetes_y_train = diabetes.target[:-20]  # 训练标记
diabetes_y_test = diabetes.target[-20:]  # 预测对比标记

# 回归训练及预测
regr = linear_model.LinearRegression()
regr.fit(diabetes_x_train, diabetes_y_train)
# coefficients
print('coefficients = %f\n' % float(regr.coef_))

# 均方误差（残差）
print('Residual sum of squares: %.2f' %
      np.mean((regr.predict(diabetes_x_test) - diabetes_y_test) ** 2))

# 变异指数,为1时最优
print('variance score: %.2f' % regr.score(diabetes_x_test, diabetes_y_test))

# 输出图
plt.title(u'LinearRegression Diabetes')  # 标题
plt.xlabel(u'Attributes')  # X轴坐标
plt.ylabel(u'Measure of disease')  # y轴坐标

# 点的准确位置
plt.scatter(diabetes_x_test, diabetes_y_test, color='red')

# 预测结果
plt.plot(diabetes_x_test, regr.predict(
    diabetes_x_test), color='blue', linewidth=3)

plt.show()
