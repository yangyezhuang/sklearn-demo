# -*- coding: utf-8 -*-

# 0. 导入相关模块
from sklearn import datasets
# 导入线性回归模型函数
from sklearn.linear_model import LinearRegression

# matplotlib是python专门用于画图的库
import matplotlib.pyplot as plt

# 1. 加载数据集
# 这里将全部数据用于训练，并没有对数据进行划分，上例中将数据划分为训练和测试数据，后面会讲到交叉验证
loaded_data = datasets.load_boston()

# 2. 数据预处理
data_X = loaded_data.data
data_y = loaded_data.target
# data_X是训练数据
# data_y是导入的标签数据

# 3. 创建线性回归模型
linear_model = LinearRegression()

# 4. 模型训练数据，得出参数
linear_model.fit(data_X, data_y)

# 5. 模型预测
# 利用模型，对新数据，进行预测，与原标签进行比较
pre_y = linear_model.predict(data_X[:4, :])

# 6. 结果展示与模型评估
# 6.1结果展示
print(pre_y)
print(data_y[:4])

# 解决matplotlib中的中文显示问题
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# 图形展示，采用点线图。
plt.plot(data_y[:4], label='实际值')
plt.plot(pre_y, label='预测值')

plt.legend()
plt.show()

# 6.2 模型评估
# 使用均方误差对其进行打分，输出精确度，
# 即利用训练好的模型对data_X进行预测，得到预测后，和原本标签进行比较
print("MSE: %.4f" % linear_model.score(data_X, data_y))

# 7. 模型参数
# 输出模型的两个参数，在这里分别指的是，线性回归模型的斜率和截距
print(linear_model.coef_)
print(linear_model.intercept_)

# 取出之前定义的模型的参数
print(linear_model.get_params())
