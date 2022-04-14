import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model


def load_data():
    # 从sklearn中导入数据
    diabetes = datasets.load_diabetes()
    # 只用到数据集中的一种特征
    diabetes_x = diabetes.data[:, np.newaxis, 2]

    # 将数据分成训练和测试集
    X_train = diabetes_x[:-20]  # 训练样本
    X_test = diabetes_x[-20:]  # 测试样本 后20行
    y_train = diabetes.target[:-20]  # 训练标记
    y_test = diabetes.target[-20:]  # 预测对比标记

    # 回归训练及预测
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)

    print('coefficients = %f\n' % float(regr.coef_))
    # 均方误差（残差）
    print('RMSE: %.2f' % np.mean((regr.predict(X_test) - y_test) ** 2))
    # 变异指数,为1时最优
    print('variance score: %.2f' % regr.score(X_test, y_test))

    return X_test, y_test, regr


def graph(X_test, y_test, regr):
    # 输出图
    plt.title(u'LinearRegression Diabetes')  # 标题
    plt.xlabel(u'Attributes')  # X轴坐标
    plt.ylabel(u'Measure of disease')  # y轴坐标
    plt.scatter(diabetes_x_test, y_test, 'r')
    # 预测结果
    plt.plot(X_test, regr.predict(X_test), 'b', linewidth=1)
    plt.show()


if __name__ == '__main__':
    diabetes_x_test, diabetes_y_test, regr = load_data()
    graph(diabetes_x_test, diabetes_y_test, regr)
