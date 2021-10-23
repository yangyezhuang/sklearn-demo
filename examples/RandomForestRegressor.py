import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# 加载并处理数据
def load_data(data_path):
    data = pd.read_csv(data_path, sep='\s+')
    # print('空值分布：\n', data.isnull().sum())  # 查看空值情况
    data['V37'].fillna(data['V37'].mean().round(3), inplace=True)  # 替换
    data.dropna(subset=['V0', 'target'], inplace=True)
    # 箱线图
    # print(data.describe())
    # plt.boxplot(data['V0'])
    # plt.show()
    data_normal = data[(data['V0'] < 10) & (data['V0'] > -2)]  # 删除异常值所在行
    print(data_normal.shape)

    return data_normal


# 训练模型并保存
def training(data):
    # 相关性分析
    d1 = data.corr()['target'].abs().sort_values()  # abs()取绝对值，
    # print('相关性：\n', d1)

    y = data['target']
    x = data.drop('target', axis=1)

    # pca 降维
    pca = PCA(n_components=28)
    x_pca = pca.fit_transform(x)

    # 划分训练集,测试集
    X_train, X_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2)
    print(X_train.shape)
    std = StandardScaler()
    std.fit(X_train)
    X_train_std = std.transform(X_train)
    X_test_std = std.transform(X_test)
    # ===============随机森林===============
    # 寻找最佳值
    # for i in [1, 10, 100]:
    #     rgs_model = RandomForestRegressor(n_estimators=i)  # 随机森林模型
    #     rgs_model.fit(X_train_std, y_train)
    #     score = rgs_model.score(X_test_std, y_test)
    #     print(score, i)

    # ===============KNN回归=================
    # 寻找最佳值（回归）
    # for i in range(2, 100):
    #     knn_clf = KNeighborsRegressor(n_neighbors=i)
    #     knn_clf.fit(X_train_std, y_train)
    #     score = knn_clf.score(X_test_std, y_test)
    #     print(score, i)

    rgs_model = RandomForestRegressor(n_estimators=100)  # 随机森林模型
    rgs_model.fit(X_train_std, y_train)  # 训练模型
    # pre_y_test = rgs_model.predict(X_test_std)
    # print('pre_y_test:',pre_y_test)
    joblib.dump(rgs_model, './model/randomForest.pkl')  # 保存模型
    score = rgs_model.score(X_test_std, y_test)  # 查看模型准确率
    print('模型准确率：', score)

    return x_pca


# 预测
def predict(data, input):
    std = StandardScaler()
    std.fit(data)
    dd = std.transform(input)  # 标准化
    model = joblib.load('./model/randomForest.pkl')  # 加载模型
    return model.predict(dd)


# 画图
def graph(origin, res):
    plt.rcParams['font.family'] = 'simhei'
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    plt.title('Predict_risk')
    plt.plot(origin, 'r-', label='原始')
    plt.plot(res, 'g-', label='预测')
    plt.ylabel('risk')
    plt.ylim(-2, 2)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data_path = '.\data\zhengqi.txt'
    data = load_data(data_path)

    x_pca = training(data)

    res = predict(x_pca, x_pca[:10])  # 预测数据
    print('predict：', res)

    origin = list(data['target'][:10])  # 原始数据
    graph(origin, res)
