import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split



class Steam(object):
    def load_data(self, data_path):
        """
        加载并处理数据
        :param data_path: 数据源路径
        :return: 处理好的数据
        """
        data = pd.read_csv(data_path, sep='\s+')
        data['V37'].fillna(data['V37'].mean().round(3), inplace=True)
        data.dropna(subset=['target'], inplace=True)
        # 箱线图
        # plt.boxplot(data['V0'])
        # plt.show()
        data = data[(data['V0'] < 10) & (data['V0'] > -2)]

        return data

    def training(self, data):
        """
        训练模型并保存
        :param data: 处理好的数据
        :return: x_pca
        """

        # 查看相关性
        corr = data.corr()['target'].abs().sort_values()
        # print(corr)

        y = data['target']
        x = data.drop('target', axis=1)

        # PCA降维
        pca = PCA(n_components=28)
        x_pca = pca.fit_transform(x)

        X_train, X_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2)
        std = StandardScaler()
        std.fit(X_train)
        X_train_std = std.transform(X_train)
        X_test_std = std.transform(X_test)

        # =============随机森林===========
        # for i in [1, 10, 100]:
        #     rgs_model = RandomForestRegressor(n_estimators=i)
        #     rgs_model.fit(X_train_std, y_train)
        #     score = rgs_model.score(X_test_std, y_test)
        #     print(score, i)

        # ============KNN===============
        # for i in range(2, 20):
        #     knn_model = KNeighborsRegressor(n_neighbors=i)
        #     knn_model.fit(X_train_std, y_train)
        #     score = knn_model.score(X_test_std, y_test)
        #     print(score, i)


        rgs_model = RandomForestRegressor(n_estimators=100)
        rgs_model.fit(X_train_std, y_train)
        joblib.dump(rgs_model, '../model/steam.pkl')
        pre_y_test = rgs_model.predict(X_test_std)
        score = rgs_model.score(X_test_std, y_test)

        print('模型准确率：', score)
        print('MSE：', metrics.mean_squared_error(y_test, pre_y_test))  # 均方误差是非负值，越接近零
        print('RMSE：', np.sqrt(metrics.mean_squared_error(y_test, pre_y_test)))  # 均方根误差，越小越好
        print('MAE：', metrics.mean_absolute_error(y_test, pre_y_test))  # 平均绝对误差
        print('R2决定系数：', metrics.r2_score(y_test, pre_y_test))  # r2的值越接近1越好

        return x_pca

    def predict(self, data, input):
        """
        加载 model进行预测
        :param data: x_pca
        :param input: x_pca[:10]
        :return: 预测结果
        """
        std = StandardScaler()
        std.fit(data)
        input_std = std.transform(input)
        model = joblib.load('../model/steam.pkl')
        return model.predict(input_std)

    def graph(self, origin, predict):
        """
        绘图
        :param origin: 原始数据
        :param predict: 预测结果
        :return:
        """
        plt.rcParams['font.family'] = 'simhei'
        plt.rcParams['axes.unicode_minus'] = False
        plt.plot(origin, label='原始')
        plt.plot(predict, label='预测')
        plt.ylabel('风险值')
        plt.ylim(-2, 2)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    s = Steam()
    data_path = '../data/zhengqi.txt'
    data = s.load_data(data_path)
    # print(data.isnull().sum())

    x_pca = s.training(data)
    # print(x_pca)

    res = s.predict(x_pca, x_pca[:10])
    print('预测结果：', res)

    origin = list(data['target'])[:10]
    s.graph(origin, res)
