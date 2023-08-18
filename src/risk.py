import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


class Risk(object):
    """ 商品销售风险预测分析 """

    def load_data(self, data_path):
        """
        加载并处理数据
        :param data_path 源数据路径
        :return: data 处理好的数据
        """
        data = pd.read_csv(data_path)
        data = data[['开始时间', '风险值', '地区名', '销售额']]
        data['开始时间'] = data['开始时间'].map(lambda t: int(t.split('-')[1]))
        data['销售额'] = data['销售额'].astype(np.int32)
        data = pd.get_dummies(data, columns=['地区名'])
        # boxplot
        plt.boxplot(data['销售额'])
        plt.show()
        return data

    def training(self, data):
        """
        训练并保存模型
        :param data: 已经处理过的数据
        :return: 返回特征值
        """
        y = data['风险值']
        x = data.drop('风险值', axis=1)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        std = StandardScaler()
        std.fit(X_train)
        X_train_std = std.transform(X_train)
        X_test_std = std.transform(X_test)

        # ===========KNN==========
        # for i in range(2, 20):
        #     knn_model = KNeighborsRegressor(n_neighbors=i)
        #     knn_model.fit(X_train_std, y_train)
        #     score = knn_model.score(X_test_std, y_test)
        #     print(score, i)

        # =========随机森林=========
        # for i in [1, 10, 100]:
        #     rgs_model = RandomForestRegressor(n_estimators=i)
        #     rgs_model.fit(X_train_std, y_train)
        #     score = rgs_model.score(X_test_std, y_test)
        #     print(score, i)

        knn_model = KNeighborsRegressor(n_neighbors=12)
        knn_model.fit(X_train_std, y_train)
        pre_y_test = knn_model.predict(X_test_std)
        joblib.dump(knn_model, '../model/risk.pkl')
        score = knn_model.score(X_test_std, y_test)

        print('模型准确率：', score)
        print('MSE：', metrics.mean_squared_error(y_test, pre_y_test))  # 均方误差是非负值，模型越好MSE越接近零
        print('RMSE：', np.sqrt(metrics.mean_squared_error(y_test, pre_y_test)))  # 均方根误差，rmse 越小越好
        print('MAE为：', metrics.mean_absolute_error(y_test, pre_y_test))  # 平均绝对误差
        print('决定系数：', metrics.r2_score(y_test, pre_y_test))  # r2的值越接近1，说明回归直线对观测值的拟合程度越好

        return x

    def predict(self, data, input):
        """
        使用model进行预测
        :param data:
        :param input:
        :return:
        """
        std = StandardScaler()
        std.fit(data)
        input_std = std.transform(input)
        model = joblib.load('../model/risk.pkl')

        return model.predict(input_std)

    #
    def save_type(self, df):
        """
        聚类
        :param df:
        :return:
        """
        kmeans = KMeans(3, init='k-means++')
        kmeans.fit(df)
        # print('kmeansPredicter labels:', np.unique(kmeansPredicter.labels_))
        df_res = pd.DataFrame(kmeans.cluster_centers_.flatten())
        sort_res = df_res.sort_values(by=0)
        sort_res.T.to_csv('../type/risk_type.csv', header=None, index=None, mode='w')


    def get_map(self, num):
        """
        映射标签
        :param num:
        :return:
        """
        arr = pd.read_csv('../type/risk_type.csv', nrows=1, header=None)
        arr = arr.values[0]
        if num <= arr[0]:
            return {'risk': num, 'label': 'low'}
        elif arr[0] < num <= arr[1]:
            return {'risk': num, 'label': 'normal'}
        elif num > arr[1]:
            return {'risk': num, 'label': 'high'}


    def graph(self, origin, predict):
        """
        生成预测图
        :param origin: 原始数据
        :param predict: 预测结果
        :return:
        """
        plt.rcParams['font.family'] = 'simhei'
        plt.title("销售风险预测")
        plt.plot(origin, label='原始')
        plt.plot(predict, label='预测')
        plt.ylabel('风险值')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    r = Risk()
    data_path = '../data/per_month_sale_and_risk.csv'
    data = r.load_data(data_path)
    # 训练模型
    x = r.training(data)
    # 调用模型进行预测
    res = r.predict(x, x[:10])
    print('predict：', res)
    # 聚类
    r.save_type(pd.DataFrame(data['风险值']))
    for i in res:
        print(r.get_map(i))  # 映射

    origin = list(data['风险值'])[:10]
    # 调用绘图函数
    r.graph(origin, res)
