import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def load_data(data_path):
    data = pd.read_csv(data_path, sep='\s+')
    data['V37'].fillna(data['V37'].mean().round(3), inplace=True)

    data.dropna(subset=['target'], inplace=True)

    # plt.boxplot(data['V0'])
    # plt.show()
    data_normal = data[(data['V0'] < 10) & (data['V0'] > -2)]

    return data_normal


def training(data):
    # 相关项分析
    d1 = data.corr()['target'].abs().sort_values()
    print(d1)

    y = data['target']
    x = data.drop('target', axis=1)

    pac = PCA(n_components=28)
    x_pca = pac.fit_transform(x)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2)
    std= StandardScaler()
    std.fit(X_train)
    X_train_std = std.transform(X_train,y_train)
    X_test_std = std.transform(X_test,y_test)

    # ========随机森林========
    for i in range(1,10,100):
        rgs_model = RandomForestRegressor()
        rgs_model.fit(X_train_std,y_train)



def predict(data, input):
    pass


if __name__ == '__main__':
    data_path = './data/zhengqi.txt'
    data = load_data(data_path)
    print(data.shape)

    training(data)
