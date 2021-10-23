import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def load_data(data_path):
    origin_data = pd.read_csv(data_path)
    data = origin_data[['开始时间', '风险值', '地区名', '销售额']]
    data['销售额'] = data['销售额'].astype(np.int32)
    data_del = pd.get_dummies(data, columns=['地区名'])
    data_del['month'] = data_del['开始时间'].map(lambda x: int(x.split('-')[1]))
    data_del = data_del.drop(['开始时间'], axis=1)
    # # 画箱线图
    # plt.boxplot(data_del['风险值'])
    # plt.show()
    # plt.boxplot(data_del['销售额'])
    # plt.show()
    return data_del


def training(data):
    # 划分数据集
    y = data['风险值']
    x = data.drop('风险值', axis=1)
    # 划分训练集，测试集
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # 标准化
    std = StandardScaler()
    std.fit(X_train)
    X_train_std = std.transform(X_train)  # 标准化
    X_test_std = std.transform(X_test)

    # 寻找最佳值
    # for i in range(2, 20):
    #     knn_clf = KNeighborsRegressor(n_neighbors=i)  # 回归
    #     knn_clf.fit(X_train_std, y_train)
    #     score = knn_clf.score(X_test_std, y_test)
    #     print(score, i)

    # 得出最佳值
    knn_clf = KNeighborsRegressor(n_neighbors=12)
    knn_clf.fit(X_train_std, y_train)  # 拟合模型
    joblib.dump(knn_clf, './models/risk02.pkl')  # 保存模型
    score = knn_clf.score(X_test_std, y_test)  # 用测试集测试模型准确率
    print('模型准确率：', score)


def predict(data,input):
    # 根据特征值预测风险值
    x = data.drop('风险值', axis=1)  # 获取特征值
    # 进行标准化
    std = StandardScaler()
    std.fit(x)
    dd = std.transform(input)  #
    model = joblib.load('./models/risk02.pkl')

    return model.predict(dd)


def save_type(df, count=3, mode='w'):
    kmeans = KMeans(count, 'k-means++')
    kmeans.fit(df)
    df_res = pd.DataFrame(kmeans.cluster_centers_.flatten())
    sort_res = df_res.sort_values(by=0)
    sort_res.T.to_csv('type1.csv', header=None, index=None, mode=mode)


def get_risk_label(num):
    arr = pd.read_csv('type.csv', nrows=1, header=None)
    arr = arr.values[0]
    if num <= arr[0]:
        return {"risk": num, "label": "low risk"}
    elif num > arr[0] and num <= arr[1]:
        return {"risk": num, "label": "normal risk"}
    elif num > arr[1]:
        return {"risk": num, "label": "high risk"}


if __name__ == '__main__':
    data_path = 'E:\Code\PycharmProjects\scikit-learn\DM\DM-2\data\per_month_sale_and_risk.csv'
    data = load_data(data_path)
    print(data)

    training(data)


    # 预测
    y = data['风险值']
    x = data.drop('风险值', axis=1)
    res = predict(data, x[:10])
    print('预测结果：\n', y[:10].values)
    # 保存风险聚类
    save_type(data.iloc[:, 0:1])
    for i in res:
        print(get_risk_label(i))
    # 对结果绘图
    plt.plot(res, 'r-', label='predict')
    plt.plot(y[:10].values, 'b-', label='test')
    plt.legend()
    plt.show()
