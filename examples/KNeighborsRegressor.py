import joblib  # 加载和保存模型
import numpy as np
import pandas as pd
from sklearn.svm import SVC  # 支持向量机
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  # 聚类
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor  # 随机森林
from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn import metrics


# 加载并处理数据
def load_data(data_path):
    origin_data = pd.read_csv(data_path)
    data = origin_data[['开始时间', '风险值', '地区名', '销售额']]
    data['销售额'] = data['销售额'].astype(np.int32)
    data_del = pd.get_dummies(data, columns=['地区名'])
    data_del['month'] = data_del['开始时间'].map(lambda t: int(t.split('-')[1]))
    data_del = data_del.drop('开始时间', axis=1)

    # 箱线图，观察异常值
    # plt.boxplot(data_del['销售额'])
    # plt.show()
    # plt.boxplot(data_del['风险值'])
    # plt.show()

    return data_del


# 训练模型并保存
def training(data):
    # 导入已经处理好的数据
    y = data['风险值']  # y：对应的风险值
    x = data.drop('风险值', axis=1)  # x：特征值
    # 把数据切割为训练集,测试集
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # 因为销售额的数量级和其他特征不一样，要对数据进行标准化
    std = StandardScaler()
    std.fit(X_train)
    X_train_std = std.transform(X_train)  # 标准化
    X_test_std = std.transform(X_test)

    # ===============随机森林===============
    # 寻找最优参数
    # for i in [1, 10, 100]:
    #     rgs_model = RandomForestRegressor(n_estimators=i)  # 随机森林模型
    #     rgs_model.fit(X_train_std, y_train)
    #     score = rgs_model.score(X_test_std, y_test)
    #     print(score, i)

    # ===============KNN回归=================
    # for i in range(2, 20):
    #     knn_clf = KNeighborsRegressor(n_neighbors=i)
    #     knn_clf.fit(X_train_std, y_train)
    #     score = knn_clf.score(X_test_std, y_test)
    #     print(score, i)

    # =============SVC(支持向量机)===========
    # for i in [1, 10]:
    #     for k in ['linear', 'poly', 'rbf']:
    #         svc_model = SVC(C=i, kernel=k)
    #         svc_model.fit(X_train, y_train)
    #         score = svc_model.score(X_test, y_test)
    #         print(score, i, k)

    knn_clf = KNeighborsRegressor(n_neighbors=12)
    knn_clf.fit(X_train_std, y_train)  # 拟合模型
    joblib.dump(knn_clf, "model/risk01.pkl")  # 保存模型
    # 用测试集测试模型的准确率
    score = knn_clf.score(X_test_std, y_test)
    print('模型准确率：', score)

    pre_y_test = knn_clf.predict(X_test_std)
    # 回归模型评估  MSE均方误差，RMSE回归分析模型中最常用的评估方法
    print('SCORE为：', knn_clf.score(X_test_std, y_test))
    print('MSE为：', metrics.mean_squared_error(y_test, pre_y_test))  # 均方误差是非负值，模型越好MSE越接近零
    print('MSE为(直接计算)：', np.mean((y_test - pre_y_test) ** 2))
    print('RMSE为：', np.sqrt(metrics.mean_squared_error(y_test, pre_y_test)))  # 均方根误差，rmse 越小越好
    print('MAE为：', metrics.mean_absolute_error(y_test, pre_y_test))  # 平均绝对误差

    return x


# 使用model进行预测
def predict(data, input):
    std = StandardScaler()
    std.fit(data)
    dd = std.transform(input)
    model = joblib.load("model/risk01.pkl")  # 加载模型
    # 预测值
    return model.predict(dd)


# 聚类函数，并实现排序、保存
def save_type(df, count=3, mode='w'):
    # 创建KMeans类对象
    kmeans = KMeans(count, 'k-means++')
    # 调用fit方法
    kmeans.fit(df)
    df_res = pd.DataFrame(kmeans.cluster_centers_.flatten())
    sort_res = df_res.sort_values(by=0)
    sort_res.T.to_csv('type.csv', header=None, index=None, mode=mode)


# 映射风险值
def get_risk_label(num):
    arr = pd.read_csv('type.csv', nrows=1, header=None)
    arr = arr.values[0]
    if num <= arr[0]:
        return {"risk": num, "label": "low risk"}
    elif num > arr[0] and num <= arr[1]:
        return {"risk": num, "label": "normal risk"}
    elif num > arr[1]:
        return {"risk": num, "label": "high risk"}


# 对结果绘图
def graph(origin, predict):
    plt.plot(predict, 'r-', label='predict')
    plt.plot(origin, 'b-', label='projects')
    plt.ylabel('risk')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data_path = '.\data\per_month_sale_and_risk.csv'
    data = load_data(data_path)

    # 训练模型,返回特征值
    x = training(data)

    # 预测
    res = predict(x, x[:10])
    print('预测结果：\n', res)
    # 聚类,映射
    save_type(data.iloc[:, 0:1])
    for i in res:
        print(get_risk_label(i))

    origin = list(data['风险值'])[:10]
    graph(origin, res)
