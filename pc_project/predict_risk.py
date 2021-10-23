import pandas as pd
import numpy as np
import joblib  # 加载和保存模型
from utils import load_data, get_risk_label
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

path = '../data/per_month_sale_and_risk.csv'


def train_risk_model():
    # 导入所有待训练的数据，数据已经经过数值化处理
    data = load_data(path)
    # x 里面是特征
    # y 里面是对应的风险值
    y = data['风险值']
    x = data.drop('风险值', axis=1)
    # 把数据切割为训练集 X_train ，y_train；测试集 X_test y_test
    # 训练集为了训练模型
    # 测试集为了测试模型的准确率
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # 因为销售额的数量级和其他特征不一样
    # 要对数据进行标准化
    standarScaler = StandardScaler()
    standarScaler.fit(X_train)
    X_train_std = standarScaler.transform(X_train)
    X_test_std = standarScaler.transform(X_test)
    knn_clf = KNeighborsRegressor(n_neighbors=12)
    # 拟合模型
    knn_clf.fit(X_train_std, y_train)
    # 保存模型 到 models/risk.pkl
    joblib.dump(knn_clf, "./models/risk.pkl")
    # 用测试集测试模型的准确率
    score = knn_clf.score(X_test_std, y_test)
    print(score)

    # 手工输入10条特征，预测标签（也就是风险）
    # print(knn_clf.predict(X_test_std[:10]))
    # print(y_test[:10])


#     加载已经保存的模型，预测风险
def predict_risk(input):
    """
    :param input: 输入待预测的特征
    :return: 返回风险值
    """
    data = load_data(path)
    x = data.drop('风险值', axis=1)
    standarScaler = StandardScaler()
    standarScaler.fit(x)
    dd = standarScaler.transform(input)

    # 加载模型
    model = joblib.load("./models/risk.pkl")
    # 预测值
    return model.predict(dd)


if __name__ == '__main__':
    # 训练模型，并保存到./models/risk.pkl
    # train_risk_model()

    # 预测
    data = load_data(path)
    y = data['风险值']
    x = data.drop('风险值', axis=1)
    res = predict_risk(x[:10])
    print(res)
    print(y[:10].values)
