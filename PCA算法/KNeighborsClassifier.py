import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  # 分类器
from sklearn.model_selection import train_test_split


# 加载并处理数据
def load_data(data_path):
    data = pd.read_csv(data_path, header=None)

    # 处理空值
    data[1] = data[1].map(lambda x: np.nan if str(x) == 'None' else x)
    data.dropna(subset=[1], inplace=True)

    # print(data.info())

    # 将字符串(object)转换为数值类型
    data[1] = data[1].astype(np.int64)

    # 箱线图
    # cnt = data.shape[1] - 1
    # for i in range(cnt):
    #     plt.subplot(2, 4, i + 1)
    #     plt.boxplot(data[i])
    # plt.show()
    #
    # 获取正常值
    data = data[(data[3] < 80) & (data[2] > 0) & (data[5] > 0)]

    return data


# 训练模型并保存
def training(data):
    # 查看相关系数
    d1 = data.corr()[8].abs().sort_values()
    # print('相关性：\n', d1)

    y = data[8]
    x = data.drop(8, axis=1)

    # PCA降维
    pca = PCA(n_components=7)
    x_pca = pca.fit_transform(x)
    # print(x)

    # 查看空准确率
    y.value_counts(normalize=True)
    X_train, X_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2)
    # 标准化
    print(X_train.shape)
    std = StandardScaler()
    std_tf = std.fit(X_train)
    X_train_std = std_tf.transform(X_train)
    X_test_std = std_tf.transform(X_test)

    # =============分类器================
    # for i in range(2, 20):
    #     knn_model = KNeighborsClassifier(n_neighbors=i)
    #     knn_model.fit(X_train_std, y_train)
    #     socre = knn_model.score(X_test_std, y_test)
    #     print(socre, i)

    knn_model = KNeighborsClassifier(n_neighbors=13)
    knn_model.fit(X_train_std, y_train)
    pre_y_test = knn_model.predict(X_test_std)

    # 保存模型
    joblib.dump(knn_model, './model/knnClass.pkl')
    socre = knn_model.score(X_test_std, y_test)
    print('模型准确率：', socre)

    print('准确性:', accuracy_score(y_test, pre_y_test))
    print('精度：', precision_score(y_test, pre_y_test, average=None))
    print('召回率：', recall_score(y_test, pre_y_test, average=None))
    print('F分数：', f1_score(y_test, pre_y_test, average=None))

    return x_pca


# 预测
def predict(data, input):
    std = StandardScaler()
    std.fit(data)
    dd = std.transform(input)
    model = joblib.load('./model/knnClass.pkl')
    return model.predict(dd)


if __name__ == '__main__':
    data_path = '../data/pima.data'

    data = load_data(data_path)
    print(data.shape)

    x_pca = training(data)

    res = predict(x_pca, x_pca[:2])
    print('predict：', res)
