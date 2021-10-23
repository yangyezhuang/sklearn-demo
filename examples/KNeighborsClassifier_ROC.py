import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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

    # =============KNeighborsClassifier分类器================
    # for i in range(2, 20):
    #     knn_model = KNeighborsClassifier(n_neighbors=i)
    #     knn_model.fit(X_train_std, y_train)
    #     socre = knn_model.score(X_test_std, y_test)
    #     print(socre, i)
    #
    # # 分类模型评估
    # print('SCORE：', knn_model.score(X_test_std, y_test))
    # print('准确率：',accuracy_score(y_test, pre_y_test))
    # print('精确率：', precision_score(y_test, pre_y_test))
    # print('召回率：', recall_score(y_test, pre_y_test))
    # print('F1值：', f1_score(y_test, pre_y_test))
    # print('Cohen’s Kappa：', cohen_kappa_score(y_test, pre_y_test))
    #
    # # 分类模型评价报告
    # from sklearn.metrics import classification_report
    # print('预测数据的分类报告：', '\n',classification_report(y_test, pre_y_test))
    #
    # # 保存模型
    # joblib.dump(knn_model, 'models/knnClassifier.pkl')

    # =============SVM分类器================
    from sklearn.svm import SVC
    # for i in [0.01,0.1,1, 10]:
    #     for k in ['linear', 'poly', 'rbf']:
    #         svc_model = SVC(C=i, kernel=k)
    #         svc_model.fit(X_train_std, y_train)
    #         score = svc_model.score(X_test_std, y_test)
    #         print(score, i, k)
    svc_model = SVC(C=1, kernel='linear')
    svc_model.fit(X_train_std, y_train)
    pre_y_test = svc_model.predict(X_test_std)

    # 分类模型评估
    print('SCORE：', svc_model.score(X_test_std, y_test))
    print('准确率：', metrics.accuracy_score(y_test, pre_y_test))
    print('精确率：', metrics.precision_score(y_test, pre_y_test))
    print('召回率：', metrics.recall_score(y_test, pre_y_test))
    print('F1值：', metrics.f1_score(y_test, pre_y_test))
    print('Cohen_Kappa：',metrics.cohen_kappa_score(y_test, pre_y_test))  # 一个介于(-1, 1)之间的数. score>0.8意味着好的分类；0或更低意味着不好
    print('ROC值：', metrics.roc_auc_score(y_test, pre_y_test))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pre_y_test)
    # print(fpr,type(fpr))
    # print(tpr, type(tpr))
    print('auc:', metrics.auc(fpr, tpr))  # auc(fpr, tpr)的值 就是 roc_auc_score(y_test, pre_y_test)
    # 画ROC曲线图
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr,
             lw=lw, label='ROC curve (area = %0.2f)' % metrics.roc_auc_score(y_test, pre_y_test))
    plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    # 分类模型评价报告
    from sklearn.metrics import classification_report
    print('预测数据的分类报告：', '\n', classification_report(y_test, pre_y_test))

    # 保存模型
    joblib.dump(svc_model, 'models/svcClassifier.pkl')

    return x_pca


# 预测
def predict(data, input):
    std = StandardScaler()
    std.fit(data)
    dd = std.transform(input)
    # model = joblib.load('models/knnClassifier.pkl')
    model = joblib.load('models/svcClassifier.pkl')
    return model.predict(dd)


# 画图
def graph(origin, res):
    plt.rcParams['font.family'] = 'simhei'
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    plt.plot(origin, 'r-', label='原始')
    plt.plot(res, 'g-', label='预测')
    plt.ylim(-2, 2)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data_path = 'data\pima.data'

    data = load_data(data_path)
    print(data.shape)
    x_pca = training(data)
    res = predict(x_pca, x_pca[:10])
    print('predict：', res)
    # # 画双折线图
    # # print(list(data[8])[:2])
    # origin = list(data[8])[:10]  # 原始数据
    # graph(origin, res)
