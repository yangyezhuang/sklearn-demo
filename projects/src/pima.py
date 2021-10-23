import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score


class Pima(object):
    def load_data(self, data_path):
        data = pd.read_csv(data_path, header=None)
        data[1] = data[1].map(lambda x: np.nan if str(x) == 'None' else x)
        data.dropna(subset=[1], inplace=True)

        data[1] = data[1].astype(np.int64)
        data = data[(data[3] < 80) & (data[2] > 0) & (data[5] > 0)]

        return data

    def training(self, data):
        y = data[8]
        x = data.drop(8, axis=1)

        # 查看相关性
        d1 = data.corr()[8].abs().sort_values()
        # print(d1)
        # PCA降维
        pca = PCA(n_components=7)
        x_pca = pca.fit_transform(x)

        X_train, X_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2)
        std = StandardScaler()
        std.fit(X_train)
        X_train_std = std.transform(X_train)
        X_test_std = std.transform(X_test)

        # ===========KNN 分类器===========
        # for i in range(2, 20):
        #     knn_model = KNeighborsClassifier(n_neighbors=i)
        #     knn_model.fit(X_train_std, y_train)
        #     score = knn_model.score(X_test_std, y_test)
        #     print(score, i)

        # ==========SVC 分类器==========
        # for i in [0.01, 0.1, 1, 10]:
        #     for k in ['linear', 'rbf', 'poly']:
        #         svc_model = SVC(C=i, kernel=k)
        #         svc_model.fit(X_train_std, y_train)
        #         score = svc_model.score(X_test_std, y_test)
        #         print(score, i)

        # ============交叉验证=================
        # cv_scores = []  # 用来放每个模型的结果值
        # for i in range(2, 20):
        #     knn = KNeighborsClassifier(n_neighbors=i)  # knn模型，这里一个超参数可以做预测，当多个超参数时需要使用另一种方法GridSearchCV
        #     scores = cross_val_score(knn, X_train_std, y_train, cv=10,
        #                              scoring='accuracy')  # cv：选择每次测试折数  accuracy：评价指标是准确度,可以省略使用默认值，具体使用参考下面。
        #     cv_scores.append(scores.mean())
        # plt.plot(range(2, 20), cv_scores)
        # plt.xlabel('K')
        # plt.ylabel('Precision')  # 通过图像选择最好的参数
        # plt.show()

        svc_model = SVC(C=1, kernel='linear')
        svc_model.fit(X_train_std, y_train)
        joblib.dump(svc_model, '../model/pima.pkl')
        score = svc_model.score(X_test_std, y_test)
        pre_y_test = svc_model.predict(X_test_std)

        print('模型准确率：', score)
        print('精确率：', metrics.precision_score(y_test, pre_y_test))
        print('召回率：', metrics.recall_score(y_test, pre_y_test))
        print('F1值：', metrics.f1_score(y_test, pre_y_test))
        print('ROC值：', metrics.roc_auc_score(y_test, pre_y_test))
        print('cohen_kappa:', metrics.cohen_kappa_score(y_test, pre_y_test))
        print('预测数据的分类报告\n', metrics.classification_report(y_test, pre_y_test))

        fpr, tpr, thresholds = metrics.roc_curve(y_test, pre_y_test)
        # print('auc:', auc(fpr, tpr))  # auc(fpr, tpr)的值 就是 roc_auc_score(y_test, pre_y_test)
        # 画ROC曲线图
        plt.figure()
        plt.title('Receiver operating characteristic')
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % metrics.roc_auc_score(y_test, pre_y_test))
        plt.plot([0, 1], [0, 1], '--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()

        return x_pca

    def predict(self, data, input):
        std = StandardScaler()
        std.fit(data)
        input_std = std.transform(input)
        model = joblib.load('../model/pima.pkl')
        return model.predict(input_std)

    def graph(self, origin, predict):
        plt.rcParams['font.family'] = 'simhei'
        plt.rcParams['axes.unicode_minus'] = False
        plt.plot(origin, 'o-', label='原始')
        plt.plot(predict, label='预测')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    data_path = '../data/pima.data'
    p = Pima()
    data = p.load_data(data_path)
    # print(data)

    x_pca = p.training(data)

    res = p.predict(x_pca, x_pca[:10])
    print('预测结果：', res)

    origin = list(data[8])[:10]
    p.graph(origin, res)
