from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_data():
    data = pd.read_csv('../data/per_month_sale_and_risk.csv')
    data = data[['开始时间', '风险值', '地区名', '销售额']]
    data['销售额'] = data['销售额'].astype(int)
    # 把地区名转化为数值
    data_del = pd.get_dummies(data, columns=['地区名'])
    # 新建月份列
    data_del['month'] = data_del['开始时间'].map(lambda t: int(t.split('-')[1]))
    data_del = data_del.drop('开始时间', axis=1)
    return data_del


# 聚类函数，并实现排序、保存
def save_type(df, mode='w'):
    # 创建KMeans类对象
    # kmeans = KMeans(3, 'k-means++')
    # 调用fit方法
    # kmeans.fit(df)
    # result = kmeans.predict(df)
    # print(np.unique(result))
    # y_=kmeans.fit_predict(df)
    # print(df[y_==0]['风险值'])
    # print(df[y_ == 1]['风险值'])
    # print(df[result == 2]['风险值'])
    # re1=kmeans.predict([[0.75]])
    # print(re1,df[result == re1[0]]['风险值'])
    #
    scale = MinMaxScaler().fit(df)  # 训练规则
    df_dataScale = scale.transform(df)  # 应用规则
    kmeans = KMeans(n_clusters=3, random_state=123).fit(df_dataScale)  # 构建并训练模型
    print('构建的K-Means模型为：\n', kmeans)
    result = kmeans.predict(df_dataScale)
    print('风险值类别为：', result[0])

    # print('0: ',df[result == 0]['销售额'].min(), df[result == 0]['销售额'].max())
    # print('1: ',df[result == 1]['销售额'].min(), df[result == 1]['销售额'].max())
    # print('2: ',df[result == 2]['销售额'].min(), df[result == 2]['销售额'].max())
    # re1=kmeans.predict([[1682.597334371035]])
    # print(re1[0])

    df_res = pd.DataFrame(kmeans.cluster_centers_.flatten())
    sort_res = df_res.sort_values(by=0)
    sort_res.T.to_csv('./type.csv', header=None, index=None, mode=mode)

    C_i = kmeans.predict(df_dataScale)
    # 还需要知道聚类中心的坐标
    Muk = kmeans.cluster_centers_
    # 画图
    plt.rcParams['font.family'] = 'simhei'
    plt.title('Kmeans聚类', fontsize=20)
    plt.xlabel('风险值')
    plt.ylabel('风险值')
    plt.scatter(df_dataScale[:, 0], df_dataScale[:, 1],
                c=C_i, cmap=plt.cm.Paired, s=0.2)  # plt.cm.Paired,表示两个两个相近色彩输出
    # 画聚类中心
    plt.scatter(Muk[:, 0], Muk[:, 1], marker='*', s=60)
    for i in range(3):
        plt.annotate('中心' + str(i + 1), (Muk[i, 0], Muk[i, 1]))
    plt.show()


# 风险值  销售额    地区名_宜兴市    地区名_新吴区    地区名_梁溪区    地区名_江阴市    地区名_滨湖区    地区名_锡山区    month
if __name__ == '__main__':
    # 读入所有数据
    dd = get_data()
    # 保存风险聚类
    # save_type(dd.iloc[:,:1])
    # 保存销量聚类
    # t = dd['销售额']
    # t2 = dd.iloc[:, 1:3]
    # print(t)
    save_type(dd.iloc[:, :2], mode='a')
