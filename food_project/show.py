# -*- coding: utf-8 -*-
# 使用K-Means算法聚类消费行为特征数据

import pandas as pd
from utils import load_data
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

path = '../data/per_month_sale_and_risk.csv'
data = load_data(path)
data = data[['风险值', '销售额', 'month']]
# 参数初始化
# input_path = 'F:/DataMining/chapter5/consumption_data.xls'  # 销量及其他属性数据
output_path = 'data_type.xls'  # 保存结果的文件名
k = 3  # 聚类的类别
iteration = 500  # 聚类最大迭代次数
# print('data: \n', data)
# print('means: \n', data.mean())
# print('stds: \n', data.std())
data_zs = 1.0 * (data - data.mean()) / data.std()  # 数据标准化
# print(data_zs)
from sklearn.cluster import KMeans

model = KMeans(n_clusters=k, max_iter=iteration)  # 分为k类，n_jobs=1即不并发执行
model.fit(data_zs)  # 开始聚类
print('labels: \n', model.labels_)
print('cluster_centers_: \n', model.cluster_centers_)

# 简单打印结果
r1 = pd.Series(model.labels_).value_counts()  # 将数组格式的labels转换为Series格式再统计各个类别的数目
# r1.index = ['a', 'b', 'c']
print('r1: \n', r1)
r2 = pd.DataFrame(model.cluster_centers_)  # 将二维数组格式的cluster_centers_转换为DataFrame格式
print('r2: \n', r2)
r = pd.concat([r2, r1], axis=1)  # 横向拼接接(0是纵向),将r1变成一列拼接在r2的最右边，所有拼接的列的列名默认从0开始
r.columns = data.columns.tolist() + ['类别数目']  # 重命名表头
print('r: \n', r)
#
# 详细输出原始数据及其类别
output_data = pd.concat([data, pd.Series(model.labels_, index=data.index)], axis=1)  # 详细输出每个样本对应的类别
output_data.columns = list(data.columns) + ['聚类类别']  # 重命名表头
# output_data.to_excel(output_path)  # 保存结果

# 使用TSNE进行数据降维并展示聚类结果

tsne = TSNE()
tsne.fit_transform(data_zs)  # 进行数据降维
# tsne.embedding_可以获得降维后的数据
print('tsne.embedding_: \n', tsne.embedding_)
tsn = pd.DataFrame(tsne.embedding_, index=data.index)  # 转换数据格式
print('tsne: \n', tsne)

# 不同类别用不同颜色和样式绘图
color_style = ['r.', 'go', 'b*']
for i in range(k):
    d = tsn[output_data[u'聚类类别'] == i]
    # dataframe格式的数据经过切片之后可以通过d[i]来得到第i列数据
    plt.plot(d[0], d[1], color_style[i], label='聚类' + str(i + 1))
plt.legend()
plt.show()
