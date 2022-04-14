import pandas as pd
from utils import load_data
from sklearn.cluster import KMeans


# 聚类函数，并实现排序、保存
def save_type(df, count=3, mode='w'):
    # 创建KMeans类对象
    kmeans = KMeans(count, 'k-means++')
    # 调用fit方法
    kmeans.fit(df)
    df_res = pd.DataFrame(kmeans.cluster_centers_.flatten())
    sort_res = df_res.sort_values(by=0)
    sort_res.T.to_csv('type.csv', header=None, index=None, mode=mode)


# 风险值  销售额    地区名_宜兴市    地区名_新吴区    地区名_梁溪区    地区名_江阴市    地区名_滨湖区    地区名_锡山区    month
if __name__ == '__main__':
    # 读入所有数据
    path = '../data/per_month_sale_and_risk.csv'
    dd = load_data(path)
    # 保存风险聚类
    save_type(dd.iloc[:, 0:1])
    # 保存销量聚类
    save_type(dd.iloc[:, 1:2], count=3, mode='a')

    # df=pd.DataFrame(np.arange(12).reshape(3,4))
    # # print(df)
    # # print(df[0])
    # print(df.iloc[:,0:4])
