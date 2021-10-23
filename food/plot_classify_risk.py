import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import classification_report
# from sklearn.multiclass import OneVsRestClassifier

# Matplotlib中文显示问题
# plt.rcParams['font.family'] = 'simhei' 
# plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

food = pd.read_csv('../data/per_month_sale_and_risk.csv')

# Step 2. 数据预处理
# Step 2.1 把对属性start_time和place进行编码
start_time = pd.get_dummies(food.开始时间)
place = pd.get_dummies(food.地区名)

# 地区+月份+销售额
data = pd.concat([place, start_time], axis=1)
data = pd.concat([data, food['销售额']], axis=1)
# 地区
# data = place
# 月份
# data = start_time
# 销售额
# data = food['销售额'].to_frame()
# 地区+月份
# data = pd.concat([place,start_time],axis=1)
# 地区+销售额
# data = pd.concat([place,food['销售额']],axis=1)
# 月份+销售额
# data = pd.concat([start_time,food['销售额']],axis=1)

# 分类 风险值大于0.7为高风险，数值为2，在0.7-0.4为中风险，数值为1，小于0.4为低风险，数值为0
riskClass = pd.DataFrame(np.ones(len(food)), columns=['riskClass'])
riskClass['riskClass'][food['风险值'] > 0.7] = 2
riskClass['riskClass'][food['风险值'] < 0.3] = 0

# Step 2.2 数据归一化处理
data = (data - data.min(0)) / (data.max(0) - data.min(0))

# Step 3.划分训练集和验证集 
l = len(data)
test_index = [j for j in range(0, l, 5)]
all_index = [j for j in range(l)]
train_index = list(set(all_index).difference(set(test_index)))

# 验证集
test_x = data.loc[test_index]
test_y = riskClass.riskClass.loc[test_index]
# 训练集
train_x = data.loc[train_index]
train_y = riskClass.riskClass.loc[train_index]

# Step 4. 最小二乘训练与预测    
treeDecision = tree.DecisionTreeClassifier()  # 构建模型(线性回归)
model = treeDecision.fit(train_x, train_y)  # 模型训练

pre_tree = model.predict(test_x)  # 预测，结果存入pre中
print('使用DecisionTree模型的结果:')
print(classification_report(test_y, pre_tree, target_names=['calss0', 'calss1', 'calss2']))

# Step5 画散点图
# Step5.1 数据准备
plt_data = food.loc[test_index]
plt_data = plt_data.reset_index(drop=True)
plt_data = pd.concat([plt_data, pd.DataFrame(columns=['pre_tree'], data=pre_tree)], axis=1)
plt_data = plt_data.rename(columns={'地区名': 'place', '开始时间': 'start_time', '销售额': 'sell'})  # 改成英语名

# Step5.2 画图

# 设置Seaborn主题，有5个适用于不同的应用和人群偏好：
# darkgrid 黑色网格（默认）
# whitegrid 白色网格
# dark 黑色背景
# white 白色背景
# ticks 应该是四周都有刻度线的白背景
sns.set(style="whitegrid")

# 解决Seaborn中文显示问题
sns.set(font='SimHei')

# Seaborn分布散点图
# x：设置分组统计字段
# y：设置分布统计字段
# jitter：当数据点重合较多时，可用该参数做一些调整
# dodge：控制组内分类是否彻底分拆
# hue：内部的分类
# order：对x参数所选字段内的类别进行排序以及筛选
sns.stripplot(x='place', y='sell', hue='pre_tree', data=plt_data, jitter=0.3)
plt.show()
