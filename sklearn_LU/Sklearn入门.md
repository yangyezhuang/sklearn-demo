#                          Scikit-learn 入门

## 一、机器学习基础

### 1、机器学习的一般流程

```
(1)定义分析目标
(2)收集数据
(3)数据预处理
(4)数据建模
(5)模型训练
(6)模型评估(参数微调)
(7)模型应用
```

###  2、监督学习与非监督学习

*(1)监督学习(有导师学习)-Supervised learning*
		数据的类标志是已知的
		由训练数据集得到模型，用于新的对象
		----**分类Classification**
*(2)非监督的学习(无导师学习)-Unsupervised learning*
		数据的类标志是未知的
		根据对象之间的距离或相似性进行划分
		----**聚类Clustering**

y'=f(X)

y1'=f(X1)  X1->f->Y1

(X,Y)

Y-Y'

## 二、模型评估指标汇

​      选择正确的度量来评估机器学习模型，我们什么时候评估我们的机器学习模型呢?答案不是只有一次。通常，我们在实际的数据科学工作流中两次使用机器学习模型验证指标:

**(1)模型比较**：为您的任务选择最佳机器学习（ML）模型

**(2)模型改进**：调整超参数

​    为了更清楚地了解这两者之间的区别，可通过机器学习（ML)实现的工作流程来解释：在为任务y设置所有特征X后，您可以准备多个机器学习模型作为候选。

​      那么你怎么才能最终为你的任务选择一个呢?是的，这是使用模型验证度量的第一点。Scikit-learn提供了一些快捷方法来比较模型，比如cross - validation。

​          在您选择了一个准确度最好的机器学习模型后，您将跳转到超参数调优部分，以提高精度和通用性。这里是您将使用这些度量的第二点。

### 1 分类问题

**(1)、混淆矩阵** 

​        混淆矩阵是监督学习中的一种可视化工具，主要用于比较分类结果和实例的真实信息。矩阵中的每一行代表实例的预测类别，每一列代表实例的真实类别。

![混淆矩阵](.\混淆矩阵.png)

```
真正(True Positive , TP)：被模型预测为正的正样本。
假正(False Positive , FP)：被模型预测为正的负样本。
假负(False Negative , FN)：被模型预测为负的正样本。
真负(True Negative , TN)：被模型预测为负的负样本。

真正率(True Positive Rate,TPR)：TPR=TP/(TP+FN)，即被预测为正的正样本数 /正样本实际数。
假正率(False Positive Rate,FPR) ：FPR=FP/(FP+TN)，即被预测为正的负样本数 /负样本实际数。
假负率(False Negative Rate,FNR) ：FNR=FN/(TP+FN)，即被预测为负的正样本数 /正样本实际数。
真负率(True Negative Rate,TNR)：TNR=TN/(TN+FP)，即被预测为负的负样本数 /负样本实际数/2
```

**(2)、准确率（Accuracy）**

准确率是最常用的分类性能指标。
	Accuracy = (TP+TN)/(TP+FN+FP+TN)

即正确预测的正反例数 /总数

**(3)、精确率（Precision）**

精确率容易和准确率被混为一谈。其实，精确率只是针对预测正确的正样本而不是所有预测正确的样本。表现为预测出是正的里面有多少真正是正的。可理解为查准率。
	Precision = TP/(TP+FP)

即正确预测的正例数 /预测正例总数

**(4)、召回率（Recall）**

召回率表现出在实际正样本中，分类器能预测出多少。与真正率相等，可理解为查全率。
	Recall = TP/(TP+FN)，即正确预测的正例数 /实际正例总数

**(5)、F1 score**

F值是精确率和召回率的调和值，更接近于两个数较小的那个，所以精确率和召回率接近时，F值最大。很多推荐系统的评测指标就是用F值的。
	2/F1 = 1/Precision + 1/Recall

**(6)、ROC曲线**

逻辑回归里面，对于正负例的界定，通常会设一个阈值，大于阈值的为正类，小于阈值为负类。如果我们减小这个阀值，更多的样本会被识别为正类，提高正类的识别率，但同时也会使得更多的负类被错误识别为正类。为了直观表示这一现象，引入ROC。根据分类结果计算得到ROC空间中相应的点，连接这些点就形成ROC curve，横坐标为False Positive Rate(FPR假正率)，纵坐标为True Positive Rate(TPR真正率)。一般情况下，这个曲线都应该处于(0,0)和(1,1)连线的上方,如图：

<img src=".\ROC曲线.png" alt="ROC曲线" style="zoom: 80%;" />

```
ROC曲线中的四个点和一条线:
点(0,1)：即FPR=0, TPR=1，意味着FN＝0且FP＝0，将所有的样本都正确分类。
点(1,0)：即FPR=1，TPR=0，最差分类器，避开了所有正确答案。
点(0,0)：即FPR=TPR=0，FP＝TP＝0，分类器把每个实例都预测为负类。
点(1,1)：分类器把每个实例都预测为正类。
总之：ROC曲线越接近左上角，该分类器的性能越好。而且一般来说，如果ROC是光滑的，那么基本可以判断没有太大的overfitting
```

**(7)、AUC**

AUC（Area Under Curve）被定义为ROC曲线下的面积(ROC的积分)，通常大于0.5小于1。随机挑选一个正样本以及一个负样本，分类器判定正样本的值高于负样本的概率就是 AUC 值。AUC值(面积)越大的分类器，性能越好，如图：

**(8)、PR曲线**

PR曲线的横坐标是精确率P，纵坐标是召回率R。评价标准和ROC一样，先看平滑不平滑（蓝线明显好些）。一般来说，在同一测试集，上面的比下面的好（绿线比红线好）。当P和R的值接近时，F1值最大，此时画连接(0,0)和(1,1)的线，线和PRC重合的地方的F1是这条线最大的F1（光滑的情况下），此时的F1对于PRC就好像AUC对于ROC一样。一个数字比一条线更方便调型。

![PRO曲线](.\PRO曲线.png)

```
有时候模型没有单纯的谁比谁好（比如图二的蓝线和青线），所以选择模型还是要结合具体的使用场景。下面是两个场景：
1) 地震的预测 对于地震的预测，我们希望的是RECALL非常高，也就是说每次地震我们都希望预测出来。这个时候我们可以牺牲PRECISION。情愿发出1000次警报，把10次地震都预测正确了，也不要预测100次对了8次漏了两次。
2) 嫌疑人定罪 基于不错怪一个好人的原则，对于嫌疑人的定罪我们希望是非常准确的。即时有时候放过了一些罪犯（recall低），但也是值得的。

对于分类器来说，本质上是给一个概率，此时，我们再选择一个CUTOFF点（阀值），高于这个点的判正，低于的判负。那么这个点的选择就需要结合你的具体场景去选择。反过来，场景会决定训练模型时的标准，比如第一个场景中，我们就只看RECALL=99.9999%（地震全中）时的PRECISION，其他指标就变得没有了意义。
当正负样本数量差距不大的情况下，ROC和PR的趋势是差不多的，但是在正负样本分布极不均衡的情况下，PRC比ROC更能真实的反映出实际情况，因为此时ROC曲线看起来似乎很好，但是却在PR上效果一般。
复制代码
```

### 2、回归问题

​        拟合（回归）问题比较简单，所用到的衡量指标也相对直观。假设yi是第i个样本的真实值，ŷi是对第i个样本的预测值。

**(1)、 平均绝对误差(MAE)**

平均绝对误差MAE（Mean Absolute Error）又被称为L1范数损失（L1-norm loss）：
$$
\boldsymbol{MAE(y,\hat y)} = \boldsymbol{\frac{1}{n_{samples}} \sum_{i=1}^{n_{samples}} \left|y_i-\hat y_i\right|}
$$
**(2) 、平均平方误差（MSE）**

平均平方误差MSE（Mean Squared Error）又被称为L2范数损失（L2-norm loss）：

$$
\boldsymbol{MSE(y,\hat y)} = \boldsymbol{\frac{1}{n_{samples}} \sum_{i=1}^{n_{samples}} \left|y_i-\hat y_i\right|^2}
$$
**(3)、均方根误差（RMSE）**

RMSE虽然广为使用，但是其存在一些缺点，因为它是使用平均误差，而平均值对异常点（outliers）较敏感，如果回归器对某个点的回归值很不理性，那么它的误差则较大，从而会对RMSE的值有较大影响，即平均值是非鲁棒的。
$$
\boldsymbol{RMSE(y,\hat y)} = \boldsymbol{ \sqrt{\frac{1}{n_{samples}} \sum_{i=1}^{n_{samples}} \left|y_i-\hat y_i\right|^2}}
$$
**(4)、决定系数**

决定系数（Coefficient of determination）又被称为R2，也称为判*定系数*，也称为拟合优度。表示可根据自变量的变异来解释因变量的变异部分。R²最大值为1。R²的值越接近1，说明回归直线对观测值的拟合程度越好；反之，R²的值越小，说明回归直线对观测值的拟合程度越差。

## 三、Scikit-learn 机器学习库

###  1、什么是SKlearn
​	    SciKit learn的简称是SKlearn，是一个python库，专门用于机器学习的模块。以下是它的官方网站，文档等资源都可以在里面找到http://scikit-learn.org/stable/#，官方文档中文版：https://sklearn.apachecn.org/和https://www.scikitlearn.com.cn/
### 2、SKlearn包含的机器学习方式
​		分类，回归，无监督，数据降维，数据预处理等等，包含了常见的大部分机器学习方法。
### 3、如何正确选择模型
​         SKlearn给出了如何选择正确的方法：

![ml_map](.\ml_map.png)

图中对于什么样的问题，采用什么样的方法给出了清晰的描述，包括数据量不同的区分。

### 4、Klearn的强大数据库
​		数据库网址：http://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets
里面包含了很多数据，学习过程中可以直接拿来使用，以做便理解各种模型与算法:

​                            **sklearn.datasets:  Datasets**



**例如：**

```
A.鸢尾花数据集
     打开里面的鸢尾花数据集，我们可以看到页面上同样有调用示例：

#调用模块
from sklearn.datasets import load_iris
data = load_iris()
#导入数据和标签
data_X = load_data.data
data_y = load.data.target

B.波士顿房价数据集
#换种方式调用模块，注意区别
from sklearn import datasets
loaded_data = datasets.load_boston()
#导入数据
data_X = loaded_data.data
data_y = loaded_data.target
```

而且在SKlearn官网，对于每一个数据集，在后面都给出了，使用该数据集的示例，例如Boston房价数据集：

​                        **Examples using  sklearn.datasetes.load_boston**

## 四、SKlearn中的通用学习模式

​         SKlearn中学习模式的调用，有很强的统一性，很多都是类似的，学会一个，其他基本差不多。

### 1、鸢尾花数据集

​     本例，调用鸢尾花数据集，然后使用K近邻的方法对其进行预测。

```
# 0. 导入相关模块
from sklearn.model_selection import train_test_split
from sklearn import datasets
# 导入K近邻分类器函数
from sklearn.neighbors import KNeighborsClassifier

# 1. 加载数据
iris = datasets.load_iris()

# 2. 数据预处理
#导入数据和标签
iris_X = iris.data
iris_y = iris.target
#划分为训练集和测试集数据
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)
#print(y_train)

# 3. 构建模型
# 创建KNN分类器(用于回归)
knn_model = KNeighborsClassifier()

# 4. 模型训练 n*m (L), n*1
knn_model.fit(X_train,y_train)

# 5. 使用训练好的KNN进行数据预测
pre_y = knn_model.predict(X_test)

# 6. 结果展示与模型评估
# 6.1结果展示
print(pre_y)
print(y_test)

# 解决matplotlib中的中文显示问题
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
# 图形展示，采用点线图。
plt.plot(y_test,label= '实际值')
plt.plot(pre_y,label= '预测值')

plt.legend()
plt.show()

# 6.2 模型评估
#使用均方误差对其进行打分，输出精确度，
#即利用训练好的模型对X_test进行预测，得到预测后，和原本标签进行比较
print("MSE: %.4f" % knn_model.score(X_test,y_test))

# R2：决定系数
from sklearn.metrics import r2_score
r2_score(y_test, pre_y) # KNN Regression

# 7. 模型参数
#取出之前定义的模型的参数
print(knn_model.get_params())
```

### 2、波士顿房价数据集
​		本例，调用房价数据集，然后使用线性回归的方法对其进行预测。

```
# 0. 导入相关模块
#matplotlib是python专门用于画图的库
import matplotlib.pyplot as plt
from sklearn import datasets
# 导入线性回归模型函数
from sklearn.linear_model import LinearRegression

# 1. 加载数据集
# 这里将全部数据用于训练，并没有对数据进行划分，上例中将数据划分为训练和测试数据，后面会讲到交叉验证
loaded_data = datasets.load_boston()

# 2. 数据预处理
data_X = loaded_data.data
data_y = loaded_data.target
#data_X是训练数据
#data_y是导入的标签数据

# 3. 创建线性回归模型
linear_model = LinearRegression()

# 4. 模型训练数据，得出参数
linear_model.fit(data_X, data_y)

# 5. 模型预测
#利用模型，对新数据，进行预测，与原标签进行比较
pre_y = linear_model.predict(data_X[0:4,:])

# 6. 结果展示与模型评估
# 6.1结果展示
print(pre_y)
print(data_y[:4])

# 解决matplotlib中的中文显示问题
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
# 图形展示，采用点线图。
plt.plot(data_y[:4],label= '实际值')
plt.plot(pre_y,label= '预测值')

plt.legend()
plt.show()

# 6.2 模型评估
# 使用均方误差对其进行打分，输出精确度，
# 即利用训练好的模型对data_X进行预测，得到预测后，和原本标签进行比较
print("MSE: %.4f" % linear_model.score(data_X,data_y))

# 7. 模型参数 y = a*x + b
#输出模型的两个参数，在这里分别指的是，线性回归模型的斜率和截距
print(linear_model.coef_)
print(linear_model.intercept_)

#取出之前定义的模型的参数
print(linear_model.get_params())
```

## 五、Scikit-learn 中的模型

**回归**

       本节主要讲述一些用于回归的方法，其中目标值 y 是输入变量 x 的线性组合。 数学概念表示为：如果 y 是预测值，那么有：
$$
\boldsymbol{\hat y}(w,x) = \boldsymbol{w_0+w_1x_1+...+w_p} \boldsymbol{x_p}
$$


在整个模块中，我们定义向量 
$$
w =(w_1,w_2,...,w_p)
$$
作为 系数`coef_` ，定义w0 作为截距`intercept_` 。

如果需要使用广义线性模型进行分类，请参阅 logistic 回归。



### 1.LinearRegression

       线性回归通常用于根据连续变量估计实际数值（房价、呼叫次数、总销售额等）。我们通过拟合最佳直线来建立自变量和因变量的关系。这条最佳直线叫做回归线，并且用 Y= a *X + b 这条线性等式来表示。

在这个等式中：

    ●Y：因变量
    ●a：斜率
    ●X：自变量
    ●b ：截距

系数 a 和 b 可以通过最小二乘法获得： 

         拟合一个带有系数 w = (w1, ..., wp) 的线性模型，使得数据集实际观测数据和预测数据（估计值）之间的残差平方和最小。其数学表达式为:
$$
\underset{w}{min\,} {|| X w - y||_2}^2
$$
        线性回归的两种主要类型是一元线性回归和多元线性回归。一元线性回归的特点是只有一个自变量。多元线性回归的特点正如其名，存在多个自变量。找最佳拟合直线的时候，你可以拟合到多项或者曲线回归。这些就被叫做多项或曲线回归。


```
from sklearn.linear_model import LinearRegression         # 导入线性回归模型 #
# 1.读取数据集 trainX,test_x,trainY,test_y

# 2.构建模型
module = LinearRegression()

# 3.模型训练
module.fit(trainX, trainY)

# 4.模型预测
pre_y = module.predict(testX)

# 5.*模型评价
module.score(x, y)

函数score(self, X, y, sample_weight=None)的作用是返回该次预测的系数R^2((coefficient of determination)决定系数) ,其中:
     R^2 =（1-u/v）。
     u=((y_true - y_pred) ** 2).sum()
     v=((y_true - y_true.mean()) ** 2).sum()
#其中可能得到的最好的分数是1，并且可能是负值（因为模型可能会变得更加糟糕）。当一个模型不论输入何种特征值，其总是输出期望的y的时候，此时返回0。
```

普通最小二乘法的复杂度
        该方法使用 X 的奇异值分解来计算最小二乘解。如果 X 是一个形状为 (n_samples, n_features)的矩阵，设 
$$
n_{samples} \geq n_{features}
$$
 , 则该方法的复杂度为 
$$
O(n_{samples} n_{fearures}^2)
$$

### 2. 岭回归

      Ridge回归通过对系数的大小施加惩罚来解决 普通最小二乘法 的一些问题。 岭系数最小化的是带罚项的残差平方和:
$$
\underset{w}{min\,} {{|| X w - y||_2}^2 + \alpha {||w||_2}^2}
$$
其中， 
$$
\alpha \geq 0
$$
是控制系数收缩量的复杂性参数： a 的值越大，收缩量越大，模型对共线性的鲁棒性也更强。

        与其他线性模型一样， Ridge 用 `fit` 方法完成拟合，并将模型系数存储在其 `coef_` 成员中:

```
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
print(reg.coef_)  #输出拟合后的系数
```



### 3.LogisticRegression

​          别被它的名字迷惑了！这是一个分类算法而不是一个回归算法。该算法可根据已知的一系列因变量估计离散数值（比方说二进制数值 0 或 1 ，是或否，真或假）。简单来说，它通过将数据拟合进一个逻辑函数来预估一个事件出现的概率。因此，它也被叫做逻辑回归。因为它预估的是概率，所以它的输出值大小在 0 和 1 之间（正如所预计的一样）。

​            从数学上看，在结果中，概率的对数使用的是预测变量的线性组合模型。


```
from sklearn.linear_model import LogisticRegression         # 逻辑回归 #
module = LogisticRegression()
module.fit(x, y)
module.score(x, y)
module.predict(test)
```

### 4.KNN
该算法可用于分类问题和回归问题。然而，在业界内，K – 最近邻算法更常用于分类问题。K – 最近邻算法是一个简单的算法。它储存所有的案例，通过周围k个案例中的大多数情况划分新的案例。根据一个距离函数，新案例会被分配到它的 K 个近邻中最普遍的类别中去。

这些距离函数可以是欧式距离、曼哈顿距离、明式距离或者是汉明距离。前三个距离函数用于连续函数，第四个函数（汉明函数）则被用于分类变量。如果 K=1，新案例就直接被分到离其最近的案例所属的类别中。有时候，使用 KNN 建模时，选择 K 的取值是一个挑战。

在选择使用 KNN 之前，你需要考虑的事情：

    ●KNN 的计算成本很高。
    ●变量应该先标准化（normalized），不然会被更高范围的变量偏倚。
    ●在使用KNN之前，要在野值去除和噪音去除等前期处理多花功夫。

```
from sklearn.neighbors import KNeighborsClassifier     #K近邻#
from sklearn.neighbors import KNeighborsRegressor
module = KNeighborsClassifier(n_neighbors=6)
module.fit(x, y)
predicted = module.predict(test)
predicted = module.predict_proba(test)
```

### 5.SVM
这是一种分类方法。在这个算法中，我们将每个数据在N维空间中用点标出（N是你所有的特征总数），每个特征的值是一个坐标的值。

举个例子，如果我们只有身高和头发长度两个特征，我们会在二维空间中标出这两个变量，每个点有两个坐标（这些坐标叫做支持向量）。

现在，我们会找到将两组不同数据分开的一条直线。两个分组中距离最近的两个点到这条线的距离同时最优化。

上面示例中的黑线将数据分类优化成两个小组，两组中距离最近的点（图中A、B点）到达黑线的距离满足最优条件。这条直线就是我们的分割线。接下来，测试数据落到直线的哪一边，我们就将它分到哪一类去。

将这个算法想作是在一个 N 维空间玩 JezzBall。需要对游戏做一些小变动：

    ●比起之前只能在水平方向或者竖直方向画直线，现在你可以在任意角度画线或平面。
    ●游戏的目的变成把不同颜色的球分割在不同的空间里。
    ●球的位置不会改变。

```
from sklearn import svm                                #支持向量机#
module = svm.SVC()
module.fit(x, y)
module.score(x, y)
module.predict(test)
module.predict_proba(test)
```

### 6.naive_bayes
```
from sklearn.naive_bayes import GaussianNB            #朴素贝叶斯分类器#
module = GaussianNB()
module.fit(x, y)
predicted = module.predict(test)
```

### 7.DecisionTree
```
from sklearn import tree                              #决策树分类器#
module = tree.DecisionTreeClassifier(criterion='gini')
module.fit(x, y)
module.score(x, y)
module.predict(test)
```

### 8.K-Means
```
from sklearn.cluster import KMeans                    #kmeans聚类#
module = KMeans(n_clusters=3, random_state=0)
module.fit(x, y)
module.predict(test)
```

### 9.RandomForest
```
from sklearn.ensemble import RandomForestClassifier  #随机森林#
from sklearn.ensemble import RandomForestRegressor
module = RandomForestClassifier()
module.fit(x, y)
module.predict(test)
```

### 10.GBDT
```
from sklearn.ensemble import GradientBoostingClassifier      #Gradient Boosting 和 AdaBoost算法#
from sklearn.ensemble import GradientBoostingRegressor
module = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0)
module.fit(x, y)
module.predict(test)
```

## 六、模型验证

### 1、 交叉验证用于模型比较

训练/测试拆分和交叉验证的可视化表示：

<img src=".\交叉验证.png" alt="交叉验证" style="zoom: 50%;" />

```
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

我们拆分数据的原因和方式的起点是泛化。因为我们构建机器学习模型的目标是使用未来未知数据的真实实现。因此，我们不需要过度拟合过去数据的无用模型。

### 2、交叉验证方法

  <img src=".\交叉验证_1.png" alt="交叉验证_1" style="zoom: 50%;" />

```python
# Decision Tree Classifieras for estimator
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
```

**（1）cross_val_score**：最简单的编码方法

我们可以通过参数“cv”来决定数据拆分的数量。通常5被认为是标准拆分数。

```python
# X, y = wine.data, wine.target
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y, cv=5)
print(scores) # cv = number of splited data
print(scores.mean())
```

**（2）cross_validate**：可自定义

```bash
scoring = ['precision_macro', 'recall_macro']
scores = cross_validate(clf, X, y, scoring=scoring, cv=5)
print(scores)
```



## 七、Scikit-learn 中的各种回归用法

基本回归：线性、决策树、SVM、KNN

集成方法：随机森林、Adaboost、GradientBoosting、Bagging、ExtraTrees

### 1. 数据准备

为了实验用，写一个二元函数，y=0.5*np.sin(x1)+ 0.5*np.cos(x2)+0.1*x1+3。其中x1的取值范围是0~50，x2的取值范围是-10~10，x1和x2的训练集一共有500个，测试集有100个。其中，在训练集的上加了一个-0.5~0.5的噪声。生成函数的代码如下：

```
def f(x1, x2):
  y = 0.5 * np.sin(x1) + 0.5 * np.cos(x2) + 0.1 * x1 + 3
  return y
  
def load_data():
  x1_train = np.linspace(0,50,500)
  x2_train = np.linspace(-10,10,500)
  data_train = np.array([[x1,x2,f(x1,x2) + (np.random.random(1)-0.5)] for x1,x2 in zip(x1_train, x2_train)])
  x1_test = np.linspace(0,50,100)+ 0.5 * np.random.random(100)
  x2_test = np.linspace(-10,10,100) + 0.02 * np.random.random(100)
  data_test = np.array([[x1,x2,f(x1,x2)] for x1,x2 in zip(x1_test, x2_test)])
  return data_train, data_test
```

###　2. scikit-learn的简单使用

scikit-learn非常简单，只需实例化一个算法对象，然后调用fit()函数就可以了，fit之后，就可以使用predict()函数来预测了，然后可以使用score()函数来评估预测值和真实值的差异，函数返回一个得分。

完整程式化代码为：

```
import numpy as np
import matplotlib.pyplot as plt

###########1.数据生成部分##########
def f(x1, x2):
  y = 0.5 * np.sin(x1) + 0.5 * np.cos(x2) + 3 + 0.1 * x1
  return y

def load_data():
  x1_train = np.linspace(0,50,500)
  x2_train = np.linspace(-10,10,500)
  data_train = np.array([[x1,x2,f(x1,x2) + (np.random.random(1)-0.5)] for x1,x2 in zip(x1_train, x2_train)])
  x1_test = np.linspace(0,50,100)+ 0.5 * np.random.random(100)
  x2_test = np.linspace(-10,10,100) + 0.02 * np.random.random(100)
  data_test = np.array([[x1,x2,f(x1,x2)] for x1,x2 in zip(x1_test, x2_test)])
  return data_train, data_test
train, test = load_data()
x_train, y_train = train[:,:2], train[:,2] #数据前两列是x1,x2 第三列是y,这里的y有随机噪声
x_test ,y_test = test[:,:2], test[:,2] # 同上,不过这里的y没有噪声

###########2.回归部分##########
def try_different_method(model):
  model.fit(x_train,y_train)
  score = model.score(x_test, y_test)
  result = model.predict(x_test)
  plt.figure()
  plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
  plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
  plt.title('score: %f'%score)
  plt.legend()
  plt.show()

###########3.具体方法选择##########
####3.1决策树回归####
from sklearn import tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()

####3.2线性回归####
from sklearn import linear_model
model_LinearRegression = linear_model.LinearRegression()

####3.3SVM回归####
from sklearn import svm
model_SVR = svm.SVR()

####3.4KNN回归####
from sklearn import neighbors
model_KNeighborsRegressor = neighbors.KNeighborsRegressor()

####3.5随机森林回归####
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)#这里使用20个决策树

####3.6Adaboost回归####
from sklearn import ensemble
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)#这里使用50个决策树

####3.7GBRT回归####
from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树

####3.8Bagging回归####
from sklearn.ensemble import BaggingRegressor
model_BaggingRegressor = BaggingRegressor()

####3.9ExtraTree极端随机树回归####
from sklearn.tree import ExtraTreeRegressor
model_ExtraTreeRegressor = ExtraTreeRegressor()

###########4.具体方法调用部分##########
try_different_method(model_DecisionTreeRegressor)


```

