# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
import csv

filename = './food_record_ansi.csv'
daily_risks = {}
store_risks = {}

with open(filename, 'r') as fr:
    csv_reader = csv.reader(fr)
    next(csv_reader)
    for line in csv_reader:
        store = line[0]
        day = line[-2]
        if (day not in daily_risks):
            daily_risks[day] = [float(line[-1])]
        else:
            daily_risks[day].append(float(line[-1]))

        if (store not in store_risks):
            store_risks[store] = [float(line[-1])]
        else:
            store_risks[store].append(float(line[-1]))


def risk_prediction(store_id):
    # split into train and test sets
    # X = series.values
    X = store_risks[store_id]
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]

    predictions = list()

    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    # evaluate forecasts
    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)
    # plot forecasts against actual outcomes
    plt.plot(test)
    plt.plot(predictions, color='red')
    plt.show()


x1, y1 = make_gaussian_quantiles(
    cov=2.0, n_samples=500, n_features=2, n_classes=2, random_state=1)
x2, y2 = make_gaussian_quantiles(
    mean=(3, 3), cov=1.5, n_samples=500, n_features=2, n_classes=2, random_state=1)

X = np.concatenate((x1, x2))  # default is in the direction of axis x
y = np.concatenate((y1, -y2 + 1))
# print(X.shape)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
# plt.show()

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=300, learning_rate=0.8)
bdt.fit(X, y)

print("score:", bdt.score(X, y))
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
print(xx.shape)
print(xx.ravel().shape)

# np.c_  short for column concatenate np.r_ short for row concatenate
Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
plt.show()
