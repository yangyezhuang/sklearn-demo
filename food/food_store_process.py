import numpy as np
import csv
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

food_recordfilename = './food_record_ansi.csv'

daily_risks = {}
store_risks = {}

with open(food_recordfilename, 'r') as fr:
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
# print("RR: ", daily_risks)


'''
    直方图展示每日所有店家风险的数值分布.
    直方图中，横、纵坐标的意义:
        横轴：风险值；纵轴：风险值落在此区间的商家个数
'''


def risk_histogram(day):
    risk_set = daily_risks[day]
    # print("Num of risks: ", len(risk_set))
    a = np.hstack(np.array(risk_set))
    _ = plt.hist(a, bins=15)  # bin数目设置为15，可改
    plt.title("Histogram of risk on " + day)
    plt.show()


def risk_prediction(store_id):
    # split into train and test sets
    # X = series.values
    X = store_risks[store_id]
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train];
    print(history)
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


if __name__ == '__main__':
    day_test = "2019-02-01"
    # 某天所有商家风险的数值分布
    risk_histogram(day_test)

    # 预测
    # store_test = "3d8ab846-7725-11ea-98c9-525400c74cae"
    # risk_prediction(store_test)
