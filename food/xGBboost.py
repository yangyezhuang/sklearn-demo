# -*- coding: utf-8 -*-


# 时间系列模型
# forecast monthlybirths with xgboost 
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from matplotlib import pyplot


# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))

    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


# split a univariatedataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]


# fit an xgboost model and make a one step prediction
def xgboost_forecast(train, testX):
    # transform list into array 
    train = asarray(train)  # split into input and output columns

    trainX, trainy = train[:, : -1], train[:, -1]
    # fit model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(trainX, trainy)

    # make a one-step prediction     
    yhat = model.predict(asarray([testX]))

    print(yhat)
    return yhat[0]


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)

    print("train:", train)
    print("test:", test)

    # seed history with training dataset
    history = [x for x in train]

    # step over each time-step in the testset
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, : -1], test[i, -1]

        # fit model on history and make a prediction
        yhat = xgboost_forecast(history, testX)

        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])

        # summarize progress
        print('>expected=%.1f,predicted=%.1f' % (testy, yhat))

    # estimate prediction error
    error = mean_absolute_error(test[:, -1], predictions)

    return error, test[:, -1], predictions


if (__name__ == "__main__"):
    # load the dataset,数据url:https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv
    # series =read_csv( 'daily-total-female-births.csv', header= 0, index_col= 0) 
    series = read_csv('per_month_sale_and_risk.csv')

    # 店铺ID
    store_id = "3d8ab846-7725-11ea-98c9-525400c74cae"

    series = series.loc[series['店铺ID'] == store_id, ['开始时间', '销售额', '风险值']]

    series['开始时间'] = series.开始时间.str[5:7].astype('int')

    values = series.values
    # transform the timeseries data into supervised learning
    # 跳过了下面的调用
    # data =series_to_supervised(values, n_in= 3) 
    # print(data)

    # evaluate
    mae, y, yhat = walk_forward_validation(values, 9)

    print('MAE: %.3f' % mae)
    # plot expected vs preducted
    pyplot.plot(y, label='Expected')
    pyplot.plot(yhat, label='Predicted')

    pyplot.legend()
    pyplot.show()
