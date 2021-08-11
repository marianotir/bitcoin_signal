
# -----------------
# import modules
# -----------------
import requests
import pandas as pd
import json
import datetime as dt
import os
import time
from datetime import datetime
from datetime import datetime, timedelta

import pickle

from pycoingecko import CoinGeckoAPI


from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


# config
cg = CoinGeckoAPI()
url = 'https://api.coingecko.com/api/v3/coin/truebit'


ROLLING_WINDOW = 8
MODEL_NAME = 'model_1'


def get_coin_history(coin,n_days):

   price_list = []
   date_list = []
   for n in range(n_days,0,-1):
       # coin = 'bitcoin'
       date = datetime.date(datetime.today() - timedelta(days=n))
       date_string = str(date.day) + '-' + str(date.month) + '-' + str(date.year)
       price = cg.get_coin_history_by_id(id=coin,date=date_string, localization='false')['market_data']['current_price']['usd']
       price_list.append(price)
       date_list.append(date)

   intermediate_dictionary = {'Date':date_list, 'Price':price_list}

   df_coin = pd.DataFrame(intermediate_dictionary)

   return df_coin


def time_series_to_ml(data,rolling_window):
    for i in range(1,rolling_window):
        column_name = 't-'+str(i)
        data[column_name] = data['Price'].shift(i)
    return data


def coin_model_train(data,rolling_window,model_name):

    data = data[['Price']]
    data = time_series_to_ml(data,rolling_window)

    data.dropna(axis=0, inplace=True)

    X_train = data.iloc[:,1:]
    y_train = data[['Price']]

    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)

    name = model_name + '.pkl'
    pickle.dump(model, open(name,'wb'))

    return model


def get_model(model_name):

    name = model_name + '.pkl'
    model = pickle.load(open(name,'rb'))

    return model


def coin_predict(data,rolling_window,model_name):

    data = data[['Price']]
    data = time_series_to_ml(data,rolling_window)

    data.dropna(axis=0, inplace=True)

    X_pred  = data.iloc[-1:,0:-1]

    model = get_model(model_name)

    y_pred = model.predict(X_pred)

    pred = int(y_pred[0])

    return pred


# store data
def store_data(data,coin,hist_days):
    name = 'history_' + coin + '_' + str(hist_days) + '.csv'
    data.to_csv(rf'{os.getcwd()}\{name}', index = False)



def main():

    coin = 'bitcoin'
    hist_days = 50
    data = get_coin_history(coin,hist_days)

    store_data(data,coin,hist_days)

    rolling_window = ROLLING_WINDOW
    model_name = MODEL_NAME

    coin_model_train(data,rolling_window,model_name)

    pred = coin_predict(data,rolling_window,model_name)

    text = 'Prediction for ' + coin + ': ' + str(pred)
    print(text)



if __name__ == "__main__":
    main()

