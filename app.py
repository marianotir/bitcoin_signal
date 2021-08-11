


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn import datasets, linear_model
import os

import datetime as dt
from datetime import datetime
from datetime import datetime, timedelta

from pycoingecko import CoinGeckoAPI




app = Flask(__name__)

cg = CoinGeckoAPI()


def get_coin_history(coin,n_days):

   price_list = []
   for n in range(n_days,0,-1):
       date = datetime.date(datetime.today() - timedelta(days=n))
       date_string = str(date.day) + '-' + str(date.month) + '-' + str(date.year)
       price = cg.get_coin_history_by_id(id=coin,date=date_string, localization='false')['market_data']['current_price']['usd']
       price_list.append(price)

   intermediate_dictionary = {'Price':price_list}

   df_coin = pd.DataFrame(intermediate_dictionary)

   return df_coin


def time_series_to_ml(data,rolling_window):
    for i in range(1,rolling_window):
        column_name = 't-'+str(i)
        data[column_name] = data['Price'].shift(i)
    return data


model_name = 'model_1.pkl'
model = pickle.load(open(model_name,'rb'))


@app.route('/')
def home():
    return "<p><h1> Get Bitcoin Prediction <h1> </p>"


@app.route('/api_v1/predict/<string:coin_name>/')
def predict(coin_name):

    coin = coin_name
    hist_days = 15
    rolling_window = 8

    if coin_name == 'Bitcoin' or coin_name == 'bitcoin':

        coin = 'bitcoin'
        data = get_coin_history(coin,hist_days)
        data = data[['Price']]
        data = time_series_to_ml(data,rolling_window)

        data.dropna(axis=0, inplace=True)

        X_pred  = data.iloc[-1:,0:-1]

        y_pred = model.predict(X_pred)

        prediction = int(y_pred[0])
        price = int(X_pred['Price'])

        if prediction>price:
            signal = 'buy'
        else:
            signal = 'sell'

        return jsonify({'Coin': coin,
                        'Price':price,
                        'Prediction':prediction,
                        'Signal':signal})

    else:

        return "<p> Only bitcoin predictions allowed in this version </p>"




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)