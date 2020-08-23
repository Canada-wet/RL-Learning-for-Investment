import pandas as pd
import numpy as np
import yfinance as yf
from preprocessing.data_engin import *
from datetime import date
from configs.inputs import path


def download(ticker, start='2000-01-01', end=date.today(), save=True, **kwargs):
    df = yf.download(ticker, start, end)
    if save:
        df.to_csv(path + "data/daily/" + ticker +".csv")
        print(ticker + ".csv is saved in ../data/daily/")
    return df


def load_data(ticker, intraday, **kwargs):
    if intraday:
        df = data_dispose(ticker)
    else:
        try:
            df = pd.read_csv(path + "data/daily/" + ticker.upper() + '.csv')
            df.set_index("Date", inplace=True)
            df.index = pd.to_datetime(df.index)
            if kwargs:
                df = df[(df.index >= kwargs["start"]) & (df.index <= kwargs["end"])]
        except:
            df = download(ticker, save=True)
    # print (df)
    return df


def load_series(ticker, intraday, **kwargs):
    OHLCV_df = load_data(ticker, intraday, **kwargs)
    OHLCV_df.dropna(inplace=True)
    if "Close" in OHLCV_df.columns:
        OHLCV_df.drop(["Close"], axis=1, inplace=True)
    OHLCV_df['Adj Close'] = OHLCV_df['Adj Close']/OHLCV_df['Adj Close'][0]
    OHLCV_df['Low'] = OHLCV_df['Low']/OHLCV_df['Low'][0]
    OHLCV_df['High'] = OHLCV_df['High']/OHLCV_df['High'][0]
    OHLCV_df['Open'] = OHLCV_df['Open']/OHLCV_df['Open'][0]
    risk_free = load_data("US3M", False) / 100
    risk_free.index = pd.to_datetime(risk_free.index.strftime('%Y-%m'))
    risk_free = risk_free.reindex(OHLCV_df.index, method='ffill')
    sp500 = load_data("^GSPC", False)["Adj Close"][OHLCV_df.index]
    return OHLCV_df, risk_free, sp500
