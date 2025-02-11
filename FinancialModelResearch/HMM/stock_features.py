import numpy as np
import pandas as pd
import yfinance as yf
import ta 
# Technical Analysis Library：https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html

def get_stock_data(ticker, start, end, interval):
    data = yf.download(ticker, start=start, end=end, interval=interval)
            
    return data[['Open', 'Close', 'High', 'Low', 'Volume']]

def get_key_features(data):
    df = pd.DataFrame()
    
    df['Close'] = data['Close']
    df['High'] = data['High']
    df['Low'] = data['Low']
    df['Open'] = data['Open']
    df['Volume'] = data['Volume']
    
    # Moving Averages
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()

    # EMA
    df['EMA9'] = ta.trend.EMAIndicator(close=df['Close'], window=9).ema_indicator()
    df['EMA20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()

    # RSI
    rsi_indicator = ta.momentum.RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi_indicator.rsi()

    # MACD
    macd_indicator = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd_indicator.macd()
    df['MACD_Signal'] = macd_indicator.macd_signal()

    # Volatility
    dr_indicator = ta.others.DailyReturnIndicator(df['Close'])
    df['Daily_Return'] = dr_indicator.daily_return() 

    # ATR
    atr_indicator = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
    df['ATR'] = atr_indicator.average_true_range() 

    df = handle_missing_values(df)
    
    return df


def handle_missing_values(df):
    price_cols = ['Open', 'Close', 'High', 'Low', 'Volume']
    indicator_cols = [
        'MA50', 'MA200', 'EMA9', 'EMA20',   
        'RSI', 'MACD', 'MACD_Signal',       
        'Daily_Return', 'ATR'               
    ]
    
    # use forward fill for price data
    df[price_cols] = df[price_cols].ffill()
    
    # fill missing values in indicator columns with 0
    df[indicator_cols] = df[indicator_cols].fillna(0)
    
    # replace inf and -inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    # fill NaN with mean
    df = df.fillna(df.mean())
    
    return df


def get_feature_sets():
    return ([
        ["Close"],
        ["Volume"],

        ["MA50"],
        ["MA200"],

        ['EMA9'],
        ['EMA20'],

        ["RSI"],

        ["MACD"],
        ['MACD_Signal'],    

        ["Daily_Return"],
                    
        ["ATR"],   
                                
        ["MA200", "MA50"], 
        ["MA200", "RSI"],                 
        ["MA200", "ATR"],  
        ["MA200", "MA50", "ATR", "MACD"],
        ["MA200", "MACD", "ATR", "RSI"],
        ["MA200", "MACD", "MACD_Signal", "MA50"],
        ["MA200", "Daily_Return", "ATR", "RSI"]

        
])   


if __name__ == "__main__":
    print(get_feature_sets())
    stock_data = get_stock_data("AAPL", "2024-01-01", "2025-01-01", "1d")
    features = get_key_features(stock_data)
    print(features.tail())