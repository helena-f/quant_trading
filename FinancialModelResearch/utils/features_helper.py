import numpy as np
import pandas as pd

from HMM.stock_features import handle_missing_values

# calcualte key features for stock price prediction by formulation
def get_key_features(data):
    df = pd.DataFrame()
    
    df['Close'] = data['Close']
    df['High'] = data['High']
    df['Low'] = data['Low']
    df['Open'] = data['Open']
    df['Volume'] = data['Volume']
    
    # Moving Averages
    df['MA50'] = data['Close'].rolling(window=50).mean()
    df['MA200'] = data['Close'].rolling(window=200).mean()

    # RSI
    delta = data['Close'].diff()
    # use ema instead of sma
    # gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    # loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    # df['RSI'] = 100 - (100 / (1 + gain / loss))
    gain = (delta.where(delta > 0, 0)).ewm(span=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    

    # Volatility
    df['Volatility'] = data['Close'].rolling(20).std() / data['Close']
    df['Daily_Return'] = data['Close'].pct_change()
    df['Price_Momentum'] = data['Close'].pct_change(periods=10)

    # Volume
    df['Volume_MA20'] = df['Volume'].ewm(span=20).mean()  
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']

    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ATR
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    # use ema instead of sma
    # ranges = pd.concat([high_low, high_close, low_close], axis=1)
    # df['ATR'] = ranges.max(axis=1).rolling(14).mean()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(span=14, adjust=False).mean()
    
    
    df = handle_missing_values(df)
    
    return df