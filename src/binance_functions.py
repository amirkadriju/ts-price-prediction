import os
import pandas as pd
from dotenv import load_dotenv
from binance.client import Client

# connects to binance using the API_KEY and API_SECRET
def connect_to_binance():
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    API_SECRET = os.getenv("API_SECRET")
    client = Client(API_KEY, API_SECRET)

    return client

# gets historical data and transforms data to only keep time, open, high, low, close and volume
def get_hist_data_and_transform_to_df(client, symbol, interval, from_date, to_date=None):
    if to_date == None:
        klines = client.get_historical_klines(symbol, interval, from_date)
    else:
        klines = client.get_historical_klines(symbol, interval, from_date, to_date)

    df = pd.DataFrame(klines)
    df = df.iloc[:,:6]
    df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    df.set_index('Time', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')
    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)

    return df

