import binance_functions as bf

# gets historical data from binance
if __name__ == "__main__":
    symbol = 'BTCUSDT'          # crypto pairing
    intervals = ['1h', '1d']    # list of intervals
    from_date = '1970-01-01'    # data from which date
    # to_date = '2020-12-31'    # data until which date
    
    # connect to binance
    client = bf.connect_to_binance()
    for interval in intervals:
        df = bf.get_hist_data_and_transform_to_df(client, symbol, interval, from_date)

        # save df as feather
        actual_from_date = df.index.min().date()
        file_path = f'./data/{symbol.lower()}_from_{actual_from_date}_interval_{interval}.feather'

        df.to_feather(file_path)
        print("data was saved as feather")
