import yfinance as yf


# train data 
# symbol = 'AAPL'
symbol = '^GSPC'
start_train = '2010-01-01'
end_train = '2023-12-31'
# X, y, scaler_train = create_sequences(symbol,start_train,end_train, seq_length)

data = yf.download(symbol, start=start_train, end=end_train)
data.to_csv(f"../input/train_{symbol}_{start_train}_{end_train}.csv",encoding='utf-8')

# test data
start_test = '2024-01-01'
end_test = '2024-06-30'
data_test = yf.download(symbol,start=start_test,end=end_test)
data_test.to_csv(f"../input/test_{symbol}_{start_test}_{end_test}.csv",encoding='utf-8')
# X_test, y_test, scaler_test = create_sequences(symbol,start_test,end_test, seq_length)