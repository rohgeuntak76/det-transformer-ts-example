import pandas as pd

file = pd.read_csv("../input/train_AAPL_2010-01-01_2023-12-31.csv",header=[0,1],index_col=0)

# print(file.columns)
# print(file[:10])
print(file['Close'])
