import math
import yfinance as yf
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim # informer
import plotext as pltt

def create_sequences(symbol,start,end, seq_length):
  # symbol = 'AAPL'
  data = yf.download(symbol, start=start, end=end)
  prices = data['Close'].values.reshape(-1, 1)
  scaler = MinMaxScaler(feature_range=(-1, 1)) # diff
  prices_normalized = scaler.fit_transform(prices)
  prices_tensor = torch.FloatTensor(prices_normalized)
  # print(data[:10])
  # print(prices_tensor[:10])

  # train_X, train_y = create_sequences(prices_tensor, seq_length)
  xs,ys = [],[]
  for i in range(len(prices_tensor)-seq_length-1):
    x = prices_tensor[i:(i+seq_length)] # use prices_tensor[0:5] to predict data[5], use data[1:6] to predict data[6]
    y=prices_tensor[i+seq_length]
    xs.append(x)
    ys.append(y)
  return torch.stack(xs), torch.stack(ys), scaler

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module): # inherts from nn.Modeul which is a base class of PyTorch with useful methods and attributes
    def __init__(self, input_dim, seq_length, num_layers, num_heads, dim_feedforward, output_dim):
        '''
        inpt_dim = num of features in input data (e.g. 1 if only closing price)
        seq_length = length of input sequence (e.g. 5 if use 5 days to predict 6th day)
        num_layers = num of layers in Transformer encoder
        num_heads = num of heads in multi-head attention mechanism
        dim_feedforward = dimension of feedforward network in transformer. Each feedforwawrd network of a transformer layer will transform input data into a vector of 2048 dimensions and then pass onto next layer. Implicitly used in nn.TransformerEncoderLayer
        output_dim = num of features in output (e.g. 1 if only predicting the next closing price)
        '''
        super(TransformerModel, self).__init__() # calling the constructor (def __init__) of nn.Module

        self.embedding = nn.Linear(input_dim, seq_length) # use linear layer as an embedding layer. Independet variable X = num of features of X + length of each sequence of X
        '''
        In NLP, embedding layer transforms discrete tokens (like words) into continuous vectors
        In time series data, embedding layer transform features (e.g. dates, categorical data) into continuous vector
        '''

        transformer_layer = nn.TransformerEncoderLayer(d_model=seq_length, nhead=num_heads, dim_feedforward=dim_feedforward)
        '''
        a single transformer layer, that includes
          - a multi-head self-attention mechanism
          - a feedforward neural network

        '''

        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        '''
        encoder to stack multiple transformer layers

        (q) what is encoder?
        - in 1 sentence, it is to capture the short and long term patter of the time series stock price, and this is done by stacking transformer layers
        - it is the part of a model that processes input data
        - primary role is to transform input data into a format (typically by stacking layers) suitable for further processing
        - example
          - input data (e.g.stock prices), represent each data using high-dimension vectors (i.e. multiple numerical features)
          - stacking identifical transformer layers
          - as data pass through the stack layers, the data undergo hierarchical transformation.
          - Lower layers capture local dependencies, while higher layers capture more global dependencies and abstractions.
            - lower/earlier layer: patterns in input data that are close to each other in terms of position in a sequence, e.g. short term impact of stock price
            - higher/later layer: e.g. long term impact of stop price
          - output: extracting the important information from the important data

        (q) what is decoder
        - decoder is not used for time-series data
        '''

        self.fc_out = nn.Linear(seq_length, output_dim)
        '''
        - output layer to transform 'output of transformer encoder (i.e. stacked transformer
        layers)' to 'final output of desired dimensions'
        - output layer gives the final output /  actual prediction of the model
        '''
        self.pos_encoder = PositionalEncoding(seq_length)

    def forward(self, src):
      '''
      a special method in PyTorch to process input (src) and returns output

      forward function is only called implicitly.  This is due to how nn.Module class is designed

      (usage 1: training)
      - y_pred = model(x_batch) internally becomes y_pred = model.forward(x_batch)
      (usage 2: predicting)
      - when we write "predictions = model(input_data)", it internally becomes "predictions = model.forward(input_data)
      '''
      # print(f"input : {src.size()}")
      src = self.embedding(src)
      # print(f"output of first linear : {src.size()}")
      # print(f"output of first linear : {src}")
      # exit()
      src = self.pos_encoder(src)
      # print(f"output of pos encoder : {src}")
      # exit()
      src = src.permute(1, 0, 2)  # Reshape for transformer. Reshape input tensor to fit requirements of transformer encoder, which is this format (sequence length, batch size, features)
      # print(f"ts input : {src.size()}")
      output = self.transformer(src)
      # print(f"ts output : {output.size()}")
      output = output.permute(1, 0, 2)  # Reshape back
      # print(f"output before linear : {output.size()}")
      # self.fc_out(output)
      output = self.fc_out(output) # [batch, seq, seq] -> [batch, seq, 1]
      
      # print(f"return value : {output[:,-1,:].squeeze(-1)}")
      # print(f"output after linear : {output.size()}")
      # print(f"after fc_out : {self.fc_out(output).size()}")
      # exit()
      # return output[:,-1,:].squeeze(-1)
      return output[:,-1,:] # [batch, seq, 1] -> [batch, 1] // use last value

def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    # for epoch in range(epochs):
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(x_batch)
        # print(f"model output : {y_pred.size()}")
        # print(f"label  : {y_batch.size()}")
        # exit()
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    
    if ( (epoch + 1) % 5) == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    elif epoch == epochs - 1:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

def predict(model, input_data, device):
    input_data = input_data.to(device)
    model.eval()  # Switch the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        prediction = model(input_data)  # Get the model's prediction
    # print(f"predict size: {prediction.size()}")
    # exit()
    return prediction

def eval_with_dataset(model, scaler,X,y):
  # X, y = train_X, train_y
    model.eval() # Prepare the model for evaluation

    all_predictions = []
    all_actuals = []
    
    with torch.no_grad():
        for i in range(len(X)):
            single_prediction = predict(model, X[i].unsqueeze(0),device)
            single_prediction = single_prediction.cpu()
            # predicted_value = single_prediction.squeeze().numpy()[-1]  # Extract the last element
            predicted_value = single_prediction.item()  # Extract the last element
            # print(predicted_value)
            # exit()
            all_predictions.append(predicted_value)
            all_actuals.append(y[i].item())

    # Convert predictions and actuals to the original scale
    all_predictions = scaler.inverse_transform(np.array(all_predictions).reshape(-1, 1))
    all_actuals = scaler.inverse_transform(np.array(all_actuals).reshape(-1, 1))

    # Calculate MSE and MAE
    mse = mean_squared_error(all_actuals, all_predictions)
    mae = mean_absolute_error(all_actuals, all_predictions)
    mape = mean_absolute_percentage_error(all_actuals, all_predictions)
    #
    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Absolute Percentage Error: {mape}')

    # plt.figure(figsize=(12, 6))
    all_actuals_list = all_actuals.reshape(1,-1).squeeze().tolist()
    all_predictions_list = all_predictions.reshape(1,-1).squeeze().tolist()

    pltt.clear_figure()
    pltt.plot(all_actuals_list, label='Actual Prices', color='blue')
    pltt.plot(all_predictions_list, label='Predicted Prices', color='red')
    pltt.title('Actual vs Predicted Stock Prices (training data)')
    pltt.xlabel('Time (Days)')
    pltt.ylabel('Stock Price')
    # pltt.legend()
    pltt.show()

def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(seq_length, batch_size, input_dim, num_layers,
         num_heads, dim_feedforward, output_dim, lr,
         epochs,device): # (note: seq_length needs to be a multiple of num_heads)

    ################################# train model
    set_seed(0)
    # train data 
    symbol = 'AAPL'
    start_train = '2010-01-01'
    end_train = '2023-12-31'
    X, y, scaler_train = create_sequences(symbol,start_train,end_train, seq_length)
    print(X[:3])
    print(y[:3])
    
    # test data
    start_test = '2024-01-01'
    end_test = '2024-06-30'
    X_test, y_test, scaler_test = create_sequences(symbol,start_test,end_test, seq_length)
    print(X_test[:3])
    print(y_test[:3])
    # exit()
    train_data = TensorDataset(X,y)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    model = TransformerModel(input_dim, seq_length, num_layers, num_heads, dim_feedforward, output_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # learning rate (i.e. step size of loss function), 0.001 is a common lr

    for epoch in range(epochs):
      train(model, train_loader, optimizer, criterion, device, epoch)

      ################################### predict 
      if ( (epoch + 1) % 5) == 0:
        eval_with_dataset(model,scaler_train,X,y) # eval with training data

    ################################### predict final model 
    eval_with_dataset(model,scaler_test,X_test,y_test) # eval with new data


if __name__ == '__main__':
  # device = torch.device("cuda")
  # HP
  seq_length = 8
  batch_size = 16
  input_dim = 1
  num_layers = 2
  num_heads = 2
  dim_feedforward = 10
  output_dim = 1
  lr = 0.0001
  epochs = 50
  device = torch.device("cuda")

  main(seq_length,batch_size, input_dim, num_layers,num_heads,dim_feedforward,output_dim,lr,epochs,device)

  # def main(seq_length=4, batch_size=16, input_dim = 1, num_layers=2,
  #        num_heads = 2, dim_feedforward=10, output_dim = 1, lr=0.001,
  #        epochs=50,device=torch.device('cpu')):# (note: seq_length needs to be a multiple of num_heads)