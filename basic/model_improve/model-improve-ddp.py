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

import determined as det
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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
        # self.register_buffer('pe', pe)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

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
      src = self.embedding(src)
      src = self.pos_encoder(src)
      src = src.permute(1, 0, 2)  # Reshape for transformer. Reshape input tensor to fit requirements of transformer encoder, which is this format (sequence length, batch size, features)
      output = self.transformer(src)
      output = output.permute(1, 0, 2)  # Reshape back
      output = self.fc_out(output) # [batch, seq, seq] -> [batch, seq, 1]
      return output[:,-1,:] # [batch, seq, 1] -> [batch, 1] // use last value

def train(model, train_loader, optimizer, criterion, device, epoch,core_context):
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
    if core_context.distributed.rank == 0:        
      if ( (epoch + 1) % 5) == 0:
          print(f'Epoch {epoch+1}, Loss: {loss.item()}')
          core_context.train.report_training_metrics(
            steps_completed=(epoch+1),
            metrics={"train_loss": loss.item()}
          )        
    

def predict(model, input_data, device):
    input_data = input_data.to(device)
    model.eval()  # Switch the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        prediction = model(input_data)  # Get the model's prediction
    # print(f"predict size: {prediction.size()}")
    # exit()
    return prediction

def eval_with_dataset(model, scaler,X,y,core_context,epoch,device):
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
    core_context.train.report_validation_metrics(
       steps_completed=epoch,
       metrics={'Mean Squared Error': mse,
                'Mean Absolute Error': mae,
                'Mean Absolute Percentage Error': mape,
                }
    )
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
         epochs,core_context): # (note: seq_length needs to be a multiple of num_heads)

    ################################# train model
    set_seed(0)
    # train data 
    symbol = 'AAPL'
    start_train = '2010-01-01'
    end_train = '2023-12-31'
    X, y, scaler_train = create_sequences(symbol,start_train,end_train, seq_length)
    print(X[:3])
    print(y[:3])
    # print(len(y)) <- 3513
    # exit()
    
    # test data
    start_test = '2024-01-01'
    end_test = '2024-06-30'
    X_test, y_test, scaler_test = create_sequences(symbol,start_test,end_test, seq_length)
    print(X_test[:3])
    print(y_test[:3])
    # exit()
    train_data = TensorDataset(X,y)

    ### DDP code snippet start
    device = torch.device(core_context.distributed.local_rank)

    train_sampler = DistributedSampler(
       train_data,
       num_replicas=core_context.distributed.size,
       rank=core_context.distributed.rank
    )
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_data,sampler=train_sampler,batch_size=batch_size, shuffle=False)

    model = TransformerModel(input_dim, seq_length, num_layers, num_heads, dim_feedforward, output_dim).to(device)
    model = DDP(model,device_ids=[device])
    ### DDP code snippet end

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # learning rate (i.e. step size of loss function), 0.001 is a common lr

    for epoch in range(epochs):
      train(model, train_loader, optimizer, criterion, device, epoch,core_context)

      ################################### predict 
      if core_context.distributed.rank == 0:
        if ( (epoch + 1) % 5) == 0:
          eval_with_dataset(model,scaler_train,X,y,core_context,epoch,device) # eval with training data

    ################################### predict final model 
    if core_context.distributed.rank == 0:
      eval_with_dataset(model,scaler_test,X_test,y_test,core_context,epoch+1,device) # eval with new data


if __name__ == '__main__':
  # HP
  seq_length = 8
  batch_size = 32 # 16 #128
  input_dim = 1
  num_layers = 2
  num_heads = 2
  dim_feedforward = 10
  output_dim = 1
  lr = 0.0001
  epochs = 50
  # device = torch.device("cuda")
  # with det.core.init() as core_context:
  # main(seq_length,batch_size, input_dim, num_layers,num_heads,dim_feedforward,output_dim,lr,epochs,device,core_context)

  #### DDP code snippet
  dist.init_process_group("nccl")
  distributed = det.core.DistributedContext.from_torch_distributed()
  with det.core.init(distributed=distributed) as core_context:
    main(seq_length,batch_size, input_dim, num_layers,num_heads,dim_feedforward,output_dim,lr,epochs,core_context)

  # def main(seq_length=4, batch_size=16, input_dim = 1, num_layers=2,
  #        num_heads = 2, dim_feedforward=10, output_dim = 1, lr=0.001,
  #        epochs=50,device=torch.device('cpu')):# (note: seq_length needs to be a multiple of num_heads)