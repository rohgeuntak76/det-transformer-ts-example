import math
import pathlib
# import yfinance as yf
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import plotext as pltt

import determined as det
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def create_sequences_offline(symbol,start,end, seq_length, is_offline,is_train):
  if is_offline:
    if is_train:
      file_name = f'input/train_{symbol}_{start}_{end}.csv'
    else:
      file_name = f'input/test_{symbol}_{start}_{end}.csv'
    print(f"Read input data from {file_name} file!")
    data = pd.read_csv(file_name,header=[0,1],index_col=0)
  else:
    data = yf.download(symbol, start=start, end=end)

  prices = data['Close'].values.reshape(-1, 1)
  scaler = MinMaxScaler(feature_range=(-1, 1)) # diff
  prices_normalized = scaler.fit_transform(prices)
  prices_tensor = torch.FloatTensor(prices_normalized)
  
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

        self.embedding = nn.Linear(input_dim, seq_length) # use linear layer as an embedding layer.
        transformer_layer = nn.TransformerEncoderLayer(d_model=seq_length, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(seq_length, output_dim)
        self.pos_encoder = PositionalEncoding(seq_length)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)  # Reshape for transformer. Reshape input tensor to fit requirements of transformer encoder, which is this format (sequence length, batch size, features)
        output = self.transformer(src)
        output = output.permute(1, 0, 2)  # Reshape back
        output = self.fc_out(output) # [batch, seq, seq] -> [batch, seq, 1]
        return output[:,-1,:] # [batch, seq, 1] -> [batch, 1] // use last value

def train(model, train_loader, optimizer, criterion, device, epoch,core_context,op):
    model.train()

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

    if core_context.distributed.rank == 0: # only chief process report
      if ( (epoch + 1) % 5) == 0:          # every 5 epochs
          print(f'Epoch {epoch}, Train Loss: {loss.item()}')
          core_context.train.report_training_metrics(
            steps_completed=(epoch),
            metrics={"train_loss": loss.item()}
          )

      op.report_progress(epoch)            # report progress for WebUI progress bar
    

def predict(model, input_data, device):
    input_data = input_data.to(device)
    model.eval()  # Switch the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        prediction = model(input_data)  # Get the model's prediction

    return prediction

def eval_with_dataset(model, scaler,X,y,core_context,epoch,device,is_finished):
    model.eval() # Prepare the model for evaluation

    all_predictions = []
    all_actuals = []
    
    with torch.no_grad():
        for i in range(len(X)):
            single_prediction = predict(model, X[i].unsqueeze(0),device)
            single_prediction = single_prediction.cpu()
            predicted_value = single_prediction.item()  # Extract the last element
            all_predictions.append(predicted_value)
            all_actuals.append(y[i].item())

    # Convert predictions and actuals to the original scale
    all_predictions = scaler.inverse_transform(np.array(all_predictions).reshape(-1, 1))
    all_actuals = scaler.inverse_transform(np.array(all_actuals).reshape(-1, 1))

    # Calculate MSE and MAE
    mse = mean_squared_error(all_actuals, all_predictions).item()
    mae = mean_absolute_error(all_actuals, all_predictions).item()
    mape = mean_absolute_percentage_error(all_actuals, all_predictions).item()
    
    if core_context.distributed.rank == 0:             # only chief process report validation metrics
        print(f'Mean Squared Error: {mse}')
        print(f'Mean Absolute Error: {mae}')
        print(f'Mean Absolute Percentage Error: {mape}')
        val_metrics = {'Mean Squared Error': mse, 'Mean Absolute Error': mae,'Mean Absolute Percentage Error': mape,}
        if not is_finished:
          core_context.train.report_validation_metrics(
            steps_completed=epoch, metrics=val_metrics
          )
        else:                                            # last evaluation with new data reported as seperate metrics
          core_context.train.report_metrics(        
            group="New Time Series",
            steps_completed=epoch,metrics=val_metrics
          ) 

        all_actuals_list = all_actuals.reshape(1,-1).squeeze().tolist()
        all_predictions_list = all_predictions.reshape(1,-1).squeeze().tolist()

        pltt.clear_figure()
        pltt.plot(all_actuals_list, label='Actual Prices', color='blue')
        pltt.plot(all_predictions_list, label='Predicted Prices', color='red')
        pltt.title('Actual vs Predicted Stock Prices (training data)')
        pltt.xlabel('Time (Days)')
        pltt.ylabel('Stock Price')
        pltt.show()
    return mse, mae, mape

def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_state(checkpoint_directory, trial_id):
    checkpoint_directory = pathlib.Path(checkpoint_directory)

    with checkpoint_directory.joinpath("checkpoint.pt").open("rb") as f:
        model = torch.load(f)
    with checkpoint_directory.joinpath("state").open("r") as f:
        start_epoch, ckpt_trial_id = [int(field) for field in f.read().split(",")]

    if ckpt_trial_id != trial_id:
        start_epoch = 0

    return model, start_epoch

def main(core_context): # (note: seq_length needs to be a multiple of num_heads)
    torch.cuda.set_device(core_context.distributed.local_rank)
    dist.init_process_group("nccl")

    info = det.get_cluster_info()
    assert info is not None, "this example only runs on MLDE"

    latest_checkpoint = info.latest_checkpoint
    if latest_checkpoint == None:
        print("No checkpoints")
        start_epoch = 0
    else:
        with core_context.checkpoint.restore_path(latest_checkpoint) as path:
            loaded_state,start_epoch = load_state(path,info.trial.trial_id)
        
    hparams = info.trial.hparams
    
    seq_length = hparams["seq_length"] #8
    batch_size = hparams["batch_size"] #16 #128
    input_dim = 1
    num_layers = hparams["num_layers"] #2
    num_heads = hparams["num_heads"]   #2
    dim_feedforward = hparams["dim_feedforward"] #10
    output_dim = 1
    lr = hparams["lr"]
    ################################# train model
    set_seed(0)
    # train data 
    symbol = 'AAPL' #'^GSPC'
    start_train = '2010-01-01'
    end_train = '2023-12-31'
    X, y, scaler_train = create_sequences_offline(symbol,start_train,end_train, seq_length,is_offline=True,is_train=True)
    
    # test data
    start_test = '2024-01-01'
    end_test = '2024-06-30'
    X_test, y_test, scaler_test = create_sequences_offline(symbol,start_test,end_test, seq_length,is_offline=True,is_train=False)
  
    train_data = TensorDataset(X,y)
    ### DDP code snippet start
    device = torch.device(core_context.distributed.local_rank)

    train_sampler = DistributedSampler(
        train_data,
        num_replicas=core_context.distributed.size,
        rank=core_context.distributed.rank
    )
    train_loader = DataLoader(train_data,sampler=train_sampler,batch_size=batch_size, shuffle=False)
    
# rank0 only test
    model = TransformerModel(input_dim, seq_length, num_layers, num_heads, dim_feedforward, output_dim).to(device)
    model = DDP(model,device_ids=[device])
    ### DDP code snippet end
    if start_epoch != 0:
       print("load state_dict from MLDE checkpoint!")
       model.load_state_dict(loaded_state)
    ### checkpoint snippet 

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # learning rate (i.e. step size of loss function), 0.001 is a common lr

    for op in core_context.searcher.operations():
      for epoch in range(start_epoch,op.length):
        ################################### Training
        train(model, train_loader, optimizer, criterion, device, epoch,core_context,op)
        ################################### evaluation 
        if ( (epoch + 1) % 5) == 0:
          eval_with_dataset(model,scaler_train,X,y,core_context,epoch,device,False) # eval with training data every 5 epochs

          if core_context.distributed.rank == 0:
            checkpoint_metadata_dict = {"steps_completed": epoch, "description": "checkpoint of transformer model which predict 1day after"} # steps_completed is required_item
            with core_context.checkpoint.store_path(checkpoint_metadata_dict) as (path, storage_id):
                torch.save(model.state_dict(), path / "checkpoint.pt")
                with path.joinpath("state").open("w") as f:
                  f.write(f"{epoch+1},{info.trial.trial_id}")

        if core_context.preempt.should_preempt():
          return
    
      ################################### predict final model with new timeline
      if core_context.distributed.rank == 0:
        _, _, mape = eval_with_dataset(model,scaler_test,X_test,y_test,core_context,epoch,device,True) # eval with new data
        op.report_completed(mape) # report metrics to searcher which want to compare performance

if __name__ == '__main__':
  #### DDP code snippet
  # dist.init_process_group("nccl")
  distributed = det.core.DistributedContext.from_torch_distributed()
  with det.core.init(distributed=distributed) as core_context:
    main(core_context)