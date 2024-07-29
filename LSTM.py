import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Step 1: Fetch the data
ticker = 'AAPL'  # Stock ticker symbol
period = '5y'    # Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '5y', '10y', 'ytd', 'max')
interval = '1d'  # Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1d', '5d', '1wk', '1mo', '3mo')

x_data = yf.download(ticker, period=period,interval=interval)
x_data = x_data[['Close']]
y_data = x_data[['Close']]

# Step 2: Data Preprocessing
scaler = MinMaxScaler(feature_range=(-1, 1))
x_data_scaled = scaler.fit_transform(x_data)
y_data_scaled = scaler.fit_transform(y_data)
print(x_data_scaled.shape)
# Convert to sequences
def create_sequences(x_data,y_data, seq_length):
    xs, ys = [], []
    
    for i in range(len(x_data) - seq_length):
        x = x_data[i:i+seq_length]
        y = y_data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 2
X, y = create_sequences(x_data_scaled,y_data_scaled, seq_length)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Split data into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 3: Define the Model
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, output_size)
        # self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
        #                     torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out,_ = self.lstm(input_seq)

        out = self.linear(lstm_out[:, -1])
        predictions = self.linear2(out)
        return predictions

model = LSTM()
loss_function = nn.MSELoss()
learnig_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learnig_rate)

# Step 4: Train the Model
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                         torch.zeros(1, 1, model.hidden_layer_size))
    
    y_pred = model(X_train)
    single_loss = loss_function(y_pred, y_train)
    single_loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch} Loss: {single_loss.item()}')

# Step 5: Making Predictions
model.eval()
with torch.no_grad():
    test_predictions = model(X_test).numpy()

# Inverse transform the predictions and actual values
#test_predictions = scaler.inverse_transform(test_predictions)
#actual = scaler.inverse_transform(y_test.numpy())
actual = y_test.numpy()
# Plot the results
plt.figure(figsize=(10,6))
plt.plot(actual, label='Actual Prices')
plt.plot(test_predictions, label='Predicted Prices')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
