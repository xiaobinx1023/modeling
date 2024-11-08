import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import talib as ta
import pdb
import vectorbt as vbt
def kalman_filter(zs, Hs, Q, Rs, x0, P0):
    """
    zs: List of observations at each time step [[z1_k, z2_k, ...], ...]
    Hs: Observation model matrix (list of H_i)
    Q: Process noise covariance
    Rs: List of measurement noise covariances [R1, R2, ...]
    x0: Initial state estimate
    P0: Initial estimate covariance
    """
    n = len(zs)
    x_estimates = []
    x = x0
    P = P0
    
    for k in range(n):
        # Prediction Step
        x_pred = x  # State transition is identity
        P_pred = P + Q
        
        # Update Step
        z_k = zs[k]
        H_k = Hs[k]
        R_k = Rs[k]
        
        # Stack observations and H matrices
        z_stack = np.array(z_k).reshape(-1, 1)
        H_stack = np.array(H_k).reshape(-1, 1)
        R_stack = np.diag(R_k)
        
        # Compute Kalman Gain
        S = H_stack @ P_pred @ H_stack.T + R_stack
        K = P_pred @ H_stack.T @ np.linalg.inv(S)
        # pdb.set_trace()
        # pdb.breakpoint()
        # Update state estimate
        y = z_stack - H_stack @ x_pred
        x = x_pred + K @ y
        
        # Update estimate covariance
        P = (np.eye(len(P)) - K @ H_stack) @ P_pred
        
        x_estimates.append(x.item())
        
    return x_estimates


# Fetch historical data
ticker = 'AAPL'
data = yf.download(ticker, start='2022-01-01', end='2024-11-1')

# Calculate indicators
data.columns = data.columns.droplevel("Ticker")

data['SMA10'] = data['Close'].rolling(window=10).mean()
data['SMA50'] = data['Close'].rolling(window=50).mean()
data['RSI'] = ta.RSI(data['Close'], timeperiod=14)
data['ADX'] = ta.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
data['Volume_MA20'] = data['Volume'].rolling(window=20).mean()

## Map indicator to observations
zs = []
Hs = []
Rs = []
for index, row in data.iterrows():
    z_k = []
    H_k = []
    R_k = []
    
    # Moving Average Crossover
    if pd.notnull(row['SMA10']) and pd.notnull(row['SMA50']):
        if row['SMA10'] > row['SMA50']:
            z_mac = 1
        elif row['SMA10'] < row['SMA50']:
            z_mac = -1
        else:
            z_mac = 0
        z_k.append(z_mac)
        H_k.append(1)
        R_k.append(0.1)
    
    # RSI
    if pd.notnull(row['RSI']):
        if row['RSI'] > 70:
            z_rsi = -1
        elif row['RSI'] < 30:
            z_rsi = 1
        else:
            z_rsi = 0
        z_k.append(z_rsi)
        H_k.append(1)
        R_k.append(0.2)
    
    # ADX
    if pd.notnull(row['ADX']):
        if row['ADX'] > 25:
            z_adx = 1
        else:
            z_adx = -1
        z_k.append(z_adx)
        H_k.append(1)
        R_k.append(0.3)
    
    # Volume Analysis
    if pd.notnull(row['Volume']) and pd.notnull(row['Volume_MA20']):
        if row['Volume'] > row['Volume_MA20']:
            z_vol = 1
        else:
            z_vol = -1
        z_k.append(z_vol)
        H_k.append(1)
        R_k.append(0.5)
    
    if z_k:
        zs.append(z_k)
        Hs.append(H_k)
        Rs.append(R_k)
    else:
        # Handle missing data
        zs.append([0])
        Hs.append([1])
        Rs.append([1])

# Initial state and covariance
x0 = np.array([0])
P0 = np.array([[1]])
Q = np.array([[0.01]])

# Run the filter
x_estimates = kalman_filter(zs, Hs, Q, Rs, x0, P0)

# Add estimates to the dataframe
data = data.iloc[len(data) - len(x_estimates):]  # Align indices
data['State_Estimate'] = x_estimates


plt.figure(figsize=(14, 7))
plt.plot(data.index, data['State_Estimate'], label='Kalman Filter State Estimate')
plt.axhline(0, color='black', linestyle='--')
plt.title('Kalman Filter State Estimate Over Time')
plt.xlabel('Date')
plt.ylabel('State Estimate')
plt.legend()
plt.show()


data['Signal'] = 0  # 1 for buy, -1 for sell

for i in range(1, len(data)):
    if data['State_Estimate'].iloc[i-1] < 0.5 and data['State_Estimate'].iloc[i] >= 0.5:
        data['Signal'].iloc[i] = 1  # Buy
    elif data['State_Estimate'].iloc[i-1] > -0.5 and data['State_Estimate'].iloc[i] <= -0.5:
        data['Signal'].iloc[i] = -1  # Sell

# Remove warnings
pd.options.mode.chained_assignment = None  # default='warn'


# Initial capital and positions
capital = 100000
positions = 0
cash = capital
portfolio_value = []

for i in range(len(data)):
    if data['Signal'].iloc[i] == 1 and positions == 0:
        # Buy
        positions = cash / data['Close'].iloc[i]
        cash = 0
    elif data['Signal'].iloc[i] == -1 and positions > 0:
        # Sell
        cash = positions * data['Close'].iloc[i]
        positions = 0
    # Portfolio value
    if positions > 0:
        total_value = positions * data['Close'].iloc[i]
    else:
        total_value = cash
    portfolio_value.append(total_value)

data['Portfolio_Value'] = portfolio_value

# Plot portfolio value over time
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Portfolio_Value'], label='Portfolio Value')
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Value ($)')
plt.legend()
plt.show()

# Create boolean masks for entries and exits
entries = data['Signal'] == 1
exits = data['Signal'] == -1

# Ensure entries and exits are boolean Series aligned with data.Close
entries = entries.astype(bool)
exits = exits.astype(bool)

# Create the portfolio
pf = vbt.Portfolio.from_signals(data['Close'], entries=entries, exits=exits)

# Print portfolio statistics
print(pf.stats())

# Plot the portfolio performance
pf.plot().show()



# buys = data.index[data['Signal'] == 1]
# sells = data.index[data['Signal'] == -1]
# pf = vbt.Portfolio.from_signals(data.Close, buys, sells)
# print(pf.stats())

# pf.plot().show()