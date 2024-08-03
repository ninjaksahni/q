import streamlit as st
import yfinance as yf
import stumpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math

# Fetch historical stock data
def get_data(ticker):
    data = yf.download(ticker, start='2020-01-01', end=datetime.today().strftime('%Y-%m-%d'))
    return data

# Calculate technical indicators including volume-based indicators
def calculate_indicators(data):
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Volume indicators
    data['OBV'] = (np.sign(delta) * data['Volume']).cumsum()
    data['VMA'] = data['Volume'].rolling(window=50).mean()
    
    return data

# Apply STUMPY for pattern detection
def analyze_patterns(data, window_size):
    time_series = data['Close'].values
    matrix_profile = stumpy.stump(time_series, m=window_size)
    
    pattern_distances = matrix_profile[:, 0]  # Distances of the closest match
    threshold = np.percentile(pattern_distances, 10)  # Top 10% of distances
    signals = np.where(pattern_distances < threshold, 1, 0)
    
    # Ensure signals array matches the length of the DataFrame
    padded_signals = np.zeros(len(data))
    if len(signals) > 0:
        padded_signals[window_size:window_size+len(signals)] = signals[:len(padded_signals)-window_size]
    data['Pattern Signal'] = padded_signals
    data['Pattern Position'] = data['Pattern Signal'].diff()
    
    return data, matrix_profile

# Generate trading signals with volume analysis
def generate_signals(data, window_size):
    data, matrix_profile = analyze_patterns(data, window_size)
    data['Signal'] = 0
    
    # Combine indicators with volume-based signals
    data['Signal'] = np.where(
        (data['Close'] < data['SMA50']) & 
        (data['RSI'] < 30) & 
        (data['Pattern Signal'] == 1) &
        (data['Volume'] > data['VMA']), 1, 0
    )
    data['Position'] = data['Signal'].diff()
    return data, matrix_profile

# Backtest the strategy
def backtest_strategy(data, initial_balance=10000):
    balance = initial_balance
    shares = 0
    portfolio_value = []
    positions = []
    for i in range(len(data)):
        if data['Position'].iloc[i] == 1:
            shares = balance / data['Close'].iloc[i]
            balance = 0
            positions.append({'type': 'buy', 'price': data['Close'].iloc[i]})
        elif data['Position'].iloc[i] == -1:
            balance = shares * data['Close'].iloc[i]
            shares = 0
            positions.append({'type': 'sell', 'price': data['Close'].iloc[i]})
        portfolio_value.append(balance + shares * data['Close'].iloc[i])
    
    data['Portfolio Value'] = portfolio_value
    return data, positions

# Calculate performance metrics
def calculate_performance(data, positions):
    buy_prices = [p['price'] for p in positions if p['type'] == 'buy']
    sell_prices = [p['price'] for p in positions if p['type'] == 'sell']
    
    if len(buy_prices) == 0 or len(sell_prices) == 0:
        return 0, 0, 0, 0
    
    profits = [(sell_prices[i] - buy_prices[i]) for i in range(min(len(buy_prices), len(sell_prices)))]
    
    win_rate = len([p for p in profits if p > 0]) / len(profits) if profits else 0
    profit_factor = sum([p for p in profits if p > 0]) / abs(sum([p for p in profits if p < 0])) if sum([p for p in profits if p < 0]) != 0 else float('inf')
    returns = data['Portfolio Value'].pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * math.sqrt(252) if returns.std() != 0 else 0
    max_drawdown = ((data['Portfolio Value'].cummax() - data['Portfolio Value']) / data['Portfolio Value'].cummax()).max()
    
    return win_rate, profit_factor, sharpe_ratio, max_drawdown

# Generate actionable buy, hold, or sell signal with color
def generate_action_signal(data):
    if data['Signal'].iloc[-1] == 1:
        return "Buy", "green"
    elif data['Signal'].iloc[-1] == -1:
        return "Sell", "red"
    else:
        return "Hold", "cyan"

# Streamlit application
st.title('MRS+PATTERN MATRIX+VA - KSv0.2' )

# Input for ticker symbol and window size for STUMPY
ticker = st.text_input('Enter Stock Ticker', 'AAPL')
window_size = st.slider('Select Window Size for STUMPY', min_value=2, max_value=50, value=10)

# Fetch and display data
data = get_data(ticker)
st.write(f"Showing data for {ticker}")
st.dataframe(data.tail())

# Calculate indicators and generate signals
data = calculate_indicators(data)
data, matrix_profile = generate_signals(data, window_size)
data, positions = backtest_strategy(data)
win_rate, profit_factor, sharpe_ratio, max_drawdown = calculate_performance(data, positions)

# Display performance metrics
results = {
    'Win Rate': f"{win_rate * 100:.2f}%",
    'Profit Factor': f"{profit_factor:.2f}",
    'Sharpe Ratio': f"{sharpe_ratio:.2f}",
    'Max Drawdown': f"{max_drawdown * 100:.2f}%"
}
st.write("Performance Metrics:")
st.write(results)

# Display pattern detection results
if len(data) > window_size:
    st.write("Pattern Signals:")
    st.dataframe(data[['Close', 'Pattern Signal']])
else:
    st.write("Not enough data to compute matrix profile. Please select a smaller window size.")

# Display actionable signal with color
action_signal, color = generate_action_signal(data)
st.markdown(f"<h2 style='color: {color}; text-align: center;'>Actionable Signal: **{action_signal}**</h2>", unsafe_allow_html=True)

# Plotting charts
fig, axs = plt.subplots(5, 1, figsize=(14, 22), sharex=True)

# Plot Stock Price and Moving Averages
axs[0].plot(data.index, data['Close'], label='Close Price', color='blue')
axs[0].plot(data.index, data['SMA50'], label='SMA 50', color='orange')
axs[0].plot(data.index, data['SMA200'], label='SMA 200', color='red')
axs[0].set_title('Stock Price and Moving Averages')
axs[0].legend()

# Plot RSI
axs[1].plot(data.index, data['RSI'], label='RSI', color='purple')
axs[1].axhline(30, color='red', linestyle='--', label='Oversold')
axs[1].axhline(70, color='green', linestyle='--', label='Overbought')
axs[1].set_title('Relative Strength Index (RSI)')
axs[1].legend()

# Plot Volume and OBV
axs[2].bar(data.index, data['Volume'], label='Volume', color='gray', alpha=0.5)
axs[2].plot(data.index, data['OBV'], label='On-Balance Volume (OBV)', color='black')
axs[2].set_title('Volume and On-Balance Volume (OBV)')
axs[2].legend()

# Plot Trading Signals
axs[3].plot(data.index, data['Close'], label='Close Price', color='blue')
buy_signals = data[data['Position'] == 1]
sell_signals = data[data['Position'] == -1]
axs[3].scatter(buy_signals.index, buy_signals['Close'], label='Buy Signal', marker='^', color='green', s=100)
axs[3].scatter(sell_signals.index, sell_signals['Close'], label='Sell Signal', marker='v', color='red', s=100)
axs[3].set_title('Trading Signals')
axs[3].legend()

# Plot Matrix Profile
axs[4].plot(data.index[window_size-1:], matrix_profile[:, 0], label='Matrix Profile', color='cyan')
axs[4].set_title('Matrix Profile')
axs[4].legend()

st.pyplot(fig)
