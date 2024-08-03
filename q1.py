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
    
    return data

# Generate trading signals with volume analysis
def generate_signals(data, window_size):
    data = analyze_patterns(data, window_size)
    data['Signal'] = 0
    
    # Combine indicators with volume-based signals
    data['Signal'] = np.where(
        (data['Close'] < data['SMA50']) & 
        (data['RSI'] < 30) & 
        (data['Pattern Signal'] == 1) &
        (data['Volume'] > data['VMA']), 1, 0
    )
    data['Position'] = data['Signal'].diff()
    return data

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

# Generate actionable buy, hold, or sell signal with reasons
def generate_action_signal(data):
    last_signal = data['Signal'].iloc[-1]
    
    if last_signal == 1:
        return "Buy", (
            "Buy Signal: The stock price is below the 50-day SMA, the RSI is below 30 indicating oversold conditions, "
            "the pattern signal is strong, and the trading volume is higher than the 50-day volume average."
        )
    elif last_signal == -1:
        return "Sell", (
            "Sell Signal: The stock price is above the 50-day SMA, the RSI is above 70 indicating overbought conditions, "
            "or the pattern signal is weak."
        )
    else:
        return "Hold", (
            "Hold Signal: No strong buy or sell conditions are present based on the current analysis."
        )

# Streamlit application
st.title('MRS + Pattern + VA - KSv0.1')

# Input for ticker symbol and window size for STUMPY
ticker = st.text_input('Enter Stock Ticker', 'AAPL')
window_size = st.slider('Select Window Size for STUMPY', min_value=2, max_value=50, value=10)

# Fetch and display data
data = get_data(ticker)
st.write(f"Showing data for {ticker}")
st.dataframe(data.tail())

# Calculate indicators and generate signals
data = calculate_indicators(data)
data = generate_signals(data, window_size)
data, positions = backtest_strategy(data)
win_rate, profit_factor, sharpe_ratio, max_drawdown = calculate_performance(data, positions)

# Display performance metrics with explanations
st.write("Performance Metrics:")
st.write({
    'Win Rate': f"{win_rate * 100:.2f}%",
    'Profit Factor': f"{profit_factor:.2f}",
    'Sharpe Ratio': f"{sharpe_ratio:.2f}",
    'Max Drawdown': f"{max_drawdown * 100:.2f}%"
})

st.write("""
**Win Rate**: The percentage of trades that resulted in a profit out of all trades executed. It indicates how often the strategy is successful.

**Profit Factor**: The ratio of the total profit to the total loss. A value greater than 1 suggests that the strategy is profitable overall.

**Sharpe Ratio**: A measure of risk-adjusted return. It divides the average return by the standard deviation of returns, with a higher value indicating better risk-adjusted performance.

**Max Drawdown**: The maximum observed loss from a peak to a trough before a new peak is reached. It represents the worst-case loss scenario of the strategy.
""")

# Display pattern detection results
if len(data) > window_size:
    st.write("Pattern Signals:")
    st.dataframe(data[['Close', 'Pattern Signal']])
else:
    st.write("Not enough data to compute matrix profile. Please select a smaller window size.")

# Display actionable signal with reasons
action_signal, action_reason = generate_action_signal(data)
st.markdown(f"### Actionable Signal: **{action_signal}**")
st.write(action_reason)
