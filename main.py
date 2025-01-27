import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import yfinance as yf

# Function to calculate MACD
def calculate_macd(data, short_window, long_window, signal_window):
    data['EMA_short'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['EMA_long'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['EMA_short'] - data['EMA_long']
    data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()


# Function to calculate RSI
def calculate_rsi(data, rsi_period):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))


# Backtest strategy
def backtest_strategy(data, macd_threshold, rsi_threshold):
    buy_signals = (data['MACD'] > data['Signal_Line']) & (data['RSI'] < rsi_threshold)
    data['Position'] = np.where(buy_signals, 1, 0)
    data['Daily_Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Position'].shift(1) * data['Daily_Return']
    return data['Strategy_Return'].sum()


# Optimization function
def optimize_strategy(data, macd_params, rsi_params):
    best_params = None
    best_return = -np.inf
    results = []

    for short_window, long_window, signal_window, rsi_period, rsi_threshold in product(
            macd_params['short_window'], macd_params['long_window'],
            macd_params['signal_window'], rsi_params['rsi_period'], rsi_params['rsi_threshold']):

        temp_data = data.copy()
        calculate_macd(temp_data, short_window, long_window, signal_window)
        calculate_rsi(temp_data, rsi_period)
        strategy_return = backtest_strategy(temp_data, macd_threshold=0, rsi_threshold=rsi_threshold)

        results.append((short_window, long_window, signal_window, rsi_period, rsi_threshold, strategy_return))

        if strategy_return > best_return:
            best_return = strategy_return
            best_params = (short_window, long_window, signal_window, rsi_period, rsi_threshold)

    results_df = pd.DataFrame(results,
                              columns=['Short Window', 'Long Window', 'Signal Window', 'RSI Period', 'RSI Threshold',
                                       'Return'])
    return best_params, best_return, results_df


# Example usage
if __name__ == "__main__":
    # Fetch data from Yahoo Finance
    ticker = 'AAPL'  # Replace with your desired ticker symbol
    data = yf.download(ticker, start='2022-01-01', end='2023-01-01')  # Adjust dates as needed

    # Ensure data is clean
    data = data[['Close']]
    # data.dropna(inplace=True)

    # Parameter ranges for optimization
    macd_params = {
        'short_window': range(8, 13),  # Short EMA range
        'long_window': range(21, 26),  # Long EMA range
        'signal_window': range(8, 13)  # Signal line range
    }
    rsi_params = {
        'rsi_period': range(10, 15),  # RSI period range
        'rsi_threshold': range(20, 40, 5)  # RSI threshold range
    }

    # Optimize strategy
    best_params, best_return, results_df = optimize_strategy(data, macd_params, rsi_params)
    print("Best Parameters:", best_params)
    print("Best Return:", best_return)

    # Save results to a CSV
    results_df.to_csv("optimization_results.csv", index=False)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label=f'{ticker} Close Price')
    plt.title(f"{ticker} Close Price")
    plt.legend()
    plt.show()
