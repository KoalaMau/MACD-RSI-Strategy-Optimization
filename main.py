import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import mplfinance as mpf
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


# Backtest strategy with TWA calculation
def backtest_strategy(data, rsi_over_sold, investment_amount=1000):
    data['was_over_sold'] = data['RSI'] <= rsi_over_sold
    data['was_over_sold_in_last_10'] = data['was_over_sold'].rolling(window=10).sum() > 0
    data['crossover_bull'] = (data['MACD'] > data['Signal_Line']) & (data['MACD'].shift(1) <= data['Signal_Line'].shift(1))
    data['buy_signal'] = data['was_over_sold_in_last_10'] & data['crossover_bull']

    # Calculate investment and cumulative investment
    data['investment'] = np.where(data['buy_signal'], investment_amount, 0)
    data['investment_cumsum'] = data['investment'].cumsum()

    # Calculate portfolio value using cumulative returns
    data['daily_return'] = data['Close'].pct_change().fillna(0)
    data['cumulative_return'] = (1 + data['daily_return']).cumprod()

    # Track portfolio value (investment multiplied by returns)
    data['portfolio_value'] = data['investment_cumsum'] * data['cumulative_return']

    # Calculate TWA return (only for periods where cumulative investment > 0)
    data['TWA_return'] = np.where(data['investment_cumsum'] > 0,
                                  data['portfolio_value'] / data['investment_cumsum'], 0)
    final_twa_return = data['TWA_return'].iloc[-1] if data['investment_cumsum'].iloc[-1] > 0 else 0

    return data, final_twa_return


# Optimization function based on TWA return
def optimize_strategy(data, macd_params, rsi_params, investment_amount=1000):
    best_params = None
    best_twa_return = -np.inf
    results = []

    for short_window, long_window, signal_window, rsi_period, rsi_over_sold in product(
            macd_params['short_window'], macd_params['long_window'],
            macd_params['signal_window'], rsi_params['rsi_period'], rsi_params['rsi_over_sold']):

        temp_data = data.copy()
        calculate_macd(temp_data, short_window, long_window, signal_window)
        calculate_rsi(temp_data, rsi_period)
        _, twa_return = backtest_strategy(temp_data, rsi_over_sold, investment_amount)

        results.append((short_window, long_window, signal_window, rsi_period, rsi_over_sold, twa_return))

        if twa_return > best_twa_return:
            best_twa_return = twa_return
            best_params = (short_window, long_window, signal_window, rsi_period, rsi_over_sold)

    results_df = pd.DataFrame(results,
                              columns=['Short Window', 'Long Window', 'Signal Window', 'RSI Period', 'RSI Over Sold',
                                       'TWA Return'])
    return best_params, best_twa_return, results_df


# Visualization
def plot_results(data, ticker, rsi_over_sold):
    # Candlestick chart with buy signals
    buy_signals = data[data['buy_signal']].index
    apds = [mpf.make_addplot(data['buy_signal'] * data['Close'], type='scatter', markersize=100, marker='^', color='green', panel=0, ylabel='Price')]

    mpf.plot(data, type='candle', addplot=apds, figsize=(14, 9), title=f'{ticker} Candlestick Chart with Buy Signals',
             ylabel='Price', style='yahoo', volume=True, mav=(50, 100, 200))

    # Line chart with MACD and RSI
    plt.figure(figsize=(14, 9))

    # Price subplot
    plt.subplot(3, 1, 1)
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.scatter(buy_signals, data.loc[buy_signals, 'Close'], color='green', marker='^', label='Buy Signal', zorder=5)
    plt.title(f'{ticker} Price and MACD')
    plt.ylabel('Price')
    plt.legend(loc='upper left')

    # MACD subplot
    plt.subplot(3, 1, 2)
    plt.plot(data['MACD'], label='MACD', color='red', linewidth=1.5)
    plt.plot(data['Signal_Line'], label='Signal Line', color='green', linewidth=1.5)
    plt.bar(data.index, data['Histogram'], label='Histogram',
            color=np.where(data['Histogram'] >= 0, 'green', 'red'), width=1, alpha=0.5)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.ylabel('MACD')
    plt.legend(loc='upper left')

    # RSI subplot
    plt.subplot(3, 1, 3)
    plt.plot(data['RSI'], label='RSI', color='purple')
    plt.axhline(80, color='red', linewidth=1, linestyle='--', label='Overbought')
    plt.axhline(rsi_over_sold, color='green', linewidth=1, linestyle='--', label='Oversold')
    plt.title('RSI')
    plt.ylabel('RSI')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Fetch data
    ticker = 'AAPL'  # Replace with your desired ticker symbol
    data = yf.download(ticker, start='2022-01-01', end='2023-01-01')  # Adjust dates as needed
    data = data[['Close']]  # Ensure only Close prices are used

    # Parameter ranges
    macd_params = {
        'short_window': range(10, 13),
        'long_window': range(25, 26),
        'signal_window': range(9, 13)
    }
    rsi_params = {
        'rsi_period': range(14, 15),
        'rsi_over_sold': range(35, 45, 5)
    }

    # Optimize strategy
    best_params, best_twa_return, results_df = optimize_strategy(data, macd_params, rsi_params)
    print("Best Parameters:", best_params)
    print("Best TWA Return:", best_twa_return)

    # Save results to a CSV
    results_df.to_csv("optimization_results_twa.csv", index=False)

    # Apply strategy with the best parameters
    calculate_macd(data, short_window=best_params[0], long_window=best_params[1], signal_window=best_params[2])
    calculate_rsi(data, rsi_period=best_params[3])
    data, final_twa_return = backtest_strategy(data, rsi_over_sold=best_params[4])

    print(f"Final TWA Return: {final_twa_return}")

    # Plot results
    plot_results(data, ticker, best_params[4])





# # Example usage
# if __name__ == "__main__":
#     # Fetch data from Yahoo Finance
#     ticker = 'AAPL'  # Replace with your desired ticker symbol
#     data = yf.download(ticker, start='2022-01-01', end='2023-01-01', multi_level_index=False)  # Adjust dates as needed
#
#     # Ensure data is clean
#     data = data[['Close']]
#
#     # Parameter ranges for optimization
#     macd_params = {
#         'short_window': range(8, 13),  # Short EMA range
#         'long_window': range(21, 26),  # Long EMA range
#         'signal_window': range(8, 13)  # Signal line range
#     }
#     rsi_params = {
#         'rsi_period': range(10, 15),  # RSI period range
#         'rsi_over_sold': range(20, 40, 5)  # RSI oversold range
#     }
#
#     # Optimize strategy
#     best_params, best_return, results_df = optimize_strategy(data, macd_params, rsi_params)
#     print("Best Parameters:", best_params)
#     print("Best Return:", best_return)
#
#     # Save results to a CSV
#     results_df.to_csv("optimization_results.csv", index=False)
#
#     # Calculate BEST MACD and RSI
#     calculate_macd(data, short_window=12, long_window=26, signal_window=9)
#     calculate_rsi(data, rsi_period=14)
#
#     # Apply strategy
#     rsi_over_sold = 30
#     data = backtest_strategy(data, rsi_over_sold)
#
#     # Plot the results
#     plot_results(data, ticker, rsi_over_sold)
