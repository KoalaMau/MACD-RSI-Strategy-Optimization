import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import mplfinance as mpf
import yfinance as yf
import numpy as np

# -------------------------------------------
# 1. Indicator Calculations
# -------------------------------------------

def calculate_macd(data, short_window, long_window, signal_window):
    """
    Calculate MACD indicators and histogram.
    """
    data['EMA_short'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['EMA_long'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['EMA_short'] - data['EMA_long']
    data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    data['Histogram'] = data['MACD'] - data['Signal_Line']


def calculate_rsi(data, rsi_period):
    """
    Calculate RSI.
    """
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()

    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))


# -------------------------------------------
# 2. Backtesting Functions
# -------------------------------------------

def backtest_strategy(data, rsi_over_sold, investment_amount=1000):
    data = data.copy()

    data['was_over_sold'] = data['RSI'] <= rsi_over_sold
    data['oversold_recently'] = data['was_over_sold'].rolling(window=10, min_periods=1).max().astype(bool)
    data['crossover_bull'] = (data['MACD'] > data['Signal_Line']) & (
                               data['MACD'].shift(1) <= data['Signal_Line'].shift(1))
    data['buy_signal'] = data['oversold_recently'] & data['crossover_bull']

    positions = []
    data['investment'] = 0.0
    data['portfolio_value'] = 0.0

    buy_signals = data[data['buy_signal']]

    for date, row in buy_signals.iterrows():
        shares = investment_amount / row['Close']
        positions.append({'date': date, 'shares': shares, 'buy_price': row['Close']})
        data.loc[date, 'investment'] = investment_amount

    total_invested = investment_amount * len(positions)
    final_price = data['Close'].iloc[-1]
    portfolio_value = sum(pos['shares'] * final_price for pos in positions)
    strategy_return = portfolio_value / total_invested if total_invested > 0 else 0

    # Correct and efficient portfolio value calculation
    for pos in positions:
        # Use .loc and a *copy* of the slice to avoid in-place issues
        data.loc[pos['date']:, 'portfolio_value'] = data.loc[pos['date']:, 'portfolio_value'] + (pos['shares'] * data.loc[pos['date']:, 'Close']).copy()

    return data, strategy_return


def backtest_monthly(data, signal_buy_data, investment_amount=1000):
    data = data.copy()
    monthly_dates = data.index.to_series().groupby(data.index.to_period('M')).last()
    positions = []
    data['monthly_investment'] = 0.0
    data['monthly_portfolio'] = 0.0

    total_invested = 0.0

    for date in monthly_dates:
        if date in data.index:
            closest_signal_date = signal_buy_data[signal_buy_data.index <= date].index.max()
            if pd.isna(closest_signal_date):
                continue

            signal_investment = signal_buy_data.loc[closest_signal_date, 'investment']

            # The FIX: Check if it's a Series and get the single value
            if isinstance(signal_investment, pd.Series):
                if signal_investment.empty: # Check if the series is empty
                    continue
                signal_investment = signal_investment.iloc[0]  # Extract the single value

            if signal_investment == 0:  # Now this works correctly
                continue

            price = data.loc[date, 'Close']
            shares = signal_investment / price
            positions.append({'date': date, 'shares': shares, 'buy_price': price, 'investment': signal_investment})
            data.loc[date, 'monthly_investment'] = signal_investment
            total_invested += signal_investment

    final_price = data['Close'].iloc[-1]
    portfolio_value = sum(pos['shares'] * final_price for pos in positions)
    monthly_return = portfolio_value / total_invested if total_invested > 0 else 0

    # Efficient Portfolio Value Calculation (same as before)
    pv = pd.Series(0.0, index=data.index)
    for pos in positions:
        shares_series = pd.Series(0.0, index=data.index)
        shares_series.loc[pos['date']:] = pos['shares']
        close_prices = data['Close'] if isinstance(data['Close'], pd.Series) else data['Close'].squeeze()
        pv = pv.add(shares_series * close_prices, fill_value=0)
    data['monthly_portfolio'] = pv

    return data, monthly_return

# -------------------------------------------
# 3. Parameter Optimization
# -------------------------------------------

def optimize_strategy(original_data, macd_params, rsi_params, investment_amount=1000):
    results = []
    best_params = None
    best_return = -np.inf

    for short_window, long_window, signal_window, rsi_period, rsi_over_sold in product(
            macd_params['short_window'],
            macd_params['long_window'],
            macd_params['signal_window'],
            rsi_params['rsi_period'],
            rsi_params['rsi_over_sold']):

        if short_window >= long_window:
            continue

        data = original_data.copy()
        calculate_macd(data, short_window, long_window, signal_window)
        calculate_rsi(data, rsi_period)
        _, strat_return = backtest_strategy(data, rsi_over_sold, investment_amount)

        # Correctly handle the case where strat_return is a Series
        if isinstance(strat_return, pd.Series):
            strat_return = strat_return.iloc[0]  # Get the single value

        results.append((short_window, long_window, signal_window, rsi_period, rsi_over_sold, strat_return))

        if strat_return > best_return:
            best_return = strat_return
            best_params = (short_window, long_window, signal_window, rsi_period, rsi_over_sold)

    results_df = pd.DataFrame(results, columns=[
        'Short_Window', 'Long_Window', 'Signal_Window', 'RSI_Period', 'RSI_Over_Sold', 'Strategy_Return'
    ])
    return best_params, best_return, results_df


# -------------------------------------------
# 4. Plotting Functions
# -------------------------------------------
def plot_strategy_signals(data, ticker, strategy_title):
    """Plots stock price and buy signals for a given strategy."""

    plt.figure(figsize=(14, 6))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')

    buy_signals = data[data['buy_signal']]
    plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='Buy Signal')

    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.title(f'{ticker} - {strategy_title} Buy Signals')
    plt.legend()
    plt.grid(True)  # Add a grid for better readability
    plt.tight_layout()
    plt.show()

def plot_monthly_signals(data, ticker):
    """Plots stock price and buy signals for the monthly strategy."""
    plt.figure(figsize=(14, 6))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')

    buy_signals = data[data['monthly_investment'] > 0] # Find the buy signals
    plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='Buy Signal')


    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.title(f'{ticker} - Monthly Buying Strategy Signals')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------------------------------
# 5. Main Script
# -------------------------------------------
if __name__ == "__main__":
    # Fetch data from Yahoo Finance
    ticker = 'AAPL'  # Change as needed
    data = yf.download(ticker, start='2022-01-01', end='2024-01-01', auto_adjust=False, actions=False)

    data.index.name = 'Date'  # explicitly set the index name

    # -----------------------------
    # Define parameter ranges for optimization
    # You can adjust these ranges as needed.
    macd_params = {
        'short_window': range(2, 3, 1),  # e.g., 10, 11, 12, 13, 14
        'long_window': range(22, 24, 2),  # e.g., 20,...,26
        'signal_window': range(2, 4, 2)  # e.g., 8,9,10,11,12
    }
    rsi_params = {
        'rsi_period': range(14, 15,1),  # e.g., 10,...,15
        'rsi_over_sold': range(45, 50, 5)  # e.g., 25, 30, 35, 40
    }

    # -----------------------------
    # Optimize the strategy parameters
    best_params, best_strat_return, results_df = optimize_strategy(data, macd_params, rsi_params,
                                                                   investment_amount=1000)

    print("Best Parameters (Short, Long, Signal, RSI Period, RSI Over Sold):", best_params)
    print("Best Signal-based Strategy Return (Final Portfolio Value / Total Invested):", best_strat_return)

    # Save the optimization results to a CSV file
    results_df.to_csv("optimization_results.csv", index=False)

    # -----------------------------
    # Re-calculate indicators and backtest using the best parameters.
    best_short, best_long, best_signal, best_rsi_period, best_rsi_over_sold = best_params
    calculate_macd(data, best_short, best_long, best_signal)
    calculate_rsi(data, best_rsi_period)

    strat_data, strategy_return = backtest_strategy(data, best_rsi_over_sold, investment_amount=1000) # Run the strategy


    # The key change is here: Check if strategy_return is a Series
    if isinstance(strategy_return, pd.Series):
        strategy_return = strategy_return.iloc[0]  # Take the first value if it's a Series

    print(f"Final Signal-based Strategy Return: {strategy_return:.4f}")  # Now this will work correctly

    # -----------------------------
    # Backtest the monthly buying strategy for comparison.
    signal_buy_data = strat_data[strat_data['buy_signal']].copy()  # Create a copy of the buy signals data
    signal_buy_data = signal_buy_data[['investment']]  # keep only the investment column

    monthly_data, monthly_return = backtest_monthly(data, signal_buy_data,
                                                    investment_amount=1000)  # Pass signal_buy_data

    # The ULTIMATE fix: Check if monthly_return is a Series and extract the value
    if isinstance(monthly_return, pd.Series):
        monthly_return = monthly_return.iloc[0]  # Get the first element if it's a Series

    print(f"Monthly Buying Strategy Return: {monthly_return:.4f}")

    initial_price = data['Close'].iloc[0]
    final_price = data['Close'].iloc[-1]

    # Calculate the return.  If you want a single value, use .iloc[0]
    buy_and_hold_return = (final_price / initial_price)

    # Check if it's a Series and extract the single value if needed
    if isinstance(buy_and_hold_return, pd.Series):
        buy_and_hold_return = buy_and_hold_return.iloc[0]

    print(f"Buy and Hold Return: {buy_and_hold_return:.4f}")

    # -----------------------------
    # Plot the results of the signal-based strategy (using the new function)
    plot_strategy_signals(strat_data, ticker, strategy_title='Signal-based Strategy')

    # Plot the monthly strategy signals (using the new function)
    plot_monthly_signals(monthly_data, ticker)
