
# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import cryptocompare
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

# Data Acquisition Function
@st.cache_data
def get_historical_data():
    symbol = 'BTC'
    currency = 'USDT'
    start_date = '2018-01-01'
    end_date = '2024-11-01'

    # Fetch historical data (daily)
    data = cryptocompare.get_historical_price_day(symbol, currency, limit=2000, toTs=pd.Timestamp(end_date))
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('timestamp', inplace=True)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    return df.copy()

# Data Preparation Function
def prepare_data(df):
    if 'close' not in df.columns:
        raise KeyError("The 'close' column is missing in the DataFrame.")

    close_prices = df[['close']].copy()
    scaler = StandardScaler()
    close_prices['scaled_close'] = scaler.fit_transform(close_prices[['close']])
    return close_prices, scaler

# Kalman Filter Function
def apply_kalman_filter(df):
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    state_means, _ = kf.filter(df['scaled_close'].values)
    df['kalman_prediction'] = state_means
    return df

# Function to Add Buy and Sell Signals
def add_signals(predicted_prices, actual_prices):
    # Convert to Pandas Series
    predicted_prices = pd.Series(predicted_prices, index=actual_prices.index)
    signals = pd.DataFrame(index=actual_prices.index)
    signals['actual'] = actual_prices
    signals['predicted'] = predicted_prices

    # Define buy and sell signals
    signals['signal'] = 0
    signals.loc[signals['predicted'] > signals['actual'] * 1.01, 'signal'] = 1  # Buy Signal
    signals.loc[signals['predicted'] < signals['actual'] * 0.99, 'signal'] = -1  # Sell Signal

    signals['positions'] = signals['signal'].diff()
    return signals

# Plotting Function
def plot_predictions(actual_prices, predicted_prices, signals, title):
    actual_prices = pd.Series(actual_prices, index=signals.index)
    predicted_prices = pd.Series(predicted_prices, index=signals.index)

    plt.figure(figsize=(14, 7))
    plt.plot(actual_prices, color='blue', label='Actual Prices')
    plt.plot(predicted_prices, color='red', label='Kalman Filter Predictions')
    plt.scatter(signals.index[signals['positions'] == 1], actual_prices[signals['positions'] == 1], marker='^', color='green', label='Buy Signal', s=100)
    plt.scatter(signals.index[signals['positions'] == -1], actual_prices[signals['positions'] == -1], marker='v', color='red', label='Sell Signal', s=100)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid()
    st.pyplot(plt)

# Backtesting Function
def backtest(signals):
    initial_balance = 10000
    balance = initial_balance
    position = 0
    portfolio_values = []

    actual_prices = signals['actual']

    for i in range(1, len(signals)):
        if signals['positions'].iloc[i] == 1:  # Buy
            if balance > 0:
                position = balance / actual_prices.iloc[i]
                balance = 0
        elif signals['positions'].iloc[i] == -1:  # Sell
            if position > 0:
                balance = position * actual_prices.iloc[i]
                position = 0

        current_value = balance if balance > 0 else position * actual_prices.iloc[i]
        portfolio_values.append(current_value)

    # Final balance calculation
    if position > 0:
        balance = position * actual_prices.iloc[-1]

    final_value = balance
    total_return = (final_value - initial_balance) / initial_balance * 100

    n_days = len(signals) / 24  # Convert to days assuming hourly data
    annualized_return = ((final_value / initial_balance) ** (365 / n_days)) - 1 if final_value != initial_balance else 0

    return total_return, annualized_return * 100, portfolio_values

# Streamlit App
def main():
    st.title("BTC/USDT Hourly Price Prediction with Kalman Filter and Backtesting")

    # Load historical data
    df = get_historical_data()

    # Display historical data
    st.write("Historical Data (2018 - 2024):")
    st.dataframe(df)

    # Prepare data
    close_prices, scaler = prepare_data(df)

    # Apply Kalman Filter
    st.write("Applying Kalman Filter for Smoothing...")
    close_prices = apply_kalman_filter(close_prices)

    # Inverse transform the predictions
    predicted_prices = scaler.inverse_transform(close_prices['kalman_prediction'].values.reshape(-1, 1)).flatten()
    actual_prices = close_prices['close'].values

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mae = mean_absolute_error(actual_prices, predicted_prices)
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"MAE: {mae:.2f}")

    # Add buy and sell signals
    actual_prices_series = pd.Series(actual_prices, index=close_prices.index)
    predicted_prices_series = pd.Series(predicted_prices, index=close_prices.index)
    signals = add_signals(predicted_prices_series, actual_prices_series)

    # Display signals
    st.write("Buy and Sell Signals Summary:")
    st.write(signals['positions'].value_counts())

    # Plot predictions and signals
    plot_predictions(actual_prices_series, predicted_prices_series, signals, "BTC/USDT Hourly Price Prediction with Buy/Sell Signals")

    # Backtest the strategy
    total_return, annualized_return, portfolio_values = backtest(signals)

    # Display backtesting results
    st.write(f"Total Return: {total_return:.2f}%")
    st.write(f"Annualized Return: {annualized_return:.2f}%")

if __name__ == '__main__':
    main()

