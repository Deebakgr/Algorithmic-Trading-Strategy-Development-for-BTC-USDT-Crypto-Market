import streamlit as st
import pandas as pd
import numpy as np
import cryptocompare

# Function to get historical BTC/USDT data for the year 2018
def get_2018_data():
    symbol = 'BTC'
    currency = 'USDT'
    start_date = '2018-01-01'
    end_date = '2022-12-31'
    
    # Fetch historical data
    data = cryptocompare.get_historical_price_day(symbol, currency, limit=2000, toTs=pd.Timestamp(end_date))
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)
    
    # Filter data for the year 2018
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df = df[(df['timestamp'] >= pd.Timestamp(start_date)) & (df['timestamp'] <= pd.Timestamp(end_date))]
    df.set_index('timestamp', inplace=True)
    
    return df

# Function to implement moving average crossover strategy
def moving_average_crossover_strategy(df, short_window, long_window):
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0.0

    # Create short simple moving average
    signals['short_mavg'] = df['close'].rolling(window=short_window, min_periods=1, center=False).mean()

    # Create long simple moving average
    signals['long_mavg'] = df['close'].rolling(window=long_window, min_periods=1, center=False).mean()

    # Create signals
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()

    return signals

# Streamlit app
def main():
    # Streamlit layout
    st.title('Moving Average Crossover Strategy')
    
    # Define parameters
    short_window = st.slider('Select Short Window', 5, 50, 10)
    long_window = st.slider('Select Long Window', 20, 200, 50)

    # Get historical BTC/USDT data for the year 2018
    df = get_2018_data()

    # Implement moving average crossover strategy
    signals = moving_average_crossover_strategy(df, short_window, long_window)

    # Display the historical data
    st.write('Historical Data (2018):')
    st.dataframe(df)

    # Display the signals
    st.write('Trading Signals:')
    st.dataframe(signals)

    # Execute trades based on signals (this is just an example, real execution depends on your setup)
    for index, row in signals.iterrows():
        if row['positions'] == 1.0:
            st.write(f"Buy: {index}")
            # Implement your buy logic here
        elif row['positions'] == -1.0:
            st.write(f"Sell: {index}")
            # Implement your sell logic here

if __name__ == '__main__':
    main()
