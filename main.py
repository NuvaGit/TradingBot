import os
import time
import alpaca_trade_api as tradeapi
import pandas as pd
import pandas_ta as ta  # Use pandas-ta for technical indicators

# Load API keys from environment variables
# Load API keys from environment variables
API_KEY = os.environ.get('APCA_API_KEY_ID')  # Correct env var for Alpaca API Key
API_SECRET = os.environ.get('APCA_API_SECRET_KEY')  # Correct env var for Alpaca API Secret
BASE_URL = 'https://paper-api.alpaca.markets'  # Use 'https://api.alpaca.markets' for live trading

# Initialize the API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')


def get_live_data(symbol, timeframe='1Min', limit=100):
    from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit

    # Map string to TimeFrame object
    timeframe_mapping = {
        '1Min': TimeFrame(1, TimeFrameUnit.Minute),
        '5Min': TimeFrame(5, TimeFrameUnit.Minute),
        '15Min': TimeFrame(15, TimeFrameUnit.Minute),
        '1H': TimeFrame(1, TimeFrameUnit.Hour),
        '1D': TimeFrame(1, TimeFrameUnit.Day)
    }
    tf = timeframe_mapping.get(timeframe, TimeFrame(1, TimeFrameUnit.Minute))

    # Fetch bars
    barset = api.get_bars(symbol, tf, limit=limit).df

    # Since we're fetching data for a single symbol, no need to filter
    data = barset.reset_index()

    return data


def calculate_indicators(data):
    # Calculate RSI
    data['RSI'] = ta.rsi(data['close'], length=14)

    # Calculate MACD
    macd = ta.macd(data['close'], fast=12, slow=26, signal=9)
    data = pd.concat([data, macd], axis=1)

    # Fill missing values and remove any remaining NaNs
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)

    return data


def generate_signals(data):
    data['Signal'] = 0
    data['MACD_Signal_Cross'] = data['MACD_12_26_9'] - data['MACDs_12_26_9']
    data['Prev_MACD_Signal_Cross'] = data['MACD_Signal_Cross'].shift(1)

    buy_signal_condition = (
            (data['RSI'] < 30) &
            (data['MACD_Signal_Cross'] > 0) &
            (data['Prev_MACD_Signal_Cross'] <= 0)
    )
    data.loc[buy_signal_condition, 'Signal'] = 1

    sell_signal_condition = (
            (data['RSI'] > 70) &
            (data['MACD_Signal_Cross'] < 0) &
            (data['Prev_MACD_Signal_Cross'] >= 0)
    )
    data.loc[sell_signal_condition, 'Signal'] = -1

    return data


def execute_trades_test(data, symbol, qty=1):
    last_signal = data.iloc[-1]['Signal']
    print(f"Last signal: {last_signal}")

    if last_signal == 1:
        print(f"Would place a **BUY** order for {symbol}")
    elif last_signal == -1:
        print(f"Would place a **SELL** order for {symbol}")
    else:
        print("No action needed.")


def is_market_open():
    clock = api.get_clock()
    return clock.is_open


def main():
    symbol = 'AAPL'
    timeframe = '1Min'
    limit = 100
    qty = 1

    while True:
        try:
            if not is_market_open():
                print("Market is closed. Waiting for it to open.")
                time.sleep(60)
                continue

            data = get_live_data(symbol, timeframe, limit)
            data = calculate_indicators(data)
            data = generate_signals(data)

            # Print the latest data for inspection
            print(data[['timestamp', 'close', 'RSI', 'MACD_12_26_9', 'Signal']].tail())

            # Use the test execute trades function
            execute_trades_test(data, symbol, qty)

            # Sleep for the interval duration
            time.sleep(60)
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()