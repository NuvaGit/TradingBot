import os
import time
import pandas as pd
import pandas_ta as ta  # For technical indicators

# Alpaca-Py imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient


# Load API credentials from environment variables
API_KEY = 'AKDTUCR3K56BVCLQ87WY'
API_SECRET = 'BTaCgI8Ls7flz4rxoPainzFfJ3n811PCMYLEuzT4'
BASE_URL = 'https://paper-api.alpaca.markets'  # Paper trading URL





trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

def get_live_data(symbol, timeframe='1Min', limit=100):
    # Map string to TimeFrame object
    timeframe_mapping = {
        '1Min': TimeFrame(1, TimeFrameUnit.Minute),
        '5Min': TimeFrame(5, TimeFrameUnit.Minute),
        '15Min': TimeFrame(15, TimeFrameUnit.Minute),
        '1H': TimeFrame(1, TimeFrameUnit.Hour),
        '1D': TimeFrame(1, TimeFrameUnit.Day)
    }
    tf = timeframe_mapping.get(timeframe, TimeFrame(1, TimeFrameUnit.Minute))

    # Create request parameters
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        limit=limit
    )

    # Fetch bars
    barset = data_client.get_stock_bars(request_params).df

    # If data for multiple symbols, extract the symbol's data
    if isinstance(barset.index, pd.MultiIndex):
        data = barset.xs(symbol)
    else:
        data = barset

    data = data.reset_index()
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
        # Create a market order request
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        # Uncomment the line below to place an actual order
        # order = trading_client.submit_order(market_order_data)
        # print(order)
    elif last_signal == -1:
        print(f"Would place a **SELL** order for {symbol}")
        # Create a market order request
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        # Uncomment the line below to place an actual order
        # order = trading_client.submit_order(market_order_data)
        # print(order)
    else:
        print("No action needed.")

def is_market_open():
    clock = trading_client.get_clock()
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
