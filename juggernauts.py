import time
import pandas as pd
import ta
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Replace with your Alpaca API credentials
API_KEY = 'YOUR_API_KEY'
API_SECRET = 'YOUR_API_SECRET'

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
    data['RSI'] = ta.momentum.RSIIndicator(data['close'], window=14).rsi()

    # Calculate MACD
    macd_indicator = ta.trend.MACD(data['close'], window_slow=26, window_fast=12, window_sign=9)
    data['MACD'] = macd_indicator.macd()
    data['MACD_Signal'] = macd_indicator.macd_signal()
    data['MACD_Hist'] = macd_indicator.macd_diff()

    # Calculate SMA (Simple Moving Average)
    data['SMA_50'] = data['close'].rolling(window=50).mean()
    data['SMA_200'] = data['close'].rolling(window=200).mean()

    # Calculate Donchian Channels
    data['Donchian_High'] = data['high'].rolling(window=20).max()
    data['Donchian_Low'] = data['low'].rolling(window=20).min()

    # Fill missing values and remove any remaining NaNs
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)

    return data

def generate_signals(data):
    data['Signal'] = 0

    # Golden Cross Strategy
    data['Prev_SMA_50'] = data['SMA_50'].shift(1)
    data['Prev_SMA_200'] = data['SMA_200'].shift(1)

    golden_cross = (data['SMA_50'] > data['SMA_200']) & (data['Prev_SMA_50'] <= data['Prev_SMA_200'])
    death_cross = (data['SMA_50'] < data['SMA_200']) & (data['Prev_SMA_50'] >= data['Prev_SMA_200'])

    data.loc[golden_cross, 'Signal'] = 1
    data.loc[death_cross, 'Signal'] = -1

    # Donchian Channel Breakout Strategy
    breakout_up = data['close'] > data['Donchian_High'].shift(1)
    breakout_down = data['close'] < data['Donchian_Low'].shift(1)

    data.loc[breakout_up, 'Signal'] = 1
    data.loc[breakout_down, 'Signal'] = -1

    # RSI and MACD Strategy
    data['MACD_Cross'] = data['MACD'] - data['MACD_Signal']
    data['Prev_MACD_Cross'] = data['MACD_Cross'].shift(1)

    buy_signal = (
        (data['RSI'] < 30) &
        (data['MACD_Cross'] > 0) &
        (data['Prev_MACD_Cross'] <= 0)
    )
    sell_signal = (
        (data['RSI'] > 70) &
        (data['MACD_Cross'] < 0) &
        (data['Prev_MACD_Cross'] >= 0)
    )

    data.loc[buy_signal, 'Signal'] = 1
    data.loc[sell_signal, 'Signal'] = -1

    return data

def prepare_ml_data(data):
    # Features and Target
    features = data[['open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD', 'MACD_Signal', 'SMA_50', 'SMA_200']]
    target = data['close'].shift(-1)  # Predict the next close price

    # Remove the last row (as target is NaN there)
    features = features[:-1]
    target = target[:-1]

    return features, target

def train_random_forest(features, target):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Accuracy
    accuracy = model.score(X_test_scaled, y_test)
    print(f"Random Forest Regressor Accuracy: {accuracy:.2f}")

    return model, scaler

def predict_next_close(model, scaler, last_row):
    features = last_row[['open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD', 'MACD_Signal', 'SMA_50', 'SMA_200']]
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return prediction[0]

def execute_trades(data, symbol, qty=1):
    last_signal = data.iloc[-1]['Signal']
    print(f"Last signal: {last_signal}")

    if last_signal == 1:
        print(f"Placing a **BUY** order for {symbol}")
        # Create a market order request
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        # Place an actual order
        order = trading_client.submit_order(market_order_data)
        print(order)
    elif last_signal == -1:
        print(f"Placing a **SELL** order for {symbol}")
        # Create a market order request
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        # Place an actual order
        order = trading_client.submit_order(market_order_data)
        print(order)
    else:
        print("No action needed.")

def is_market_open():
    clock = trading_client.get_clock()
    return clock.is_open

def main():
    symbol = 'AAPL'
    timeframe = '1Min'
    limit = 1000  # Increased limit for better ML training
    qty = 1

    # Fetch historical data once for training
    historical_data = get_live_data(symbol, timeframe, limit)
    historical_data = calculate_indicators(historical_data)
    historical_data = generate_signals(historical_data)

    # Prepare data for ML model
    features, target = prepare_ml_data(historical_data)

    # Train Random Forest model
    model, scaler = train_random_forest(features, target)

    # Start live trading
    while True:
        try:
            if not is_market_open():
                print("Market is closed. Waiting for it to open.")
                time.sleep(60)
                continue

            data = get_live_data(symbol, timeframe, limit=200)
            data = calculate_indicators(data)
            data = generate_signals(data)

            # Predict next close price
            next_close_prediction = predict_next_close(model, scaler, data.iloc[-1])
            print(f"Predicted next close price: {next_close_prediction}")

            # Decide based on prediction
            current_close = data.iloc[-1]['close']
            if next_close_prediction > current_close:
                data.at[data.index[-1], 'Signal'] = 1  # Buy signal
            elif next_close_prediction < current_close:
                data.at[data.index[-1], 'Signal'] = -1  # Sell signal
            else:
                data.at[data.index[-1], 'Signal'] = 0  # Hold

            # Execute trades
            execute_trades(data, symbol, qty)

            # Sleep for the interval duration
            time.sleep(60)
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
