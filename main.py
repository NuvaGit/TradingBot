import time
import pandas as pd
import ta
import logging
import threading
import traceback
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
from alpaca.trading.requests import LimitOrderRequest
from decimal import Decimal



API_KEY = 'PKYATZH5P2AGSKR1CM15'
API_SECRET = '1pgIlow7kAEQnDYCnQiphHzIyNyrrbvEPb2Wpg6R'
BASE_URL = 'https://paper-api.alpaca.markets/v2'  # Paper trading URL

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# List of stocks to trade
# List of stocks to trade
STOCKS_TO_TRADE = [
    'AAPL', 'MSFT', 'NUKK', 'AGRI',
    'SOBR', 'CETX', 'PEGY', 'LGMK', 'CTM',
    'SEEL', 'TSLA', 'AMZN', 'GOOGL', 'NVDA',
    'AMD', 'META', 'SHOP', 'NFLX', 'DIS',
    'JPM', 'PFE', 'WMT', 'V',
    # Volatile Penny Stocks (NASDAQ)
    'DASH',  # DOORDASH
    'VERB',  # Verb Technology Company, Inc.
    'CIDM',  # Cinedigm Corp.
    'OCGN',  # Ocugen, Inc.
    'SNDL'   # Sundial Growers Inc.
]




ACCOUNT_EQUITY = float(trading_client.get_account().equity)
RISK_PER_TRADE = 0.01  # risk per trade
MAX_POSITION_SIZE = 0.15  # Maximum 10% of account equity per position

MODEL_RETRAIN_INTERVAL = 15 * 15  # Retrain every 15 mins

TIMEFRAME = '1D'
LIMIT = 15000

models = {}
scalers = {}
last_model_train_time = {}

lock = threading.Lock()

def get_live_data(symbol, timeframe='15Min', limit=1000):
    timeframe_mapping = {
        '1Min': TimeFrame(1, TimeFrameUnit.Minute),
        '5Min': TimeFrame(5, TimeFrameUnit.Minute),
        '15Min': TimeFrame(15, TimeFrameUnit.Minute),
        '30Min': TimeFrame(30, TimeFrameUnit.Minute),
        '1H': TimeFrame(1, TimeFrameUnit.Hour),
        '1D': TimeFrame(1, TimeFrameUnit.Day)
    }
    tf = timeframe_mapping.get(timeframe, TimeFrame(15, TimeFrameUnit.Minute))

    end_date = pd.Timestamp.now(tz='America/New_York')
    start_date = end_date - pd.Timedelta(days=3000)

    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start_date,
        end=end_date,
        limit=limit,
        feed='iex'
    )

    barset = data_client.get_stock_bars(request_params).df

    if barset.empty:
        logging.warning(f"No data returned for {symbol} between {start_date} and {end_date}.")
        return pd.DataFrame()

    if isinstance(barset.index, pd.MultiIndex):
        data = barset.xs(symbol)
    else:
        data = barset

    data = data.reset_index()
    logging.info(f"{symbol} - Retrieved {len(data)} rows of data")
    return data

def calculate_indicators(data):
    data.sort_values(by='timestamp', inplace=True)
    data.reset_index(drop=True, inplace=True)

    data['RSI'] = ta.momentum.RSIIndicator(data['close'], window=14).rsi()

    macd_indicator = ta.trend.MACD(data['close'], window_slow=26, window_fast=12, window_sign=9)
    data['MACD'] = macd_indicator.macd()
    data['MACD_Signal'] = macd_indicator.macd_signal()
    data['MACD_Hist'] = macd_indicator.macd_diff()

    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['SMA_50'] = data['close'].rolling(window=50).mean()

    data['Donchian_High'] = data['high'].rolling(window=20).max()
    data['Donchian_Low'] = data['low'].rolling(window=20).min()

    # Additional indicators
    data['EMA_20'] = data['close'].ewm(span=20, adjust=False).mean()
    data['Momentum'] = data['close'] - data['close'].shift(14)
    data['ATR'] = ta.volatility.AverageTrueRange(
        data['high'], data['low'], data['close'], window=14).average_true_range()

    data.ffill()
    data.dropna(inplace=True)

    return data

def generate_signals(data):
    data['Signal'] = 0

    # SMA Crossover
    data['Prev_SMA_20'] = data['SMA_20'].shift(1)
    data['Prev_SMA_50'] = data['SMA_50'].shift(1)

    golden_cross = (data['SMA_20'] > data['SMA_50']) & (data['Prev_SMA_20'] <= data['Prev_SMA_50'])
    death_cross = (data['SMA_20'] < data['SMA_50']) & (data['Prev_SMA_20'] >= data['Prev_SMA_50'])

    data.loc[golden_cross, 'Signal'] += 1
    data.loc[death_cross, 'Signal'] -= 1

    # Donchian Channel Breakout
    breakout_up = data['close'] > data['Donchian_High'].shift(1)
    breakout_down = data['close'] < data['Donchian_Low'].shift(1)

    data.loc[breakout_up, 'Signal'] += 1
    data.loc[breakout_down, 'Signal'] -= 1

    # RSI and MACD Signals
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

    data.loc[buy_signal, 'Signal'] += 1
    data.loc[sell_signal, 'Signal'] -= 1

    # Aggregate signals
    data['Aggregated_Signal'] = data['Signal']

    return data

def prepare_ml_data(data):
    features = data[['open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD', 'MACD_Signal',
                     'SMA_20', 'SMA_50', 'EMA_20', 'Momentum', 'ATR']]
    target = data['close'].shift(-1)  # Predict the next close price

    features = features[:-1]
    target = target[:-1]

    return features, target

def train_random_forest(features, target, symbol):
    if len(features) < 50:
        logging.warning(f"Not enough data to train model for {symbol}. Skipping.")
        return None, None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train_scaled, y_train)

        accuracy = model.score(X_test_scaled, y_test)
        logging.info(f"{symbol} Random Forest Regressor R^2 Score: {accuracy:.2f}")

        return model, scaler
    except Exception as e:
        logging.error(f"Error training model for {symbol}: {e}")
        logging.error(traceback.format_exc())
        return None, None

def predict_next_close(model, scaler, last_row):
    features = last_row[['open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD', 'MACD_Signal',
                         'SMA_20', 'SMA_50', 'EMA_20', 'Momentum', 'ATR']]
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return prediction[0]

def get_open_positions():
    positions = trading_client.get_all_positions()
    current_positions = {position.symbol: float(position.qty) for position in positions}
    return current_positions

def is_stock_held(symbol, current_positions):
    return symbol in current_positions and current_positions[symbol] > 0

def calculate_position_size(symbol, entry_price, stop_loss_price):
    account = trading_client.get_account()
    equity = float(account.equity)
    max_risk = equity * RISK_PER_TRADE

    risk_per_share = abs(entry_price - stop_loss_price)
    if risk_per_share == 0:
        return 0  # Avoid division by zero

    position_size = max_risk / risk_per_share
    max_position_value = equity * MAX_POSITION_SIZE
    max_shares = max_position_value / entry_price

    shares_to_buy = min(position_size, max_shares)
    return int(shares_to_buy)


from decimal import Decimal

def execute_trades(data, symbol, current_positions):
    last_signal = data.iloc[-1]['Aggregated_Signal']
    current_price = data.iloc[-1]['close']
    atr = data.iloc[-1]['ATR']

    stop_loss = current_price - (2 * atr)
    take_profit = current_price + (4 * atr)

    qty_to_trade = calculate_position_size(symbol, current_price, stop_loss)

    current_qty = current_positions.get(symbol, 0)  # Default to 0 if the stock is not held

    logging.info(
        f"{symbol} - Last Signal: {last_signal}, Current Quantity Held: {current_qty}, Quantity to Trade: {qty_to_trade}")

    try:
        if last_signal > 0:
            max_position_value = 1000  # Maximum allowed position of $1,000
            max_shares = max_position_value / current_price
            remaining_shares_to_buy = int(max_shares - current_qty)

            if remaining_shares_to_buy > 0 and qty_to_trade > 0:
                logging.info(
                    f"Placing BUY LIMIT order for {symbol}, Remaining Shares to Buy: {remaining_shares_to_buy}")

                # Calculate limit price slightly below current price (e.g., 0.5% lower)
                limit_price = round(current_price * 0.995, 2)

                limit_order_data = LimitOrderRequest(
                    symbol=symbol,
                    qty=min(qty_to_trade, remaining_shares_to_buy),
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    limit_price=Decimal(str(limit_price))
                )
                order = trading_client.submit_order(limit_order_data)
                logging.info(f"Limit Order Response: {order}")
            else:
                logging.info(f"Already at max position for {symbol}, skipping additional BUY.")

        elif last_signal < 0:
            if current_qty > 0:
                logging.info(f"Placing SELL LIMIT order for {symbol}, Quantity to Sell: {current_qty}")

                # Calculate limit price slightly above current price (e.g., 0.5% higher)
                limit_price = round(current_price * 1.005, 2)

                limit_order_data = LimitOrderRequest(
                    symbol=symbol,
                    qty=current_qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    limit_price=Decimal(str(limit_price))
                )
                order = trading_client.submit_order(limit_order_data)
                logging.info(f"Limit Order Response: {order}")
            else:
                logging.info(f"Not holding {symbol}, skipping SELL order.")
        else:
            logging.info(f"No trading action required for {symbol}.")

    except Exception as e:
        logging.error(f"Error executing trade for {symbol}: {e}")
        logging.error(traceback.format_exc())

def is_market_open():
    clock = trading_client.get_clock()
    logging.info(f"Market Open: {clock.is_open}")
    return clock.is_open

def retrain_model_if_needed(symbol, data):
    current_time = time.time()
    if symbol not in last_model_train_time or (current_time - last_model_train_time[symbol]) > MODEL_RETRAIN_INTERVAL:
        features, target = prepare_ml_data(data)
        model, scaler = train_random_forest(features, target, symbol)
        if model is not None and scaler is not None:
            with lock:
                models[symbol] = model
                scalers[symbol] = scaler
                last_model_train_time[symbol] = current_time
            logging.info(f"Retrained model for {symbol}")
        else:
            logging.warning(f"Model training for {symbol} failed. Skipping.")

def process_symbol(symbol):
    try:
        logging.info(f"Processing {symbol}")

        data = get_live_data(symbol, TIMEFRAME, limit=LIMIT)

        if data.empty:
            logging.warning(f"No data retrieved for {symbol}. Skipping.")
            return

        data = calculate_indicators(data)

        if data.empty or len(data) < 50:
            logging.warning(f"Not enough data after indicator calculation for {symbol}. Skipping.")
            return

        data = generate_signals(data)

        # Retrain model if needed
        retrain_model_if_needed(symbol, data)

        # Predict next close price
        model = models.get(symbol)
        scaler = scalers.get(symbol)

        if model is None or scaler is None:
            logging.warning(f"No model available for {symbol}, skipping.")
            return

        next_close_prediction = predict_next_close(model, scaler, data.iloc[-1])
        logging.info(f"Predicted next close price for {symbol}: {next_close_prediction}")

        # Decide based on prediction
        current_close = data.iloc[-1]['close']
        if next_close_prediction > current_close:
            data.at[data.index[-1], 'Aggregated_Signal'] += 1
        elif next_close_prediction < current_close:
            data.at[data.index[-1], 'Aggregated_Signal'] -= 1

        # Fetch current positions
        current_positions = get_open_positions()

        # Execute trades
        execute_trades(data, symbol, current_positions)

    except Exception as e:
        logging.error(f"An error occurred while processing {symbol}: {e}")
        logging.error(traceback.format_exc())

def main():
    max_workers = min(5,len(STOCKS_TO_TRADE))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            try:
                if not is_market_open():
                    logging.info("Market is closed. Waiting for it to open.")
                    time.sleep(60)
                    continue

                logging.info("Market is open. Starting processing of symbols.")

                futures = [executor.submit(process_symbol, symbol) for symbol in STOCKS_TO_TRADE]
                for future in futures:
                    future.result()  # Wait for all threads to complete

                logging.info("Completed processing all symbols. Sleeping until next iteration.")
                time.sleep(1)

            except KeyboardInterrupt:
                logging.info("Bot stopped by user.")
                break
            except Exception as e:
                logging.error(f"An unexpected error occurred: {e}")
                logging.error(traceback.format_exc())
                time.sleep(60)

if __name__ == "__main__":
    main()
