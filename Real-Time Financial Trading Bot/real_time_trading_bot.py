import pandas as pd
import yfinance as yf
import numpy as np
import time
from binance.client import Client

class TradingBot:
    def __init__(self, api_key, api_secret):
        """
        Initializes the trading bot with API credentials.
        """
        self.client = Client(api_key, api_secret)
        self.balance = 1000  # Starting balance in USD
        self.positions = {}

    def fetch_market_data(self, symbol, interval="1h", lookback="30d"):
        """
        Fetches historical market data from Binance.
        """
        data = self.client.get_historical_klines(symbol, interval, lookback)
        df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume", "extra"])
        df["close"] = df["close"].astype(float)
        return df

    def mean_reversion_strategy(self, prices, window=14):
        """
        Implements a mean reversion strategy.
        """
        mean_price = prices.rolling(window).mean()
        std_dev = prices.rolling(window).std()
        upper_band = mean_price + std_dev
        lower_band = mean_price - std_dev

        if prices.iloc[-1] < lower_band.iloc[-1]:
            return "buy"
        elif prices.iloc[-1] > upper_band.iloc[-1]:
            return "sell"
        return "hold"

    def execute_trade(self, symbol, action):
        """
        Executes a trade.
        """
        if action == "buy" and self.balance > 0:
            self.positions[symbol] = self.balance / 2
            self.balance -= self.positions[symbol]
            print(f"Bought {symbol} for ${self.positions[symbol]}")
        elif action == "sell" and symbol in self.positions:
            self.balance += self.positions.pop(symbol)
            print(f"Sold {symbol} for ${self.balance}")
        else:
            print("No action taken.")

    def run(self, symbol):
        """
        Runs the trading bot in real-time.
        """
        while True:
            data = self.fetch_market_data(symbol)
            prices = data["close"]
            action = self.mean_reversion_strategy(prices)
            self.execute_trade(symbol, action)
            time.sleep(3600)  # Wait for the next hour

# Example Usage:
# bot = TradingBot("your_api_key", "your_api_secret")
# bot.run("BTCUSDT")
