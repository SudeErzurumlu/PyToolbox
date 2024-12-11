import gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from sklearn.preprocessing import MinMaxScaler

class StockTradingEnvironment(gym.Env):
    def __init__(self, data, initial_balance=10000):
        """
        Initializes the trading environment with stock data and a starting balance.
        """
        super(StockTradingEnvironment, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(data.shape[1],), dtype=np.float32
        )

    def reset(self):
        """
        Resets the environment to the initial state.
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        return self.data.iloc[self.current_step].values

    def step(self, action):
        """
        Executes a trade action (buy, sell, or hold).
        """
        current_price = self.data.iloc[self.current_step]["Close"]
        if action == 1:  # Buy
            shares_to_buy = self.balance // current_price
            self.balance -= shares_to_buy * current_price
            self.shares_held += shares_to_buy
        elif action == 2:  # Sell
            self.balance += self.shares_held * current_price
            self.shares_held = 0
        self.net_worth = self.balance + self.shares_held * current_price
        self.current_step += 1

        reward = self.net_worth - self.initial_balance
        done = self.current_step >= len(self.data) - 1
        next_state = self.data.iloc[self.current_step].values if not done else None
        return next_state, reward, done, {}

# Load historical stock data
data = pd.read_csv("stock_data.csv")
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
env = StockTradingEnvironment(pd.DataFrame(data_scaled, columns=data.columns))

# Train an AI model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Example Usage
# obs = env.reset()
# done = False
# while not done:
#     action, _ = model.predict(obs)
#     obs, reward, done, _ = env.step(action)
