import numpy as np
import pandas as pd
import gym
from stable_baselines3 import PPO

class PortfolioEnv(gym.Env):
    def __init__(self, stock_data, initial_balance=10000):
        """
        Custom portfolio optimization environment.
        """
        super(PortfolioEnv, self).__init__()
        self.stock_data = stock_data
        self.n_assets = stock_data.shape[1]
        self.initial_balance = initial_balance
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_assets,), dtype=np.float32
        )
        self.reset()

    def reset(self):
        """
        Resets the environment.
        """
        self.balance = self.initial_balance
        self.current_step = 0
        self.done = False
        return self.stock_data.iloc[self.current_step].values

    def step(self, action):
        """
        Executes one step in the environment.
        """
        weights = action / np.sum(action)
        returns = self.stock_data.iloc[self.current_step].values
        portfolio_return = np.sum(weights * returns)
        self.balance += self.balance * portfolio_return
        self.current_step += 1
        if self.current_step >= len(self.stock_data) - 1:
            self.done = True
        return self.stock_data.iloc[self.current_step].values, portfolio_return, self.done, {}

# Example Usage
stock_data = pd.DataFrame(np.random.randn(100, 5) / 100)  # Simulated daily returns
env = PortfolioEnv(stock_data)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
