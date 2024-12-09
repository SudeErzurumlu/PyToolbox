import gym
import numpy as np
from stable_baselines3 import PPO

class DroneEnv(gym.Env):
    def __init__(self):
        """
        Initializes a custom drone navigation environment.
        """
        super(DroneEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10, 10), dtype=np.float32)
        self.state = np.zeros((10, 10))
        self.target = (9, 9)
        self.drone_position = (0, 0)

    def reset(self):
        """
        Resets the environment.
        """
        self.state = np.zeros((10, 10))
        self.drone_position = (0, 0)
        self.state[self.target] = 1
        return self.state.flatten()

    def step(self, action):
        """
        Takes an action and returns the next state, reward, and whether the episode is done.
        """
        x, y = self.drone_position
        if action == 0 and x > 0: x -= 1  # Up
        if action == 1 and x < 9: x += 1  # Down
        if action == 2 and y > 0: y -= 1  # Left
        if action == 3 and y < 9: y += 1  # Right
        self.drone_position = (x, y)
        
        reward = 1 if self.drone_position == self.target else -0.1
        done = self.drone_position == self.target
        self.state = np.zeros((10, 10))
        self.state[self.target] = 1
        self.state[self.drone_position] = -1
        return self.state.flatten(), reward, done, {}

# Example Usage:
# env = DroneEnv()
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10000)
