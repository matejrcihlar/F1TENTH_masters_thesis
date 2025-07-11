
import gym
from gym import spaces
import numpy as np

class F110Wrapper(gym.Env):
    def __init__(self, env):
        super(F110Wrapper, self).__init__()
        self.env = env

        # Define the observation and action space manually
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(input_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-0.34, 0.0]),
            high=np.array([0.34, 3.0]),
            dtype=np.float32
        )

    def reset(self):
        obs, _, _, _ = self.env.reset()
        return preprocess_observation(obs, ...)  # Adjust this for your input

    def step(self, action):
        action_full = np.array([[action[0], action[1]], [0.0, 0.0]])
        obs, reward, done, info = self.env.step(action_full)
        processed_obs = preprocess_observation(obs, ...)  # Your preprocessing
        return processed_obs, reward, done, info

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()
