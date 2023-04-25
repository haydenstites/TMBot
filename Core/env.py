import subprocess, os
import gymnasium as gym
import numpy as np
from pathlib import Path
from IO import read_from_tmdata

# gym.spaces.Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)

# Observation space of each TMData var
obs_vars = np.array([
    [-1000.0, 1000.0], # velocityY
    [-1000.0, 1000.0], # velocityX
    [1.0, 5.0], # gear
])
obs_lows, obs_highs = np.float32(obs_vars[:,0]), np.float32(obs_vars[:,1])

total_vars = 3 # Equals length of active

class TMEnv(gym.Env):
    def __init__(self, op_path : str, active : tuple):
        """Action space is XY game input

        Observation space TMData output vars
        """

        # e.g. active = (True, True, False, etc.)
        # Must match to variables from TMData (VelocityY, VelocityX, Gear, etc.)
        assert len(active) == len(obs_vars) == total_vars # Verify all lengths match

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=obs_lows, high=obs_highs)

        self.op_path = op_path
        self.active = active

    def step(self, action):
        vars = read_from_tmdata(self.op_path, self.active)
        

    def reset(self):
        pass
