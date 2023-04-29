import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary
import numpy as np
from collections import OrderedDict
from IO import read_from_tmdata
from util import norm_float, binary_strbool, mat_index

# self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
# self.observation_space = gym.spaces.Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)

# VelocityY float
# VelocityX float
# Gear int gear
# Drift bool drift
# Slip float slip
# Material string -> float traction? int surfaceType?
# Grounded bool
# TopContact bool
# RaceState string -> 0 1 2
# Checkpoint int
# TotalCheckpoints int
# BonkTime int
# BonkScore float

# Observation space of each TMData var
# obs_vars = np.array([
#     [-1000.0, 1000.0], # velocityY
#     [-1000.0, 1000.0], # velocityX
#     [1.0, 5.0], # gear
# ])
# obs_lows, obs_highs = np.float32(obs_vars[:,0]), np.float32(obs_vars[:,1])



# e.g. enabled = {
#     "frame" : True,
#     "velocity" : False,
#     "slip" : True,
# }
# e.g. frame_shape = (1, 32, 32) # Channels, height, width
class TMEnv(gym.Env):
    def __init__(self, op_path : str, frame_shape : tuple[int] = None, enabled : dict[str, bool] = None):
        """Action space is XY game input

        Observation space TMData output vars
        """
        # Discrete(2) converted from bool, TODO: More elegant solution?
        obs_vars = {
            "frame" : Box(low=0, high=255, shape=frame_shape, dtype=np.uint8), # Image specs of SB3
            "velocity" : Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32), # Normed from -1000 to 1000
            "gear" : Discrete(5, start=1),
            "drift" : Discrete(2),
            "slip" : Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32), # Normed from 0 to 1
            "material" : Discrete(7), # None, tech, plastic, dirt, grass, ice, other
            "grounded" : Discrete(2),
        }

        enabled = {} if enabled == None else enabled 
        for key in list(obs_vars):
            enabled.setdefault(key, True) # Any unspecified key defaults to True
            if enabled[key] is False:
                obs_vars.pop(key)

        # obs_vars is left with only used keys as specified by enabled

        self.action_space = MultiDiscrete([3, 3], dtype=np.int32)
        self.observation_space = gym.spaces.Dict(obs_vars)

        self.op_path = op_path
        self.enabled = enabled

    def step(self, action):
        obs, rew_vars = self._get_observations()     

    def reset(self):
        pass

    def _get_observations(self):
        vars = read_from_tmdata(self.op_path)
        obs = OrderedDict()

        if self.enabled["velocity"]:
            obs["velocity"] = np.array([norm_float(vars[0], -1000, 1000), norm_float(vars[1], -1000, 1000)], dtype=np.float32)
            del vars[0:2]
        if self.enabled["gear"]:
            obs["gear"] = int(vars[0])
            del vars[0]
        if self.enabled["drift"]:
            obs["drift"] = binary_strbool(vars[0])
            del vars[0]
        if self.enabled["slip"]:
            obs["slip"] = np.array([norm_float(vars[0], 0, 1)], dtype=np.float32)
            del vars[0]
        if self.enabled["material"]:
            obs["material"] = mat_index(vars[0])
            del vars[0]
        if self.enabled["grounded"]:
            obs["grounded"] = binary_strbool(vars[0])
            del vars[0]

        return obs, vars
