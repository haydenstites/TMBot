import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary
from IO import get_observations, write_actions, write_reset
from typing import Any

# e.g. enabled = {
#     "frame" : True,
#     "velocity" : False,
#     "slip" : True,
# }
# e.g. frame_shape = (1, 36, 36) # Channels, height, width
class TMBaseEnv(gym.Env):
    """Trackmania Base Environment, :meth:`reward` function must be overridden for custom implementations."""
    def __init__(self, op_path : str, frame_shape : tuple[int] = None, enabled : dict[str, bool] = None, rew_enabled : dict[str, bool] = None):
        r"""Initialization parameters for TMBaseEnv. Parameters in enabled and rew_enabled should match output variables in TMData.

        Args:
            op_path (str) : Path to Openplanet installation folder. Typically "C:\Users\NAME\OpenplanetNext"

            frame_shape (tuple[int], Optional) : Observation size of image frame passed to CNN, formatted (channels, height, width).
            Default is (1, 36, 36), which is also the elementwise minimum frame size.

            enabled (dict[str, bool], Optional) : Dictionary describing enabled parameters in observation space. Default is True for every key.

            rew_enabled (dict[str, bool], Optional) : Dictionary describing enabled parameters for reward shaping. Default is True for every key.
        """

        min_frame_shape = (1, 36, 36)
        frame_shape = min_frame_shape if frame_shape is None else frame_shape
        assert frame_shape >= min_frame_shape, f"frame_shape {frame_shape} is less than min_frame_shape {min_frame_shape}"


        # Discrete(2) converted from bool, TODO: More elegant solution?
        obs_vars = {
            "frame" : Box(low=0, high=255, shape=frame_shape, dtype=np.uint8), # Image specs of SB3
            "velocity" : Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32), # Normed from -1000 to 1000
            "gear" : Discrete(5),
            "drift" : Discrete(2),
            "slip" : Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32), # Normed from 0 to 1
            "material" : Discrete(7), # None, tech, plastic, dirt, grass, ice, other
            "grounded" : Discrete(2),
        }
        rew_keys = ("top_contact", "race_state", "checkpoint", "total_checkpoints", "bonk_time", "bonk_score")

        enabled = {} if enabled is None else enabled
        for key in list(obs_vars):
            enabled.setdefault(key, True) # Any unspecified key defaults to True
            if enabled[key] is False:
                obs_vars.pop(key) # obs_vars is left with only used keys as specified by enabled
        
        rew_enabled = {} if rew_enabled is None else rew_enabled
        for key in rew_keys:
            rew_enabled.setdefault(key, True)

        # TODO: Gamepad/Box action space option
        self.action_space = MultiDiscrete([3, 3], dtype=np.int32)
        self.observation_space = gym.spaces.Dict(obs_vars)

        self.op_path = op_path
        self.enabled = enabled
        self.rew_enabled = rew_enabled

        self.held_obs = {}
        self.held_rew = {}

    def step(self, action):
        write_actions(self.op_path, action)

        try:
            obs, rew_vars = get_observations(self.op_path, self.enabled, self.rew_enabled)
            self.held_obs, self.held_rew = obs, rew_vars
        except:
            # Triggers if file is being written
            obs, rew_vars = self.held_obs, self.held_rew

        reward, terminated, truncated, info = self.reward(obs, rew_vars)

        return obs, reward, terminated, truncated, info

    def reward(self, obs : dict[str, Any], rew_vars : dict[str, Any]) -> tuple[float, bool, bool, dict]:
        r"""Reward function for agent. Must be defined for any implementation extending TMBaseEnv.

        Args:
            obs (dict[str, Any]) : Observations given by TMBaseEnv during :meth:`step`, not supplied by user. Key is str definition of variable.

            rew_vars (dict[str, Any]) : Observations given by TMBaseEnv during :meth:`step` not directly used in training of model, not supplied by user. 
            May still be useful for reward shaping. Key is str definition of variable.

        
        Returns:
            reward (float) : Reward value given to model for given timestep.

            terminated (bool) : Set to True to "reset" enviroment. Call when agent reaches terminal state.

            truncated (bool) : Set to True to "reset" environment for external reasons e.g. :meth:`TimeLimit` wrapper.

            info (dict, Optional) : Debug information. Can be left blank if unused.
        """

        NotImplementedError

        # return reward, terminated, truncated, info

    def reset(self):
        obs = get_observations(self.op_path, self.enabled)

        write_reset(self.op_path)

        info = {}

        return obs, info
    
    def close(self):
        write_reset(self.op_path)
