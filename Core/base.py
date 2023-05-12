import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from gymnasium.core import ActType
from IO import get_observations, write_actions, write_alt
from util import get_frame, get_default_op_path
from typing import Any
from pathlib import Path

class TMBaseEnv(gym.Env):
    """Trackmania Base Environment, :meth:`reward` function must be overridden for custom implementations."""
    def __init__(self, op_path : Path = None, frame_shape : tuple[int] = None, enabled : dict[str, bool] = None, rew_enabled : dict[str, bool] = None, square_frame : bool = True):
        r"""Initialization parameters for TMBaseEnv. Parameters in enabled and rew_enabled should match output variables in TMData.

        Args:
            op_path (Path) : Path to Openplanet installation folder. Default is "C:\Users\NAME\OpenplanetNext".

            frame_shape (tuple[int]) : Observation size of image frame passed to CNN, formatted (channels, height, width).
            Must be at least (1, 36, 36). Default is (1, 50, 50).

            enabled (dict[str, bool]) : Dictionary describing enabled parameters in observation space. Default is True for every key.

            rew_enabled (dict[str, bool]) : Dictionary describing enabled parameters for reward shaping. Default is True for every key.

            square_frame (bool) : Crops observed frames to squares. Default is True.
        """
        self.op_path = get_default_op_path() if op_path is None else op_path

        min_frame_shape = (1, 36, 36)
        default_frame_shape = (1, 50, 50)
        frame_shape = default_frame_shape if frame_shape is None else frame_shape
        assert frame_shape >= min_frame_shape, f"frame_shape {frame_shape} is less than min_frame_shape {min_frame_shape}"

        obs_vars = {
            "frame" : Box(low=0, high=255, shape=frame_shape, dtype=np.uint8), # Image specs of SB3
            "velocity" : Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32), # Normed from -1000 to 1000
            "gear" : Discrete(5),
            "drift" : Discrete(2),
            "material" : Discrete(7), # None, tech, plastic, dirt, grass, ice, other
            "grounded" : Discrete(2),
        }
        rew_keys = ("top_contact", "race_state", "author_time", "time", "checkpoint", "total_checkpoints", "bonk_time")

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

        self.enabled = enabled
        self.rew_enabled = rew_enabled
        self.frame_shape = frame_shape
        self.square_frame = square_frame

        self.held_obs = {}
        self.held_rew = {}
        self.uns = {} # Unstructured data

    def step(self, action):
        write_actions(self.op_path, action)

        obs, rew_vars = self._handle_observations()

        reward, terminated, truncated, info = self.reward(action, obs, rew_vars)

        return obs, reward, terminated, truncated, info

    def reward(self, action : ActType, obs : dict[str, Any], rew_vars : dict[str, Any]) -> tuple[float, bool, bool, dict]:
        r"""Reward function for agent. Can be overridden for alternate implementations.

        Args:
            action (ActType) : Action passed to agent during :meth:`step`.
        
            obs (dict[str, Any]) : Observations given by TMBaseEnv during :meth:`step`, not supplied by user. Key is str definition of variable.

            rew_vars (dict[str, Any]) : Observations given by TMBaseEnv during :meth:`step` not directly used in training of model, not supplied by user. 
            May still be useful for reward shaping. Key is str definition of variable.
        
        Returns:
            reward (float) : Reward value given to model for given timestep.

            terminated (bool) : Set to True to "reset" enviroment. Call when agent reaches terminal state.

            truncated (bool) : Set to True to "reset" environment for external reasons e.g. :meth:`TimeLimit` wrapper.

            info (dict) : Debug information. Can be returned as empty dict if unused.
        """
        # GOAL: Get agent to finish track as fast as possible

        # Timestep based rewards:

        ts_reward, goal_reward = -.15, 0 # Small negative reward per timestep

        vel_rm, w_rm = 1, 0.05
        ts_reward += obs["velocity"][0] * vel_rm
        ts_reward += action[0] * w_rm

        # TODO: Timestep rewards scale with steps per second
        ts_rew_scale = .5
        ts_reward *= ts_rew_scale

        # Goal based rewards:

        if self.uns["held_checkpoint"] < rew_vars["checkpoint"]:
            self.uns["held_checkpoint"] = rew_vars["checkpoint"]

            intercept = .6 # Reward ratio of first checkpoint to final
            checkpoint_priority = ((rew_vars["checkpoint"] - 1) / (rew_vars["total_checkpoints"] - 1)) * (1 - intercept) + intercept

            checkpoint_rm = 7 # Value of final checkpoint
            goal_reward += checkpoint_priority * checkpoint_rm
        
        if rew_vars["race_state"] == 2: # Finish state
            goal_reward += 15

            # Ideal time - completion time
            time_diff = self.uns["start_time"] + rew_vars["author_time"] - rew_vars["time"]
            time_rm = .8 # Reward per second
            goal_reward += max((time_diff / 1000) * time_rm, -5) # Final reward cannot be too low
            
            terminated = True
        elif rew_vars["top_contact"] == 1:
            goal_reward += -5
            terminated = True
        elif rew_vars["bonk_time"] > self.uns["bonk_time"]:
            self.uns["bonk_time"] = rew_vars["bonk_time"]
            goal_reward += -5
            terminated = True
        else:
            terminated = False

        reward = ts_reward + goal_reward

        # Respawn timer
        if rew_vars["time"] < self.uns["start_time"] :
            reward = 0

        info = {}
        truncated = False
        
        return reward, terminated, truncated, info

    def reset(self):
        obs, rew_vars = self._handle_observations()

        write_alt(self.op_path, reset = True)

        self.uns["start_time"] = rew_vars["time"] + 1600
        self.uns["held_checkpoint"] = 0
        self.uns.setdefault("bonk_time", 0)

        info = {}

        return obs, info
    
    def _handle_observations(self):
        # TODO: Find % of steps being held
        try:
            obs, rew_vars = get_observations(self.op_path, self.enabled, self.rew_enabled)
            self.held_obs, self.held_rew = obs, rew_vars
        except:
            # Triggers if file is being written
            obs, rew_vars = self.held_obs, self.held_rew
        obs = self._handle_frame(obs) # Frame isn't held
        return obs, rew_vars
    
    def _handle_frame(self, obs):
        if self.enabled["frame"]:
            # TODO: RGB observations
            obs["frame"] = get_frame(self.frame_shape[1:], mode = "L", crop = self.square_frame)
        return obs
