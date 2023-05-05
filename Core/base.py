import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from gymnasium.core import ActType
from stable_baselines3.common.callbacks import BaseCallback
from IO import get_observations, write_actions, write_alt
from util import get_frame
from typing import Any

class TMBaseEnv(gym.Env):
    """Trackmania Base Environment, :meth:`reward` function must be overridden for custom implementations."""
    def __init__(self, op_path : str, frame_shape : tuple[int] = None, enabled : dict[str, bool] = None, rew_enabled : dict[str, bool] = None, square_frame : bool = True):
        r"""Initialization parameters for TMBaseEnv. Parameters in enabled and rew_enabled should match output variables in TMData.

        Args:
            op_path (str) : Path to Openplanet installation folder. Typically "C:\Users\NAME\OpenplanetNext"

            frame_shape (tuple[int]) : Observation size of image frame passed to CNN, formatted (channels, height, width).
            Must be at least (1, 36, 36). Default is (1, 50, 50).

            enabled (dict[str, bool]) : Dictionary describing enabled parameters in observation space. Default is True for every key.

            rew_enabled (dict[str, bool]) : Dictionary describing enabled parameters for reward shaping. Default is True for every key.

            square_frame (bool) : Crops observed frames to squares. Default is True.
        """

        min_frame_shape = (1, 36, 36)
        default_frame_shape = (1, 50, 50)
        frame_shape = default_frame_shape if frame_shape is None else frame_shape
        assert frame_shape >= min_frame_shape, f"frame_shape {frame_shape} is less than min_frame_shape {min_frame_shape}"

        # Discrete(2) converted from bool, TODO: More elegant solution?
        obs_vars = {
            "frame" : Box(low=0, high=255, shape=frame_shape, dtype=np.uint8), # Image specs of SB3
            "velocity" : Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32), # Normed from -1000 to 1000
            "gear" : Discrete(5),
            "drift" : Discrete(2),
            "material" : Discrete(7), # None, tech, plastic, dirt, grass, ice, other
            "grounded" : Discrete(2),
        }
        rew_keys = ("top_contact", "race_state", "checkpoint", "total_checkpoints", "bonk_time")

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
        self.frame_shape = frame_shape
        self.square_frame = square_frame

        self.held_obs = {}
        self.held_rew = {}
        self.uns = {} # Unstructured data

    def step(self, action):
        write_actions(self.op_path, action)

        obs, rew_vars = self._handle_observations()

        reward, terminated, truncated, info = self.reward(action, obs, rew_vars)

        # TODO: Zero reward when respawning

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

        # TODO: Better reward function

        self.uns.setdefault("bonk_time", 0)

        reward = obs["velocity"][0] + action[0] / 10

        if (rew_vars["race_state"] == 2):
            reward = 10
            terminated = True
        elif (rew_vars["top_contact"] == 1):
            reward = -5
            terminated = True
        elif (self.uns["bonk_time"] != rew_vars["bonk_time"]):
            self.uns["bonk_time"] = rew_vars["bonk_time"]
            reward = -3
            terminated = True
        else:
            terminated = False

        info = {}
        truncated = False
        
        return reward, terminated, truncated, info

    def reset(self):
        obs, _ = self._handle_observations()

        write_alt(self.op_path, reset = True)

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
    
class TMPauseOnUpdate(BaseCallback):
    """Pauses game execution when updating model.
    """
    def __init__(self, op_path, verbose=0):
        super().__init__(verbose)
        self.op_path = op_path
        self.first_rollout = True

    def _on_rollout_start(self) -> None:
        if (not self.first_rollout):
            print("Rollout starting, unpausing Trackmania.")
            write_alt(self.op_path, pause=True)
        else:
            self.first_rollout = False

    def _on_rollout_end(self):
        print("Rollout ending, pausing Trackmania.")
        write_alt(self.op_path, pause=True)

    def _on_step(self):
        # Required method, return False to abort training
        return True
