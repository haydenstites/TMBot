import gymnasium as gym
import numpy as np
import time
from .IO import init_tmdata, set_maps, TMDataBuffer
from .util import get_frame, get_default_op_path, linear_interp
from .gui import TMGUI
from ..external.midas import TMMidas
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from gymnasium.core import ActType
from typing import Any
from pathlib import Path

class TMBaseEnv(gym.Env):
    """Trackmania Base Environment, :meth:`reward` function can be overridden for custom implementations."""
    def __init__(self,
            map_urls : tuple[str] = None,
            op_path : Path = None,
            frame_shape : tuple[int, int, int] = None,
            enabled : dict[str, bool] = None,
            rew_enabled : dict[str, bool] = None,
            sc_algorithm : str = "imagegrab",
            square_frame : bool = True,
            fps_max : int = 20,
            midas_model : str = None,
            gui : bool = False,
            gui_kwargs : dict[str, Any] = None
        ):
        r"""Initialization parameters for TMBaseEnv. Parameters in enabled and rew_enabled should match output variables in TMData.

        Args:
            map_urls (tuple[str]) : URLs to Trackmania map files. If None, maps will not switch during training. Default is None.

            op_path (Path) : Path to Openplanet installation folder. Default is "C:/Users/NAME/OpenplanetNext".

            frame_shape (tuple[int, int, int]) : Observation size of image frame passed to CNN, formatted (channels, height, width).
                Must be at least (1, 36, 36). Default is (1, 50, 50).

            enabled (dict[str, bool]) : Dictionary describing enabled parameters in observation space. Default is True for every key.

            rew_enabled (dict[str, bool]) : Dictionary describing enabled parameters for reward shaping. Default is True for every key.

            sc_algorithm (str) : Whether to use "imagegrab" or "win32" screen capture algorithm. "win32" is faster
                but requires the game to be windowed. Default is "imagegrab" for ease of use.

            square_frame (bool) : Crops observed frames to squares. Default is True.

            fps_max (int) : Maximum allowed steps per second. Step rewards scale accordingly. Default is 20.

            midas_model (str) : Path to pretrained MiDaS model, if used. Default is None.

            gui (bool) : Enables a tkinter GUI for viewing training information. Default is False.

            gui_kwargs (dict[str, Any]) : Keyword arguments for :class:`TMGUI`.
        """
        self.op_path = get_default_op_path() if op_path is None else op_path
        init_tmdata(self.op_path)
        set_maps(self.op_path, map_urls)

        min_frame_shape = (1, 36, 36)
        default_frame_shape = (1, 50, 50)
        frame_shape = default_frame_shape if frame_shape is None else frame_shape
        assert frame_shape >= min_frame_shape, f"frame_shape {frame_shape} is less than min_frame_shape {min_frame_shape}"
        assert frame_shape[0] in (1, 3), f"frame_shape[0] must either be equal to 1 for grayscale observations or equal to 3 for RGB observations"

        if midas_model is not None:
            self.midas = TMMidas(midas_model)
            frame_shape = (frame_shape[0] + 1, *frame_shape[1:])
        else:
            self.midas = None

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

        self.frame_shape = frame_shape
        self.enabled = enabled
        self.rew_enabled = rew_enabled
        self.sc_algorithm = sc_algorithm
        self.square_frame = square_frame
        self.fps_max = fps_max
        self.gui = gui

        self.uns = {} # Unstructured data

        self.buffer = TMDataBuffer(self.op_path)

        gui_kwargs = {} if gui_kwargs is None else gui_kwargs
        if self.gui:
            self.window = TMGUI(enabled=self.enabled, rew_enabled=self.rew_enabled, env=self, **gui_kwargs)

    def step(self, action):
        self.buffer.write_actions(action)

        obs, rew_vars = self._handle_observations()

        # Limit steps per second
        self.uns.setdefault("prev_time", 0)
        true_ms = rew_vars["time"] - self.uns["prev_time"]
        min_ms = 1000 / self.fps_max
        if true_ms < min_ms:
            diff = (min_ms - true_ms) / 1000
            time.sleep(diff)
        self.uns["prev_time"] = rew_vars["time"]

        ts_reward, goal_reward, terminated, truncated, info = self.reward(action, obs, rew_vars)
        reward = ts_reward + goal_reward

        if self.gui:
            self.window.update(obs, rew_vars, ts_reward, goal_reward)

        return obs, reward, terminated, truncated, info

    def reward(self, action : ActType, obs : dict[str, Any], rew_vars : dict[str, Any]) -> tuple[float, bool, bool, dict]:
        r"""Reward function for agent. Can be overridden for alternate implementations.

        Args:
            action (ActType) : Action passed to agent during :meth:`step`.
        
            obs (dict[str, Any]) : Observations given by :class:`TMBaseEnv` during :meth:`step`, not supplied by user. Key is str definition of variable.

            rew_vars (dict[str, Any]) : Extra observations given by :class:`TMBaseEnv` during :meth:`step` not directly used in training of model, not supplied by user.
                May still be useful for reward shaping. Key is str definition of variable.
        
        Returns:
            reward (float) : Reward value given to model for given timestep.

            terminated (bool) : Set to True to "reset" enviroment. Call when agent reaches terminal state.

            truncated (bool) : Set to True to "reset" environment for external reasons e.g. `TimeLimit` wrapper.

            info (dict) : Debug information. Can be returned as empty dict if unused.
        """
        # GOAL: Get agent to finish track as fast as possible

        # Timestep based rewards:

        ts_reward, goal_reward = -.2, 0 # Small negative reward per timestep

        vel_rm, w_rm = 1.3, 0.06
        ts_reward += obs["velocity"][0] * vel_rm
        ts_reward += action[0] * w_rm

        ts_rew_scale = 9 / self.fps_max

        ts_reward *= ts_rew_scale

        # Goal based rewards:

        terminated = False

        vel_threshold = .05
        threshold_steps = 300
        if obs["velocity"][0] < vel_threshold:
            self.uns.setdefault("under_threshold", 0)
            self.uns["under_threshold"] += 1
            if self.uns["under_threshold"] >= threshold_steps:
                goal_reward += -5
                terminated = True
        else:
            self.uns["under_threshold"] = 0

        if self.uns["held_checkpoint"] < rew_vars["checkpoint"]:
            self.uns["held_checkpoint"] = rew_vars["checkpoint"]

            intercept = .6 # Reward ratio of first checkpoint to final
            checkpoint_priority = linear_interp(value=rew_vars["checkpoint"], end=rew_vars["total_checkpoints"], intercept=intercept)

            checkpoint_rmax = 6 # Value of final checkpoint
            checkpoint_br = checkpoint_priority * checkpoint_rmax

            checkpoint_vmax = 8 # Multiplier velocity reward
            checkpoint_vr = min(obs["velocity"][0] * checkpoint_vmax, 3)

            goal_reward += checkpoint_br + checkpoint_vr

        if rew_vars["race_state"] == 2: # Finish state
            goal_reward += 30

            # Ideal time - completion time
            time_diff = self.uns["start_time"] + rew_vars["author_time"] - rew_vars["time"]
            time_rm = .8 # Reward per second
            goal_reward += max((time_diff / 1000) * time_rm, -15) # Final reward cannot be too low
            
            terminated = True
        else: # Bonk and flip don't matter if finished
            if rew_vars["top_contact"] == 1:
                goal_reward += -3
                terminated = True
            if rew_vars["bonk_time"] > self.uns["bonk_time"]:
                self.uns["bonk_time"] = rew_vars["bonk_time"]
                goal_reward += -4
                terminated = True
            

        # Respawn timer
        if rew_vars["time"] < self.uns["start_time"] :
            sleep_time = (self.uns["start_time"] - rew_vars["time"]) / 1000
            print(f"Sleeping {sleep_time} seconds until respawn...")
            time.sleep(sleep_time)
            ts_reward, goal_reward = 0, 0

        info = {}
        truncated = False
        
        return ts_reward, goal_reward, terminated, truncated, info

    def reset(self):
        obs, rew_vars = self._handle_observations()

        self.buffer.write_alt(reset = True)

        self.uns["start_time"] = rew_vars["time"] + 1600
        self.uns["held_checkpoint"] = 0
        self.uns["under_threshold"] = 0
        self.uns.setdefault("bonk_time", 0)

        if self.gui:
            self.window.flush_buffers()

        info = {}

        return obs, info
    
    def _handle_observations(self):
        obs, rew = self.buffer.get_observations(self.enabled, self.rew_enabled)
        if self.enabled["frame"]:
            mode = "L" if self.frame_shape[0] == 1 else "RGB"
            obs["frame"] = get_frame(self.frame_shape[1:], mode = mode, crop = self.square_frame, algorithm = self.sc_algorithm)

            if self.midas is not None:
                shape = (1, *self.frame_shape[1:])
                depth = self.midas.step(obs["frame"]).reshape(shape)
                
                obs["frame"] = np.append(obs["frame"], depth, axis=0)

        return obs, rew
