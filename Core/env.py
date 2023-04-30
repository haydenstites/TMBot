import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary
import numpy as np
from IO import write_actions, get_observations

# e.g. enabled = {
#     "frame" : True,
#     "velocity" : False,
#     "slip" : True,
# }
# e.g. frame_shape = (1, 32, 32) # Channels, height, width
class TMBaseEnv(gym.Env):
    """Trackmania Base Environment, :meth:`step` function can be overridden for custom implementations."""
    def __init__(self, op_path : str, frame_shape : tuple[int] = None, enabled : dict[str, bool] = None, rew_enabled : dict[str, bool] = None):
        """Initialization parameters for TMBaseEnv. Parameters in enabled and rew_enabled should match output variables in TMData.

        Args:
            op_path (str) : Path to Openplanet installation folder. Typically "C:\\Users\\yournamehere\\OpenplanetNext"

            frame_shape (tuple[int], Optional) : Observation size of image frame passed to CNN, formatted (channels, height, width).
            Default is (1, 36, 36), which is also the elementwise minimum frame size.

            enabled (dict[str, bool], Optional) : Dictionary describing enabled parameters in observation space. Default is True for every key.

            rew_enabled (dict[str, bool], Optional) : Dictionary describing enabled parameters for reward shaping. Default is True for every key.
        """

        frame_shape = (1, 36, 36) if frame_shape is None else frame_shape

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

        self.action_space = MultiDiscrete([3, 3], dtype=np.int32)
        self.observation_space = gym.spaces.Dict(obs_vars)

        self.op_path = op_path
        self.enabled = enabled
        self.rew_enabled = rew_enabled

        self.write_actions = write_actions
        self.get_observations = get_observations

    def step(self, action):
        raise NotImplementedError

        # return obs, reward, terminated, truncated, info

    def reset(self):
        obs = self.get_observations(self.op_path, self.enabled)

        # Write reset key to TMData

        info = {}

        return obs, info
