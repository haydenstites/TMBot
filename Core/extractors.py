import torch.nn as nn
import gymnasium as gym
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from typing import Any, Dict, Optional

def _NatureCNN(n_input_channels):
    model = nn.Sequential(
        nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.Flatten(),
    )

    return model

class CustomFrameExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False, model : nn.Module = None, model_kwargs : Optional[Dict[str, Any]] = None):
        assert isinstance(observation_space, spaces.Box)
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image)
        super().__init__(observation_space, features_dim)

        model_kwargs = {} if model_kwargs is None else model_kwargs

        n_input_channels = observation_space.shape[0]
        self.model = _NatureCNN(n_input_channels) if model is None else model(n_input_channels, **model_kwargs)

        with torch.no_grad():
            n_flatten = self.model(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        print(f"Initialized {self.model._get_name()} with {n_input_channels} input and {n_flatten} output features.")

    def forward(self, observations):
        return self.linear(self.model(observations))
    
class ExtCombinedExtractor(BaseFeaturesExtractor):
    """Modified :meth:`CombinedExtractor` to be used with :meth:`MultiInputPolicy` allowing for custom frame space models."""
    def __init__(self, observation_space : spaces.Dict, frame_output_dim : int = 256, normalized_image : bool = False, frame_extractor : BaseFeaturesExtractor = CustomFrameExtractor, model : nn.Module = None, model_kwargs : Optional[Dict[str, Any]] = None):
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = frame_extractor(subspace, features_dim=frame_output_dim, normalized_image=normalized_image, model=model, model_kwargs=model_kwargs)
                total_concat_size += frame_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations):
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)

def custom_extractor_policy(model : nn.Module = _NatureCNN, model_kwargs : Optional[Dict[str, Any]] = None, frame_extractor : BaseFeaturesExtractor = CustomFrameExtractor):
    """Wrapper function for creating custom feature extractor for the :var:`frame` observation. Allows observation to be handled by a custom :meth:`nn.Module` object.

    Args:
        model (nn.Module) : Function returning :meth:`nn.Module` object given :int:`n_input_channels`. Used by :meth:`CustomFrameExtractor` by default to extract data from frame observation space.
        May or may not be used by custom implementations. By default returns :meth:`NatureCNN` as defined by SB3.

        model_kwargs (dict[str, Any]) : Keyword arguments passed to model.

        frame_extractor (BaseFeaturesExtractor) : Features extractor extracting data from frame observation space. :meth:`CustomFrameExtractor` by default.

    Returns:
        policy_kwargs (dict[str, Any]) : Keyword arguments for policy.
    """
    model_kwargs = {} if model_kwargs is None else model_kwargs

    policy_kwargs = dict(
        features_extractor_class=ExtCombinedExtractor,
        features_extractor_kwargs=dict(frame_extractor=frame_extractor, model=model, model_kwargs=model_kwargs),
    )

    return policy_kwargs