import torch.nn as nn
import gymnasium as gym
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from typing import Any, Dict, Optional

# TODO: Doc extrators

class SimpleCNN(nn.Module):
    def __init__(self, n_input_channels : int, layers : list[dict], activation : nn.Module = nn.ReLU()):
        """Simple CNN architecture with customizable attributes. Not intended to be invoked directly, done by :class:`ExtCombinedExtractor`.
        
        Args:
            n_input_channels (int) : Number of input channels.

            layers (list[dict]) : A list where each element corresponds to `nn.Conv2d` keyword arguments.

            activation (nn.Module) : Activation function between each convolutional layer.
        """
        super().__init__()
        self.model = nn.Sequential()

        for layer in layers:
            self.model.append(nn.Conv2d(in_channels=n_input_channels, **layer))
            self.model.append(activation)

            n_input_channels = layer["out_channels"]

        self.model.append(nn.Flatten())

    def forward(self, x):
        return self.model(x)

_NatureCNN_kwargs = dict(
    layers = [
        dict(out_channels = 32, kernel_size=8, stride=4, padding=0),
        dict(out_channels = 64, kernel_size=4, stride=2, padding=0),
        dict(out_channels = 64, kernel_size=3, stride=1, padding=0),
    ]
)

class CustomFrameExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False, model : nn.Module = None, model_kwargs : Optional[Dict[str, Any]] = None):
        assert isinstance(observation_space, spaces.Box)
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image)
        super().__init__(observation_space, features_dim)

        model_kwargs = _NatureCNN_kwargs if model_kwargs is None else model_kwargs
        model = SimpleCNN if model is None else model

        n_input_channels = observation_space.shape[0]
        self.model = model(n_input_channels, **model_kwargs)

        with torch.no_grad():
            n_flatten = self.model(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        print(f"Initialized {self.model._get_name()} with {n_input_channels} input and {n_flatten} output features, connected to {features_dim} output features.")

    def forward(self, observations):
        return self.linear(self.model(observations))
    
class ExtCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space : spaces.Dict, frame_output_dim : int = 256, normalized_image : bool = False, frame_extractor : BaseFeaturesExtractor = CustomFrameExtractor, model : nn.Module = None, model_kwargs : Optional[Dict[str, Any]] = None):
        """Modified :meth:`CombinedExtractor` to be used with `MultiInputPolicy` allowing for custom frame space models.

        Args:
            observation_space (spaces.Dict) : Dictionary observation space to be used.

            frame_output_dim (int) : Output features of frame feature extractor. Default is 256.

            normalized_image (bool) : Whether image is already normalized. Default is False.

            frame_extractor (BaseFeaturesExtractor) : Feature extractor for frame observation. Default is CustomFrameExtractor.

            model (nn.Module) : Model for frame feature extraction. A CNN is recommended. Default is `NatureCNN`.

            model_kwargs (dict[str, Any]) : Keyword arguments for frame extractor model. Default is None.
        """
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

def custom_extractor_policy(model : nn.Module = SimpleCNN, model_kwargs : Optional[Dict[str, Any]] = None, frame_extractor : BaseFeaturesExtractor = CustomFrameExtractor, frame_output_dim = 512):
    """Wrapper function for creating custom feature extractor for the frame observation. Allows observation to be handled by a custom `nn.Module` object.

    Args:
        model (nn.Module) : Function returning `nn.Module` object given `n_input_channels`. Used by :class:`CustomFrameExtractor` by default to extract data from frame observation space.
            May or may not be used by custom implementations. By default returns `NatureCNN` as defined by SB3.

        model_kwargs (dict[str, Any]) : Keyword arguments passed to model.

        frame_extractor (BaseFeaturesExtractor) : Features extractor extracting data from frame observation space. :class:`CustomFrameExtractor` by default.

    Returns:
        policy_kwargs (dict[str, Any]) : Keyword arguments for policy.
    """
    model_kwargs = {} if model_kwargs is None else model_kwargs

    policy_kwargs = dict(
        features_extractor_class=ExtCombinedExtractor,
        features_extractor_kwargs=dict(frame_extractor=frame_extractor, model=model, model_kwargs=model_kwargs, frame_output_dim=frame_output_dim),
    )

    return policy_kwargs