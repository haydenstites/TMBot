# TODO: HER? Frame stacking? Reccurrent? Architectures?

from base import TMBaseEnv
from callbacks import TMPauseOnUpdate, TMSaveOnEpoch, TMResetOnEpoch
from extractors import custom_extractor_policy
from models import Resnet
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList

op_path = r"C:\Users\Lab\OpenplanetNext"
frame_shape = (1, 50, 50) # Channels, height, width

model_kwargs = dict(conv_channels=[32, 64])
policy_kwargs = custom_extractor_policy(Resnet, model_kwargs)
policy_kwargs["net_arch"] = [256] * 4

# Pygame/tkinter window to see observations in realtime

# RGB observations
# 5 frames stacked
# simple CNN, 3 layers, (256, 512, 512)
# HER

# No steps on reset
# Max framerate (sleep between writing action and getting observation)
# Divide timestep reward by framerate (max framerate?)

# TODO: Wrapper train function with GUI and command line implementations
if __name__ == "__main__":
    env = TMBaseEnv(frame_shape = frame_shape)
    model = PPO("MultiInputPolicy", env, verbose=1, n_steps=2048, n_epochs=50, policy_kwargs=policy_kwargs) # TODO: Hyperparameter optimization
    callbacks = CallbackList([TMPauseOnUpdate(), TMResetOnEpoch(), TMSaveOnEpoch("model/TMTest")])
    model.learn(total_timesteps=50000, callback=callbacks, progress_bar=True)
