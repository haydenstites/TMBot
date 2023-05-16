import numpy as np
import argparse
from base import TMBaseEnv
from callbacks import TMPauseOnUpdate, TMSaveOnEpochs, TMResetOnEpoch
from extractors import custom_extractor_policy, SimpleCNN
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CallbackList

op_path = r"C:\Users\Lab\OpenplanetNext"
frame_shape = (3, 128, 128) # Channels, height, width

cnn_kwargs = dict( # Increase frame resolution and kernel_size, increase CNN channels 
    layers = [
        dict(out_channels = 64, kernel_size=16, stride=6, padding=0),
        dict(out_channels = 128, kernel_size=4, stride=3, padding=0),
        dict(out_channels = 128, kernel_size=3, stride=1, padding=0),
    ]
)

# frame_shape = (3, 64, 64) # Channels, height, width

# cnn_kwargs = dict( # Increase frame resolution and kernel_size, increase CNN channels 
#     layers = [
#         dict(out_channels = 32, kernel_size=8, stride=4, padding=0),
#         dict(out_channels = 64, kernel_size=4, stride=2, padding=0),
#         dict(out_channels = 64, kernel_size=3, stride=1, padding=0),
#     ]
# )

policy_kwargs = custom_extractor_policy(SimpleCNN, cnn_kwargs)
policy_kwargs["net_arch"] = [512, 512, 256, 256]

ppo_kwargs = dict(
    n_epochs=20,
    n_steps=1024,
    batch_size=256,
    ent_coef = 0.002,
    gae_lambda=0.96,
    #use_sde=True,
    verbose=1,
)

def _setup(model_path : str, n_epochs : int, gui : bool):
    env = TMBaseEnv(frame_shape = frame_shape, gui = gui)
    model = RecurrentPPO("MultiInputLstmPolicy", env, policy_kwargs=policy_kwargs, **ppo_kwargs) # TODO: Hyperparameter optimization
    callbacks = CallbackList([TMPauseOnUpdate(), TMResetOnEpoch(), TMSaveOnEpochs(model_path, n_epochs)])

    return env, model, callbacks

def train_demo(steps : int, model_path : str, n_epochs : int, gui : bool):
    env, model, callbacks = _setup(model_path, n_epochs, gui)

    model.learn(total_timesteps=steps, callback=callbacks, progress_bar=True)

def predict_demo(steps : int, model_path : str, n_epochs : int, gui : bool):
    env, model, callbacks = _setup(model_path, n_epochs, gui)

    model = model.load(model_path, env=env)

    env = model.get_env()
    obs = env.reset()

    lstm_states = None
    num_envs = 1
    episode_starts = np.ones((num_envs,), dtype=bool)

    for step in range(steps):
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True,)
        obs, rewards, dones, info = env.step(action)
        episode_starts = dones

if __name__ == "__main__":
    description = "Run sample implementations of TMBot."
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("-s", "--Steps", nargs="?", type=int, default=1e5, help="Number of train or predict steps")
    parser.add_argument("-p", "--Predict", nargs="?", type=bool, default=False, const=True, help="Run predict loop instead of train loop")
    parser.add_argument("-g", "--GUI", nargs="?", type=bool, default=False, const=True, help="Enable GUI")
    parser.add_argument("-o", "--Output", nargs="?", type=str, default=None, help="Path for saving or loading a model")
    parser.add_argument("-f", "--Frequency", nargs="?", type=int, default=3, help="Frequency of model saves by epoch")

    args = parser.parse_args()

    model_path = "model/TMBot" if args.Output is None else args.Output
    kwargs = dict(steps=args.Steps, model_path=model_path, n_epochs=args.Frequency, gui=args.GUI)

    if args.Predict:
        predict_demo(**kwargs)
    else:
        train_demo(**kwargs)