from tmbot.core import TMBaseEnv
from tmbot.extras.callbacks import default_callbacks
from tmbot.extras.extractors import custom_extractor_policy, SimpleCNN
from sb3_contrib import RecurrentPPO

frame_shape = (3, 128, 128) # Channels, height, width

cnn_kwargs = dict(
    layers = [
        dict(out_channels = 64, kernel_size=12, stride=6, padding=0),
        dict(out_channels = 128, kernel_size=5, stride=3, padding=0),
        dict(out_channels = 128, kernel_size=3, stride=1, padding=0),
    ]
)

policy_kwargs = custom_extractor_policy(SimpleCNN, cnn_kwargs)
policy_kwargs["net_arch"] = [512, 512, 256, 256]

ppo_kwargs = dict(
    n_epochs=20,
    n_steps=1024,
    batch_size=256,
    ent_coef = 0.002,
    gae_lambda=0.94, # Low on main training, higher on map_specific training
    verbose=1,
)

env = TMBaseEnv(frame_shape = frame_shape, gui = True)
model = RecurrentPPO("MultiInputLstmPolicy", env, policy_kwargs=policy_kwargs, **ppo_kwargs) # TODO: Hyperparameter optimization

model = model.load("TMBot_60", env=env)

callbacks = default_callbacks()

model.learn(total_timesteps=3e5, callback=callbacks, progress_bar=True)
