from tmbot.core import TMBaseEnv
from tmbot.extras.callbacks import default_callbacks
from tmbot.extras.extractors import custom_extractor_policy, SimpleCNN
from sb3_contrib import RecurrentPPO

map_urls= (
    "https://github.com/Hayden-Stites/testmaps/raw/master/Train1.Map.Gbx",
    "https://github.com/Hayden-Stites/testmaps/raw/master/Train2.Map.Gbx",
    "https://github.com/Hayden-Stites/testmaps/raw/master/Train3.Map.Gbx",
    "https://github.com/Hayden-Stites/testmaps/raw/master/Train4.Map.Gbx",
    "https://github.com/Hayden-Stites/testmaps/raw/master/Train5.Map.Gbx",
)
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

env = TMBaseEnv(map_urls=map_urls, frame_shape = frame_shape, gui = True)
model = RecurrentPPO("MultiInputLstmPolicy", env, policy_kwargs=policy_kwargs, **ppo_kwargs) # TODO: Hyperparameter optimization

callbacks = default_callbacks()

model.learn(total_timesteps=3e5, callback=callbacks, progress_bar=True)
