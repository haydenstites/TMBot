from base import TMBaseEnv
from callbacks import TMPauseOnUpdate, TMSaveOnEpoch, TMResetOnEpoch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList

op_path = r"C:\Users\Lab\OpenplanetNext"
frame_shape = (1, 100, 100) # Channels, height, width

enabled = {
    "frame" : True,
}

# TODO: Wrapper train function with GUI and command line implementations
if __name__ == "__main__":
    env = TMBaseEnv(op_path, frame_shape, enabled, square_frame=True)
    model = PPO("MultiInputPolicy", env, verbose=1, n_steps=2048, n_epochs=40) # TODO: Hyperparameter optimization
    callbacks = CallbackList([TMPauseOnUpdate(op_path), TMResetOnEpoch(op_path), TMSaveOnEpoch("model/TMTest")])
    model.learn(total_timesteps=50000, callback=callbacks)
