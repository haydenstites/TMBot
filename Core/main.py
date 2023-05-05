from base import TMBaseEnv, TMPauseOnUpdate
from stable_baselines3 import PPO

op_path = r"C:\Users\Lab\OpenplanetNext"
frame_shape = (1, 100, 100) # Channels, height, width

# TODO: Wrapper train function with GUI and command line implementations
if __name__ == "__main__":
    env = TMBaseEnv(op_path, frame_shape, square_frame=True)
    model = PPO("MultiInputPolicy", env, verbose=1, n_steps=2048, n_epochs=20) # TODO: Hyperparameter optimization
    model.learn(total_timesteps=25000, callback=TMPauseOnUpdate(op_path))
