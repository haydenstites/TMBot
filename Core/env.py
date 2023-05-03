import gymnasium as gym
from base import TMBaseEnv, TMPauseOnUpdate
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

op_path = r"C:\Users\Lab\OpenplanetNext"
frame_shape = (1, 36, 36) # Channels, height, width
enabled = {
    "frame" : False,
    "velocity" : True,
    "gear" : True,
    "drift" : True,
    "material" : True,
    "grounded" : True,
}
rew_enabled = {
    "top_contact" : True,
    "race_state" : True,
    "checkpoint" : True,
    "total_checkpoints" : True,
    "bonk_time" : True,
}

# TODO: Wrapper train function with GUI and command line implementations
if __name__ == "__main__":
    env = TMBaseEnv(op_path, frame_shape, enabled, rew_enabled)
    model = PPO("MultiInputPolicy", env, verbose=1) # TODO: Hyperparameter optimization
    model.learn(total_timesteps=25000, callback=TMPauseOnUpdate(op_path))
