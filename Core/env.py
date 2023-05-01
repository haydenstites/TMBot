import gymnasium as gym
from base_env import TMBaseEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, DQN, A2C

op_path = r"C:\Users\Lab\OpenplanetNext"
frame_shape = (1, 36, 36) # Channels, height, width
enabled = {
    "frame" : False,
    "velocity" : True,
    "gear" : True,
    "drift" : True,
    "slip" : True,
    "material" : True,
    "grounded" : True,
}
rew_enabled = {
    "top_contact" : True,
    "race_state" : True,
    "checkpoint" : True,
    "total_checkpoints" : True,
    "bonk_time" : True,
    "bonk_score" : True,
}

class TMEnv(TMBaseEnv):
    def __init__(self, op_path: str, frame_shape: tuple[int] = None, enabled: dict[str, bool] = None, rew_enabled: dict[str, bool] = None):
        super().__init__(op_path, frame_shape, enabled, rew_enabled)
        self.bonk_time = 0

    def reward(self, obs, rew_vars):

        reward = obs["velocity"][0] / 20

        if (rew_vars["top_contact"] == 1 or rew_vars["race_state"] == 2):
            terminated = True
        elif (self.bonk_time != rew_vars["bonk_time"]):
            self.bonk_time = rew_vars["bonk_time"]
            terminated = True
        else:
            terminated = False

        info = {}
        truncated = False
        
        return reward, terminated, truncated, info
        
if __name__ == "__main__":
    env = TMEnv(op_path, frame_shape, enabled, rew_enabled)
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
