import gymnasium as gym
from env import TMBaseEnv
from stable_baselines3.common.env_checker import check_env

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
    def _test(self):
        obs, rew_vars = self.get_observations(self.op_path, self.enabled, self.rew_enabled)

        return obs, rew_vars

    def step(self, action):
        self.write_actions(self.op_path, action)

        obs, rew_vars = self.get_observations(self.op_path, self.enabled, self.rew_enabled)

        reward = obs["velocity"][0] / 50

        if (rew_vars["top_contact"] == 1 or rew_vars["race_state"] == 2):
            terminated = True
        else:
            terminated = False

        info = {}
        truncated = False
        
        return obs, reward, terminated, truncated, info

env = TMEnv(op_path, frame_shape, enabled, rew_enabled)

check_env(env)