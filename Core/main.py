import gymnasium as gym
from env import TMEnv
from util import norm_float

op_path = r"C:\Users\Lab\OpenplanetNext" # Supplied by user elsewhere
enabled = {
    "frame" : True,
    "velocity" : True,
    "gear" : True,
    "drift" : True,
    "slip" : True,
    "material" : True,
    "grounded" : True,
}
frame_shape = (1, 32, 32) # Channels, height, width
env = TMEnv(op_path, frame_shape, enabled)
env._get_observations()
