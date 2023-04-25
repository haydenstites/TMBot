import gymnasium as gym
from env import TMEnv

# env = TMEnv()

op_path = r"C:\Users\Lab\OpenplanetNext" # Supplied by user elsewhere
elements = (True, True, True)
env = TMEnv(op_path, elements)
