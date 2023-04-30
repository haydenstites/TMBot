from pathlib import Path
import pandas as pd
import numpy as np
import gymnasium as gym
from util import norm_float, binary_strbool, mat_index, race_index

# Get all vars from TMData
def read_from_tmdata(op_path) -> list:
    data_path = Path(op_path + r"\PluginStorage\TMData\state.txt")
    f = pd.read_csv(data_path, header=None)

    vars = []
    for i in range(len(f)):
        vars.append(f.transpose()[i].values[0])

    return vars

# Write actions to TMData
def write_actions(op_path, action : np.ndarray):
    # 0 : None, 1 : Positive, 2 : Negative
    y_input = action[0]
    x_input = action[1]
    
    data_path = Path(op_path + r"\PluginStorage\TMData\in.txt")
    data = pd.DataFrame([y_input, x_input])

    data.to_csv(data_path, index=False, header=False)

# Read observations from TMData
def get_observations(op_path, enabled, rew_enabled = None):
    vars = read_from_tmdata(op_path)
    obs = {}

    if enabled["velocity"]:
        obs["velocity"] = np.array([norm_float(vars[0], -1000, 1000), norm_float(vars[1], -1000, 1000)], dtype=np.float32)
        del vars[0:2]
    if enabled["gear"]:
        obs["gear"] = int(vars[0])
        del vars[0]
    if enabled["drift"]:
        obs["drift"] = binary_strbool(vars[0])
        del vars[0]
    if enabled["slip"]:
        obs["slip"] = np.array([norm_float(vars[0], 0, 1)], dtype=np.float32)
        del vars[0]
    if enabled["material"]:
        obs["material"] = mat_index(vars[0])
        del vars[0]
    if enabled["grounded"]:
        obs["grounded"] = binary_strbool(vars[0])
        del vars[0]

    if rew_enabled:
        rew_vars = {}

        if rew_enabled["top_contact"]:
            rew_vars["top_contact"] = binary_strbool(vars[0])
            del vars[0]
        if rew_enabled["race_state"]:
            rew_vars["race_state"] = race_index(vars[0])
            del vars[0]
        if rew_enabled["checkpoint"]:
            rew_vars["checkpoint"] = int(vars[0])
            del vars[0]
        if rew_enabled["total_checkpoints"]:
            rew_vars["total_checkpoints"] = int(vars[0])
            del vars[0]
        if rew_enabled["bonk_time"]:
            rew_vars["bonk_time"] = int(vars[0])
            del vars[0]
        if rew_enabled["bonk_score"]:
            rew_vars["bonk_score"] = int(vars[0])
            del vars[0]

        return obs, rew_vars

    return obs
