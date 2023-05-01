from pathlib import Path
import pandas as pd
import numpy as np
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
    data_path = Path(op_path + r"\PluginStorage\TMData\in.txt")

    # 0 : Negative, 1 : None, 2 : Positive
    y_input = action[0] - 1
    x_input = action[1] - 1

    data = pd.DataFrame([y_input, x_input])
    data.to_csv(data_path, index=False, header=False)

def write_reset(op_path):
    data_path = Path(op_path + r"\PluginStorage\TMData\reset.txt")

    data = pd.DataFrame([1])
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
            rew_vars["bonk_score"] = float(vars[0])
            del vars[0]

        return obs, rew_vars

    return obs
