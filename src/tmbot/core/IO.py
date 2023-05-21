import pandas as pd
import numpy as np
import os
import wget
from .util import norm_float, binary_strbool, mat_index, race_index, get_default_op_path
from pathlib import Path

class TMDataBuffer():
    def __init__(self):
        self.buffer = {}
        self.uns = {}

    def write_actions(self, op_path, action):
        if self.buffer["alt"] is not None:
            write_alt(op_path, **self.buffer["alt"])

        write_actions(op_path, action)

    def write_alt(self, op_path, reset : bool = None, pause : bool = None):
        try:
            write_alt(op_path, reset, pause)
            self.buffer["alt"] = None
        except:
            self.buffer["alt"] = dict(
                reset = reset,
                pause = pause,
            )

    def get_observations(self, op_path, enabled : dict, rew_enabled : dict = None):
        try:
            self.buffer["obs"] = get_observations(op_path, enabled, rew_enabled)
        except:
            # Triggers if file is being written
            self.uns.setdefault("steps_held", 0)
            self.uns["steps_held"] += 1
        
        self.uns.setdefault("total_steps", 0)
        self.uns["total_steps"] += 1

        return self.buffer["obs"]

def init_tmdata(op_path = None):
    r"""Checks if all TMData files are present, and downloads them if any aren't.

    Args:
        op_path (Path) : Path to Openplanet installation folder. Default is "C:/Users/NAME/OpenplanetNext".
    """
    print("Setting up TMData...")
    init = False

    op_path = get_default_op_path() if op_path is None else op_path

    # Folder
    data_path = Path(op_path, "PluginStorage/TMData")
    if not os.path.exists(data_path):
        print(f"Path {data_path} does not exist, creating...")
        os.mkdir(data_path)
        init = True

    # Files
    for file in ["in.txt", "alt.txt", "state.txt"]:
        file_path = Path(data_path, file)
        if not os.path.exists(file_path):
            if file == "in.txt":
                data = pd.DataFrame([0, 0])
                data.to_csv(file_path, index=False, header=False)
            else:
                print(f"File {file_path} does not exist, TMData must be run before using TMBot.")
                init = True

    # Plugin
    plugin_path = Path(op_path, "Plugins/TMData.op")
    if not os.path.exists(plugin_path):
        print(f"File {plugin_path} does not exist, downloading...")

        url = "https://github.com/Hayden-Stites/TMBot/raw/master/data/TMData.op"
        file = wget.download(url, out=str(plugin_path))

        print("Downloaded TMData.op")
        init = True
    
    # Dll
    dll_path = Path(data_path, "TMDataInputSys.dll")
    if not os.path.exists(dll_path):
        print(f"File {dll_path} does not exist, downloading...")

        url = "https://github.com/Hayden-Stites/TMBot/raw/master/data/TMDataInputSys/x64/Debug/TMDataInputSys.dll"
        file = wget.download(url, out=str(dll_path))

        print("Downloaded TMDataInputSys.dll")
        init = True


    assert not init, (
    f"""TMData and all necessary files are downloaded to OpenPlanet path {op_path}.
    Trackmania must be run with the TMData plugin before TMBot will function properly."""
    )

# Write actions to TMData
def write_actions(op_path, action):
    r"""Writes actions to TMData.

    Args:
        op_path (Path) : Path to Openplanet installation folder. Default is "C:/Users/NAME/OpenplanetNext".

        action (ActType) : Action to write to TMData.
    """
    data_path = Path(op_path, r"PluginStorage\TMData\in.txt")

    # 0 : Negative, 1 : None, 2 : Positive
    y_input = action[0] - 1
    x_input = action[1] - 1

    data = pd.DataFrame([y_input, x_input])
    data.to_csv(data_path, index=False, header=False)

# Write reset/pause to TMData
def write_alt(op_path, reset : bool = None, pause : bool = None):
    r"""Writes data to TMData exchange file.

    Args:
        op_path (Path) : Path to Openplanet installation folder. Default is "C:/Users/NAME/OpenplanetNext".

        reset (bool) : Buffers a reset to TMData.

        pause (bool) : Buffers a pause to TMData.
    """
    data_path = Path(op_path, r"PluginStorage\TMData\alt.txt")
    assert Path.exists(data_path), f"Path {data_path} does not exist."

    f = pd.read_csv(data_path, header=None)

    if reset:
        data = pd.DataFrame([1, f.transpose()[1].values[0]])
    elif pause:
        data = pd.DataFrame([f.transpose()[0].values[0], 1])

    data.to_csv(data_path, index=False, header=False)

# Read observations from TMData
def get_observations(op_path, enabled : dict, rew_enabled : dict = None):
    r"""Grabs observations from TMData.

    Args:
        op_path (Path) : Path to Openplanet installation folder. Default is "C:/Users/NAME/OpenplanetNext".

        enabled (dict[str, bool]) : Dictionary describing enabled parameters in observation space.

        rew_enabled (dict[str, bool]) : Dictionary describing enabled parameters for reward shaping.
    Returns:
        obs (dict[str, Any]) : Observations in dict format.

        rew_vars (dict[str, Any]) : Extra observations in dict format. Can be used for reward shaping.
    """
    data_path = Path(op_path, r"PluginStorage\TMData\state.txt")
    assert Path.exists(data_path), f"Path {data_path} does not exist."

    f = pd.read_csv(data_path, header=None)

    vars = []
    for i in range(len(f)):
        vars.append(f.transpose()[i].values[0])

    obs = {}

    if enabled["velocity"]:
        obs["velocity"] = np.array([norm_float(vars[0], -1000, 1000), norm_float(vars[1], -1000, 1000)], dtype=np.float32)
    if enabled["gear"]:
        obs["gear"] = int(vars[2])
    if enabled["drift"]:
        obs["drift"] = binary_strbool(vars[3])
    if enabled["material"]:
        obs["material"] = mat_index(vars[4])
    if enabled["grounded"]:
        obs["grounded"] = binary_strbool(vars[5])

    del vars[0:6]

    if rew_enabled:
        rew_vars = {}

        if rew_enabled["top_contact"]:
            rew_vars["top_contact"] = binary_strbool(vars[0])
            del vars[0]
        if rew_enabled["race_state"]:
            rew_vars["race_state"] = race_index(vars[0])
            del vars[0]
        if rew_enabled["author_time"]:
            rew_vars["author_time"] = int(vars[0])
            del vars[0]
        if rew_enabled["time"]:
            rew_vars["time"] = int(vars[0])
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

        return obs, rew_vars

    return obs
