from pathlib import Path
import pandas as pd

# e.g. active = (True, True, False, etc.)
# Must match to variables from TMData (VelocityY, VelocityX, Gear, etc.)
def read_from_tmdata(op_path, active : tuple):
    data_path = Path(op_path + r"\PluginStorage\TMData\state.txt")
    f = pd.read_csv(data_path, header=None)

    f = f.transpose()

    vars = []
    for i in range(len(active)):
        if active[i]:
            vars.append(f[i].values[0])

    return vars
