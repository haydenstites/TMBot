from pathlib import Path
import pandas as pd

def read_from_tmdata(op_path) -> list:
    data_path = Path(op_path + r"\PluginStorage\TMData\state.txt")
    f = pd.read_csv(data_path, header=None)

    vars = []
    for i in range(len(f)):
        vars.append(f.transpose()[i].values[0])

    return vars
