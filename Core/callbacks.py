from stable_baselines3.common.callbacks import BaseCallback
from IO import write_alt
from util import get_default_op_path
from pathlib import Path

class TMPauseOnUpdate(BaseCallback):
    r"""Pauses game execution when updating model.

    Args:
        op_path (Path) : Path to Openplanet installation folder. Default is "C:\Users\NAME\OpenplanetNext".
    """
    def __init__(self, op_path : Path = None, verbose=0):
        super().__init__(verbose)
        self.op_path = get_default_op_path() if op_path is None else op_path
        self.first_rollout = True

    def _on_rollout_start(self) -> None:
        if not self.first_rollout:
            print("Rollout starting, unpausing Trackmania.")
            write_alt(self.op_path, pause=True)
        else:
            self.first_rollout = False

    def _on_rollout_end(self):
        print("Rollout ending, pausing Trackmania.")
        write_alt(self.op_path, pause=True)

    def _on_step(self):
        # Required method, return False to abort training
        return True
    
class TMSaveOnEpoch(BaseCallback):
    """Saves model after each training epoch.

    Args:
        name (str) : Name of the model.
    """

    def __init__(self, name : str = None, verbose=0):
        super().__init__(verbose)
        self.name = self.locals["tb_log_name"] if name is None else name
        self.epoch = 0
        self.first_rollout = True

    def _on_rollout_start(self) -> None:
        if not self.first_rollout:
            self.epoch += 1
            self.model.save(f"{self.name}_{self.epoch}")
        else:
            self.first_rollout = False

    def _on_step(self):
        # Required method, return False to abort training
        return True
    
class TMResetOnEpoch(BaseCallback):
    r"""Resets agent after each training epoch.

    Args:
        op_path (Path) : Path to Openplanet installation folder. Default is "C:\Users\NAME\OpenplanetNext".
    """
    def __init__(self, op_path : Path = None, verbose=0):
        super().__init__(verbose)
        self.op_path = get_default_op_path() if op_path is None else op_path
        self.epoch_changed = False

    def _on_rollout_start(self) -> None:
        # Must be buffered to next step to work properly
        self.epoch_changed = True

    def _on_step(self):
        if self.epoch_changed:
            write_alt(self.op_path, reset=True)
            self.epoch_changed = False
        return True
