from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from ..core.IO import write_alt
from ..core.util import get_default_op_path
from pathlib import Path

class TMPauseOnUpdate(BaseCallback):
    def __init__(self, op_path : Path = None, verbose=0):
        r"""Pauses game execution when updating model.

        Args:
            op_path (Path) : Path to Openplanet installation folder. Default is "C:/Users/NAME/OpenplanetNext".
        """
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
    
class TMSaveOnEpochs(BaseCallback):
    def __init__(self, name : str = None, n_epochs : int = 1, verbose=0):
        """Saves model after every n training epochs.

        Args:
            name (str) : Name of the model.
        
            n_epochs (int) : Frequency of model saving. Default is 1 (every epoch).
        """
        super().__init__(verbose)
        self.name = self.locals["tb_log_name"] if name is None else name
        self.epoch_name = 0
        self.first_rollout = True

        self.n_epochs = n_epochs
        self.n = 0

    def _on_rollout_start(self) -> None:
        if not self.first_rollout:
            self.epoch_name += 1
            self.n += 1   

            if self.n >= self.n_epochs:
                self.model.save(f"{self.name}_{self.epoch_name}")

                self.n = 0
        else:
            self.first_rollout = False

    def _on_step(self):
        # Required method, return False to abort training
        return True
    
class TMResetOnEpoch(BaseCallback):
    def __init__(self, op_path : Path = None, verbose=0):
        r"""Resets agent after each training epoch.

        Args:
            op_path (Path) : Path to Openplanet installation folder. Default is "C:/Users/NAME/OpenplanetNext".
        """
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

def default_callbacks(op_path : Path = None):
    """Returns recommended callbacks for training.

    Args:
        op_path (Path) : Path to Openplanet installation folder. Default is "C:/Users/NAME/OpenplanetNext".
    """
    return CallbackList([TMPauseOnUpdate(op_path), TMResetOnEpoch(op_path), TMSaveOnEpochs(name="TMBot", n_epochs=5)])
