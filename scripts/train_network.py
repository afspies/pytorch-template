import pytorch-template as project
from pathlib import Path
import os.path as op

project.assign_free_gpus(threshold_vram_usage=3500)
project.set_rng_seeds(seed=47)

data_path = Path(op.join(project.__path__[0], 'data'))
