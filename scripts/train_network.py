import pytorch-template as project
from pathlib import Path

project.assign_free_gpus(threshold_vram_usage=3500)
project.set_rng_seeds(seed=47)

data_path = Path(project.__path__[0]).parent / "data"
