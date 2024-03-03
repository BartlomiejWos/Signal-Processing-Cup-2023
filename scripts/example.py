#!/usr/bin/env python

import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from spcup2023.dataset import spcup23_ds

_DS_PATH = Path("../dataset")

dataset = spcup23_ds(_DS_PATH)

for label, a, b in tqdm(dataset):
    pass
