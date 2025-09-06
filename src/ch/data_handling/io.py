from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Tuple

import pickle

import pandas as pd
import scipy


def load_matlab_file(filename: Path | str) -> Dict[str, Any]:
    filename = str(filename)
    data = scipy.io.loadmat(filename)
    print("Data keys:")
    print(list(data.keys()))
    
    return data
