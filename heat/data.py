from dataclasses import dataclass
import numpy as np
from typing import List

@dataclass
class Data:
    """[Class for basic data container]
    """

    name: str
    data: np.array