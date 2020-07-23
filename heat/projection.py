from dataclasses import dataclass
from heat.data import Data

@dataclass
class Projection(Data):
    """[Projection of third dimension data on 2D coordinates]

    Args:
        Data ([type]): [description]
    """
    # inherit data, and set projection settings, then compute projection with __post_init__ function
    # has parameter projection as a string which defines the way how to project the data to 2D, e.g. hist3, kNN
    pass
