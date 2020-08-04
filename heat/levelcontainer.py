
import pandas as pd
import patsy

from dataclasses import dataclass

# class to hold data for one level of a multilevel/hierarchical/mixed-effects analyses
@dataclass
class LevelContainer:

    name: str
    data: pd.DataFrame
    design_matrix: pd.DataFrame
    model: str

    # do some check to confirm data matches design matrix

    # also allow to get a descriptive overview of data and design
    # -> aggregate summary statistics using the given model

    def make_design_matrix(self, model, design_matrix):
        pass
