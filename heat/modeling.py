from typing import List
import numpy as np

from heat import heat

class Modeling:

    def __init__(self, heat):
        self.heat = heat

        design: np.array
        predictors: List[str]

        #self.design = pd.DataFrame(self.design, columns=self.predictors)

    # fit different GLM, etc.
    def fitlm(self):
        pass

