from dataclasses import dataclass
from heat.data import Data

import numpy as np
import pandas as pd

from scipy.ndimage.filters import gaussian_filter

@dataclass
class Map(Data):
    """[Heatmap of x,y timeseries/coordinates]

    Args:
        Data ([type]): [description]

    Returns:
        [type]: [description]
    """

    super().__init__()
    extent: np.array # TODO check back later if this can be cleaned up
    #self.data = pd.DataFrame(self.data)

    def hist2(self, bins=50, range=None):

        # todo check range_in for region of interest setting across sessions
        heatmap, xedges, yedges = np.histogram2d(self.data[:,0], self.data[:,1], bins=bins, range=range)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        self.data = heatmap
        self.extent = extent

        return self

    # TODO add other grouping methods

    def zscore(self):
        """[summary]

        Returns:
            [type]: [description]
        """

        self.data = (self.data - self.data.mean())/self.data.std()

        return self

    # TODO add other normalisation methods

    def gaussian_filter(self, sigma=5):ed
        """[summary]

        Args:
            sigma (int, optional): [description]. Defaults to 5.

        Returns:
            [type]: [description]
        """

        self.data = gaussian_filter(self.data, sigma=sigma)

        return self