# this class controls the handling of several trials, it creates heatmaps for each trials and coordinates the processing pipeline

# checks data structure whether 2D, 3D or 4D

import numpy as np
import pandas as pd
# import matplotlib.image as mpimg

from scipy import stats
import scipy.spatial as sp
from scipy.ndimage import gaussian_filter

from dataclasses import dataclass
from typing import List

#from heat.heatmap import Heatmap
#from heat.projection import Projection

# import heat

@dataclass
class Heat:

    name: str
    data: pd.DataFrame
    heatmap: np.array
    bg_image: np.array

    # edges: (interquantile range [5 95] across all participants)
    # edges: List[float]

    # heatmaps: List[heat.Map]
    # projections: List[heat.Projection]

    def __post_init__(self):
        """[]
        """

        # provide bins and edges or the program makes an educated guess
        # freedman rule
        # self.bins = int(np.log2(max(self.bg_image.shape)) * 10)
        # print(self.bins)

        # get resolution
        # self.extent = [self.data.X.min(), self.data.X.max(), self.data.Y.min(), self.data.Y.max()]
        self.extent = [0, self.bg_image.shape[1], self.bg_image.shape[0], 0]
        # print(self.extent)

    def histogram2d(self, bins):

        if bins is not None:
            bins_in = bins
        else:
            # provide bins and edges or the program makes an educated guess
            # freedman rule
            bins_in = int(np.log2(max(self.bg_image.shape)) * 10)

        hist, xedges, yedges = np.histogram2d(self.data.X, self.data.Y, bins=bins_in)

        return hist, xedges, yedges

    def gaussian(self, image, sigma=5):

        gaussian_image = gaussian_filter(image, sigma=sigma)

        return gaussian_image

    def zscore(self, image):

        zscored_image = stats.zscore(image, axis=None)

        return zscored_image

    def select_aoi(self, aoi):
        pass

    def transform_data_to_view_coord(self, p, resolution, pmin, pmax):
        """
        Fit data to image resolution

        Args:
            p ([type]): [description]
            resolution ([type]): [description]
            pmin ([type]): [description]
            pmax ([type]): [description]

        Returns:
            [type]: [description]
        """

        dp = pmax - pmin
        dv = (p - pmin) / dp * resolution

        return dv

    def knn2d(self, neighbours=32, dim=2):
        """[summary]

        Args:
            x ([type]): [description]
            y ([type]): [description]
            resolution ([type]): [description]
            neighbours (int, optional): [description]. Defaults to 32.
            dim (int, optional): [description]. Defaults to 2.

        Returns:
            [type]: [description]
        """

        # Create the tree
        tree = sp.cKDTree(self.data)
        # Find the closest nnmax-1 neighbors (first entry is the point itself)

        # import pdb; pdb.set_trace()
        grid = np.mgrid[0:self.bg_image.shape[0], 0:self.bg_image.shape[1]].T.reshape(self.bg_image.shape[0]*self.bg_image.shape[1], dim)

        dists = tree.query(grid, neighbours)
        # Inverse of the sum of distances to each grid point.
        inv_sum_dists = 1. / dists[0].sum(1)

        # Reshape
        im = inv_sum_dists.reshape(self.bg_image.shape[0], self.bg_image.shape[1])
        return im