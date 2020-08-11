# create a heatmap out of timeseries input data and keep the data structure
# can always be directly used for single subject data as well as at the end to prepare plot? otherwise is used by Analyses to prepare per participant binning
# keep this very simple, stupid
# might have more function for binning later on down the road and can cutout a polygon from the input data before hist binning

# checks data structure whether 2D, 3D or 4D

import numpy as np
import pandas as pd
from scipy import stats

# import matplotlib.image as mpimg

from dataclasses import dataclass

# TODO
# - call binned_statistic during construction, so attribute heatmap of paraheat is always there

@dataclass
class ParaHeat:
    """Wrapper class built around scipy's binned_statistic functions

    Returns:
        [type]: [description]
    """

    name: str
    data: pd.DataFrame
    heatmap: None

    # bg_image: np.array

    # edges: (interquantile range [5 95] across all participants)
    # edges: List[float]

    # heatmaps: List[heat.Map]
    # projections: List[heat.Projection]

    def __post_init__(self):

        if self.data is not None:
            self.col_names = list(self.data.columns)

    #     if self.data.shape[1] is 2:
    #         self.data.Z = None


    #     """[]
    #     """

    #     # provide bins and edges or the program makes an educated guess
    #     # freedman rule
    #     # self.bins = int(np.log2(max(self.bg_image.shape)) * 10)
    #     # print(self.bins)

    #     # get resolution
    #     # self.extent = [self.data.X.min(), self.data.X.max(), self.data.Y.min(), self.data.Y.max()]
    #     # self.extent = [0, self.bg_image.shape[1], self.bg_image.shape[0], 0]
    #     # print(self.extent)

    #wrapper for 2d binning

    def create_binned_statistic(self):
        # builder pattern to set variable and call correct binned_statistic computation
        pass

    def binned_statistic(self, bins=None, agg_stats_func='count'):

        # if bins is not None:
        #     bins_in = bins
        # else:
        #     # provide bins and edges or the program makes an educated guess
        #     # freedman rule
        #     bins_in = int(np.log2(max(self.bg_image.shape)) * 10)

        # hist, xedges, yedges = np.histogram2d(self.data.X, self.data.Y, bins=bins_in)

        # return hist, xedges, yedges

        if len(self.col_names) == 2:
            ret = stats.binned_statistic_2d(self.data[self.col_names[0]], self.data[self.col_names[1]], None, agg_stats_func, bins=bins)

            return ret

    def histogram3d(self, bins=None):

        hist = np.histogramdd()

    def select_aoi(self, aoi):
        # cut out polynom from data and retain either the whats outside or inside
        pass






# Stuff for later implementation
# import scipy.spatial as sp
    # def transform_data_to_view_coord(self, p, resolution, pmin, pmax):
    #     """
    #     Fit data to image resolution

    #     Args:
    #         p ([type]): [description]
    #         resolution ([type]): [description]
    #         pmin ([type]): [description]
    #         pmax ([type]): [description]

    #     Returns:
    #         [type]: [description]
    #     """

    #     dp = pmax - pmin
    #     dv = (p - pmin) / dp * resolution

    #     return dv

    # def knn2d(self, neighbours=32, dim=2):
    #     """[summary]

    #     Args:
    #         x ([type]): [description]
    #         y ([type]): [description]
    #         resolution ([type]): [description]
    #         neighbours (int, optional): [description]. Defaults to 32.
    #         dim (int, optional): [description]. Defaults to 2.

    #     Returns:
    #         [type]: [description]
    #     """

    #     # Create the tree
    #     tree = sp.cKDTree(self.data)
    #     # Find the closest nnmax-1 neighbors (first entry is the point itself)

    #     # import pdb; pdb.set_trace()
    #     grid = np.mgrid[0:self.bg_image.shape[0], 0:self.bg_image.shape[1]].T.reshape(self.bg_image.shape[0]*self.bg_image.shape[1], dim)

    #     dists = tree.query(grid, neighbours)
    #     # Inverse of the sum of distances to each grid point.
    #     inv_sum_dists = 1. / dists[0].sum(1)

    #     # Reshape
    #     im = inv_sum_dists.reshape(self.bg_image.shape[0], self.bg_image.shape[1])
    #     return im