# create a heatmap out of timeseries input data and keep the data structure
# can always be directly used for single subject data as well as at the end to prepare plot? otherwise is used by Analyses to prepare per participant binning
# keep this very simple, stupid
# might have more function for binning later on down the road and can cutout a polygon from the input data before hist binning

# checks data structure whether 2D, 3D or 4D

import abc

import numpy as np
import pandas as pd
from scipy import stats

# import matplotlib.image as mpimg

from dataclasses import dataclass

# TODO
# - call binned_statistic during construction, so attribute heatmap of paraheat is always there

@dataclass
class ParaHeat:
    """Simple wrapper class built around scipy's binned_statistic functions

    Returns:
        [type]: [description]
    """

    name: str
    data: pd.DataFrame
    heatmap: None # TODO refactoring: rename to binned_statistic

    def __post_init__(self):

        if self.data is not None:
            self.col_names = list(self.data.columns)

    @abc.abstractmethod
    def binned_statistic(self):
        pass

    @abc.abstractmethod
    def select_aoi(self):
        pass

@dataclass
class ParaHeat2D(ParaHeat):

    def binned_statistic(self, bins=None, agg_stats_func='count'):

        # provide bins and edges or the program makes an educated guess
        # freedman rule
        if bins is None:
            bins = freedman_bins(self.bg_image.shape)

        # return hist, xedges, yedges
        ret = stats.binned_statistic_2d(self.data[self.col_names[0]], self.data[self.col_names[1]],
            None, agg_stats_func, bins=bins)
        return ret

    def select_aoi(self, aoi):
        # cut out polynom from data and retain either the whats outside or inside
        pass

@dataclass
class ParaHeat3D(ParaHeat):

    def binned_statistic(self):

        # return hist, xedges, yedges
        ret = stats.binned_statistic_2d(self.data[self.col_names[0]], self.data[self.col_names[1]], self.data[self.col_names[2]],
            agg_stats_func, bins=bins)
        return ret

    def select_aoi(self):
        pass

def freedman_bins(size):
    # provide bins and edges or the program makes an educated guess
    # freedman rule
    return int(np.log2(max(size)) * 10)













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