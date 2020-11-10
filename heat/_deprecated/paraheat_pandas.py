import pandas as pd
import scipy as sp

@pd.api.extensions.register_dataframe_accessor("hist2d")
class Bin2dAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if 'X' not in obj.columns and 'Y' not in obj.columns:
            raise AttributeError("Must have 'X' and 'Y'.")

    @property # kwargs = arguments from scipy binned statistic
    def bin_stat_2d(self, **kwargs):
        return sp.stats.binned_statistic_2d(self._obj.X, self._obj.Y, None, **kwargs)

@pd.api.extensions.register_dataframe_accessor("hist3d")
class Bin3dAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if 'X' not in obj.columns and 'Y' not in obj.columns and 'Z' not in obj.columns:
            raise AttributeError("Must have 'X', 'Y' and 'Z'.")

    @property
    def bin_stat_3d(self, **kwargs): # kwargs = arguments from scipy binned statistic
        return sp.stats.binned_statistic_2d(self._obj.X, self._obj.Y, self._obj.Z, **kwargs)