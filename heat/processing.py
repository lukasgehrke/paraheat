# simple collection of wrapped signal processing functions

import numpy as np
import pandas as pd

# refactor imports below
from scipy.ndimage import gaussian_filter

def zscore(pd_df):
    return (pd_df - pd_df.mean())/pd_df.std()

def gaussian(image, sigma=5):
    return gaussian_filter(image, sigma=sigma)