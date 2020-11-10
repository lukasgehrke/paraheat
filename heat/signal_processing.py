# simple collection of wrapped signal processing functions
import numpy as np
import scipy as sp
import pandas as pd

# def zscore(pd_df):
#     return (pd_df - pd_df.mean())/pd_df.std()

def gaussian(image, sigma=5):
    return sp.ndimage.gaussian_filter(image, sigma=sigma)