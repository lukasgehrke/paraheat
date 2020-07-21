import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# from heat.heat import Heat
#from heat.plot import Plot

import heat, plot

# from heat import Heat
# from plot import Plot

# read data
# df = pd.read_csv('/Users/lukasgehrke/Documents/temp/chatham/crd_gaze_phys-LOW_work-HIGH_equip-Bike_all_good_s.csv')
df = pd.read_csv('/Users/lukasgehrke/Documents/temp/chatham/crd_gaze_phys-LOW_work-HIGH_equip-Treadmill_all_good_s.csv')

# df = pd.read_csv('/Users/lukasgehrke/Documents/temp/chatham/crd_gaze_phys-HIGH_work-HIGH_equip-Bike_all_good_s.csv')
# df = pd.read_csv('/Users/lukasgehrke/Documents/temp/chatham/crd_gaze_phys-HIGH_work-HIGH_equip-Treadmill_all_good_s.csv')

# for subject in range(23, 48):

    # select a subject
    # this_s = df[df['pID'] == subject]

img = mpimg.imread('/Users/lukasgehrke/Documents/temp/matb.png') # read background image for heatmap
h = heat.Heat('workhigh', df, None, img) # create heatmap

# specify resolution
bins = [np.linspace(0, 1280, 100), np.linspace(0, 720, 100)]
h.heatmap, xedges, yedges = h.histogram2d(bins)
h.heatmap = h.zscore(h.heatmap)
h.heatmap = h.gaussian(h.heatmap, 2)
h.heatmap = h.heatmap.T

# create plot and draw
p = plot.Plot(h)
p.cm = p.make_cm_transparent(p.cm)
p.heat_xy(title="Gaze Heatmap Treadmill, grand-average High work, Low phys",
    lims=[0,4], superimpose_on_bg=True)
