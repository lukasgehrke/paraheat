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
df = pd.read_csv('/Users/lukasgehrke/Documents/temp/chatham/crd_gaze_phys-LOW_work-HIGH_all_good_s.csv')
# select a subject
s1 = df[df['pID'] == 5]
s1[['X', 'Y']]

# read background image for heatmap
img = mpimg.imread('/Users/lukasgehrke/Documents/temp/matb.png')
# plt.axis('off')

h = heat.Heat('workhigh', s1, None, img)
h.heatmap = h.histogram2d()
h.heatmap = h.zscore(h.heatmap)
h.heatmap = h.gaussian(h.heatmap, 2)

# create plot and add plot data
p = plot.Plot(h.heatmap)
fig, ax = p.create_plot_window()
my_cm = p.transparent_cmap(plt.cm.coolwarm)

# create plot window
ax.imshow(img, alpha=1) # for image

# add heatmap
p.heat_xy(ax, p.map, h.extent, my_cm)

# draw plot
plt.show()
