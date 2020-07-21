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
df = pd.read_csv('/Users/lukasgehrke/Documents/temp/chatham/crd_gaze_phys-LOW_work-HIGH_equip-Bike_all_good_s.csv')
# select a subject
s1 = df[df['pID'] == 40]
s1[['X', 'Y']]

# read background image for heatmap
img = mpimg.imread('/Users/lukasgehrke/Documents/temp/matb.png')

h = heat.Heat('workhigh', s1, None, img)

# specify resolution
x_edges = np.linspace(0, 1280, 100)
y_edges = np.linspace(0, 720, 100)
bins = [x_edges, y_edges]

h.heatmap = h.histogram2d(bins)
h.heatmap = h.zscore(h.heatmap)
h.heatmap = h.gaussian(h.heatmap, 1)

p = plot.Plot(h)
p.cm = p.make_cm_transparent(p.cm)
p.heat_xy()

# # create plot and add plot data
# fig, ax = p.create_plot_window()
# # fig.axis('off')
# my_cm = p.transparent_cmap(plt.cm.coolwarm)

# # create plot window
# ax.imshow(img, alpha=1) # for image

# # add heatmap
# p.heat_xy(ax, h.heatmap, h.extent, my_cm)

# # draw plot
# plt.show()
