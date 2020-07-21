import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

from heat import Heat

# read data
df = pd.read_csv('/Users/lukasgehrke/Documents/temp/chatham/crd_gaze_phys-LOW_work-HIGH_all_good_s.csv')
# select a subject
s1 = df[df['pID'] == 5]
s1[['X', 'Y']]

s2 = df[df['pID'] == 3]
s2[['X', 'Y']]

# read background image for heatmap
img = mpimg.imread('/Users/lukasgehrke/Documents/temp/matb.png')
plt.imshow(img, alpha=1) # for image
#plt.axis('off')

xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()


h = Heat('workhigh', s1, None, img)
# h_knn = h.kNN2DDens()

h.heatmap = h.histogram2d()
h.heatmap = h.gaussian(h.heatmap, sigma=2)

# zscored_smoothed_h_2d = h.zscore(smoothed_h_2d)

plt.imshow(h.heatmap, cmap='coolwarm', interpolation='nearest', alpha=.5, extent=(xmin,xmax,ymin,ymax))   # for heatmap to overlap
plt.show()









# f, ax = plt.subplots(figsize=(16, 9))
# ax.imshow(h_knn, origin='lower', extent=h.extent, cmap=cm.Blues)

# ax.set_xlim(h.extent[0], h.extent[1])
# ax.set_ylim(h.extent[2], h.extent[3])

# #sns.jointplot(s1.X, s1.Y, kind="hex")
# sns.kdeplot(s1.X, s1.Y, \
#     gridsize=100, legend=True, shade=True, cbar=True)
# ax.collections[0].set_alpha(0)
# plt.show()

# #plt.savefig('demo.pdf', transparent=True)





# im = kNN2DDens(xv, yv, resolution, neighbours)

# ax.imshow(im, origin='lower', extent=extent, cmap=cm.Blues)
# ax.set_title("Smoothing over %d neighbours" % neighbours)

# ax.set_xlim(extent[0], extent[1])
# ax.set_ylim(extent[2], extent[3])