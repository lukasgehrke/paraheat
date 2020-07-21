# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from scipy.ndimage.filters import gaussian_filter


# def myplot(x, y, s, bins=1000):
#     heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
#     heatmap = gaussian_filter(heatmap, sigma=s)

#     extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#     return heatmap.T, extent


# fig, axs = plt.subplots(2, 2)

# # Generate some test data
# x = np.random.randn(1000)
# y = np.random.randn(1000)

# sigmas = [0, 16, 32, 64]

# for ax, s in zip(axs.flatten(), sigmas):
#     if s == 0:
#         ax.plot(x, y, 'k.', markersize=5)
#         ax.set_title("Scatter plot")
#     else:
#         img, extent = myplot(x, y, s)
#         ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
#         ax.set_title("Smoothing with  $\sigma$ = %d" % s)

# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import Image

#2D Gaussian function
def twoD_Gaussian(x, y, xo, yo, sigma_x, sigma_y):
    a = 1./(2*sigma_x**2) + 1./(2*sigma_y**2)
    c = 1./(2*sigma_x**2) + 1./(2*sigma_y**2)
    g = np.exp( - (a*((x-xo)**2) + c*((y-yo)**2)))
    return g.ravel()


def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap


#Use base cmap to create transparent
mycmap = transparent_cmap(plt.cm.Reds)


# Import image and get x and y extents
I = Image.open('./deerback.jpg')
p = np.asarray(I).astype('float')
w, h = I.size
y, x = np.mgrid[0:h, 0:w]

#Plot image and overlay colormap
fig, ax = plt.subplots(1, 1)
ax.imshow(I)
Gauss = twoD_Gaussian((x, y), .5*x.max(), .4*y.max(), .1*x.max(), .1*y.max())
cb = ax.contourf(x, y, Gauss.reshape(x.shape[0], y.shape[1]), 15, cmap=mycmap)
plt.colorbar(cb)
plt.show()