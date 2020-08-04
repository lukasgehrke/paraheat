from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['AppleGothic']

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

class Plot:

    def __init__(self, heat, bg_image, extent, fig_size=(16,9)):
        self.map = heat.heatmap
        self.extent = extent
        self.bg_image = bg_image

        self.cm = plt.cm.jet

        # plot params
        self.fig_size = fig_size
        self.bg_image_alpha = .5
        self.origin = 'lower'

        SMALL_SIZE = 20
        MEDIUM_SIZE = 24
        BIGGER_SIZE = 32

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    def create_plot_window(self):
        fig, ax = p.create_plot_window()

    def heat_xy(self, title="heat", lims=[0,1], superimpose_on_bg=False, mask=None):

        fig, ax = plt.subplots(figsize=(self.fig_size))

        if superimpose_on_bg:
            ax.imshow(self.bg_image,
                alpha=self.bg_image_alpha,
                origin=self.origin)

        if mask is not None:
            contours = plt.contour(mask, levels=1, colors='black')
            plt.clabel(contours, inline=True, fontsize=8)

        heat_plot = ax.imshow(self.map,
            extent=self.extent,
            alpha=1,
            cmap=self.cm,
            origin=self.origin)

        # cb = ax.contourf(xedges, yedges,
            # self.map.reshape(xedges, yedges),
            # 15, cmap=self.cm)

        # remove bounding box
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # move ticks out from axis
        # ax.tick_params(axis="y",direction="in", pad=-100)
        # ax.tick_params(axis="x",direction="in", pad=-30)

        # title
        plt.title(title)

        # TODO add contour

        # set limits
        #plt.clim(lims)

        # colorbar
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.1)
        plt.colorbar(heat_plot, cax=cax)

    def heat_c(self):
        print(self.map)

    def make_cm_transparent(self, cmap, N=255):
        "Copy colormap and set alpha values"

        mycmap = cmap
        mycmap._init()
        mycmap._lut[:,-1] = np.linspace(0, 1, N+4)

        return mycmap

    def add_background_image(self):
        # overlay with image from task/setting
        # - get resolution of input image
        # - get aspect ration
        # - define heatmap binning based on aspect ratio and subsampling/scaling factor
        # - rescale to input image size using subsampling scaling factor
        pass

    def show(self):
        plt.show()

    def export(self, savepath, filename):
        # TODO remove and change later
        plt.savefig(savepath+filename+'.png', dpi=300, bbox_inches='tight')

    # use imshow to plot both heatmap and bg_image so they can be overlayed

