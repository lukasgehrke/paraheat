import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy import ndimage

from skimage import morphology
import skimage

### general matplotlib properties
font = {'family' : 'sans-serif',
    'sans-serif': 'Ariel'}
matplotlib.rc('font', **font)

class Plot():
    # class with logic to construct a plot with defaults

    def __init__(self, title='Plot title', data_list=[], p_list=[]):

        # set plot defaults
        self.title = title
        self.data_list = data_list
        self.p_list = p_list
        # self.cmap = self.set_colormaps(plt.cm.coolwarm, 1)
        self.lims = [-abs(np.array(self.data_list)).max() / 2 , abs(np.array(self.data_list)).max() / 2] # EEGLAB standard format for cbar

    ### below the plot setup/constructor functions
    def set_background_image(self, img_path):
        self.bg_image = mpimg.imread(img_path)
        self.bg_image_extent = get_image_extent(self.bg_image)

    def set_colormaps(self, cmaps, transparent, transparent_cmap_indices):
        self.cmap = cmaps

        # TODO capture case if only one cmap give

        for i in range(len(self.cmap)):
            self.cmap[i].set_bad('white', 0.)

            if transparent and bool(np.mean(ismember(transparent_cmap_indices, i))):
                self.cmap[i] = make_cm_transparent(self.cmap[i])

    def set_colormap_limits(self, lims):
        self.lims = lims

    def set_subplot_axes(self):

        number_of_axes = len(self.data_list)

        # TODO fix the number of subplots
        # axes_x = number_of_axes // 2 + bool(number_of_axes % 2)

        # if number_of_axes > 1:
        #     axes_y = 2
        # else:
        #     axes_y = 1

        # self.fig, self.axes = plt.subplots(axes_x, axes_y)

        # TODO find a way to set figure size
        self.fig, self.axes = plt.subplots(1, number_of_axes, figsize=(5*number_of_axes,2.5))
        # self.fig, self.axes = plt.subplots(1, number_of_axes, figsize=(5*number_of_axes,2.5))

    def set_axes_labels(self, axes_titles, x_labels, y_labels):
        c = 0
        for ax in self.axes.reshape(-1):
            ax.set_title(axes_titles[c])
            ax.set_xlabel(x_labels[c])
            ax.set_ylabel(y_labels[c])
            c = c + 1

    def set_axes_format(self):
        for ax in self.axes.reshape(-1):
            for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
                # item.set_fontname(font)
                item.set_fontsize(14)

            for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                # item.set_fontname(font)
                item.set_fontsize(12)

            # remove bounding box
            for boundary in ['top', 'right', 'bottom', 'left']:
                ax.spines[boundary].set_visible(False)

    ### below set and prepare data
    def cut_area_of_interest(self):
        pass

    #     def poly_mask(array_like, polygon_points):
    #     # convolve heatmap with boolean matrix of polygon cutout
    #     # 1. make bool array of size heatmap

    #     d0 = array_like.shape[0]
    #     d1 = array_like.shape[1]

    #     x, y = np.meshgrid(np.arange(d0), np.arange(d1)) # make a canvas with coordinates
    #     points = np.vstack((x.flatten(),y.flatten())).T

    #     p = Path(polygon_points) # make a polygon
    #     grid = p.contains_points(points)

    #     mask = grid.reshape(d0,d1) # now you have a mask with points inside a polygon

    #     return mask

    # def clip_img_poly_patch(ax, img, polygon_points):
    #     # see https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_clip_path.html for implementation

    #     patch = patches.Polygon(polygon_points, closed=True, transform=ax.transData)
    #     img.set_clip_path(patch)

    def remove_islands(self):

        for i in range(len(self.data_list)):
            self.data_list[i] = remove_significant_masks_islands(np.squeeze(self.data_list[i]))

    ### below the draw functions
    def draw_background_image(self, alpha=.5, origin='lower'):

        self.bg_image = np.flipud(self.bg_image) # TODO clean this up

        for ax in self.axes.reshape(-1):
            ax.imshow(self.bg_image, alpha=alpha, origin=origin)

    def draw_heat(self, alpha=1, origin='lower', apply_gaussian_blur=0, sigma=1, draw_cbar=0, cbar_label=''):
        c = 0
        for ax in self.axes.reshape(-1):

            data_to_plot = np.squeeze(self.data_list[c])
            # data_to_plot[data_to_plot==0] = np.nan

            data_to_plot = np.rot90(data_to_plot,3) # TODO clean this up
            data_to_plot = np.fliplr(data_to_plot)

            if apply_gaussian_blur:
                data_to_plot = gaussian_blur(data_to_plot, sigma)

            im = ax.imshow(data_to_plot, extent=self.bg_image_extent, origin=origin,
                alpha=alpha, cmap=self.cmap[c], vmin=self.lims[c][0], vmax=self.lims[c][1])

            if draw_cbar:
                add_colorbar(im, ax, cbar_label[c], 90) # TODO refactor, logic of single purpose a bit violated here

            c = c + 1

    def draw_contour(self, significance_thresholds=1, origin='lower'):
        c = 0
        for ax in self.axes.reshape(-1):
            if self.p_list[c]:
                mask = np.rot90(np.squeeze(self.p_list[c]),3) # TODO clean this up
                mask = np.fliplr(mask)
                # mask = gaussian_blur(mask,.05)
                mask = remove_significant_masks_islands(mask)

                ax.contour(mask, extent=self.bg_image_extent, origin=origin, levels=significance_thresholds, colors='black')

                # contours = ax.contour(mask, extent=self.bg_image_extent, origin=origin, levels=significance_thresholds, colors='black')
                # ax.clabel(contours, inline=True, fontsize=8) # TODO enable by flag or separate into another function
            c = c + 1

    def draw_colorbar():
        pass
        # TODO refactor and add here!! maybe use decorator to draw_heat with a colormap

    ### below export and save functions (also for online publishing of interactive plots?)
    def finish_plot(self, vertical_spacing, horizontal_spacing):
        self.fig.subplots_adjust(wspace=vertical_spacing, hspace=horizontal_spacing)

    def export(self, savepath, filename):

        # try:
        #     os.mkdir(savepath)
        # except OSError as error:
        #     print(error)

        # TODO refactor to better use path tools
        plt.savefig(savepath+'/'+filename, dpi=300)



### general purpose helper functions maintained in the same namespace
def make_cm_transparent(cmap):
    "Copy colormap and set alpha values"

    cmap._init()
    cmap._lut[:,-1] = np.linspace(0, 1, cmap._lut.shape[0])

    return cmap

def make_divergent_cm_transparent(cmap, N=255):

    cmap._init()
    pos = np.linspace(0, 1, int(np.floor(cmap._lut.shape[0]/2)))
    neg = pos[::-1]
    center = np.array([0])
    cmap._lut[:,-1] = np.concatenate([neg, center, pos])

    return cmap

def get_image_extent(img):
    return [0, img.shape[1], 0, img.shape[0]]

def add_colorbar(im, ax, label='', rotation=0):
    # TODO needs some changes about axes access
    # colorbar
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.15)
    cbar = plt.colorbar(im, cax=cax)

    cbar.set_label(label, rotation=rotation, fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    cbar.outline.set_visible(False)

### image processing functions
def gaussian_blur(image, sigma=5):
    return ndimage.gaussian_filter(image, sigma=sigma)

def remove_significant_masks_islands(image): #, structure=np.ones((1,1))):
    # return ndimage.binary_opening(binary_array, structure=structure).astype(int)

    grayscale = skimage.color.rgb2gray(image)
    binarized = np.where(grayscale>0.1, 1, 0)
    processed = morphology.remove_small_objects(binarized.astype(bool), min_size=2, connectivity=1).astype(int)
    # TODO make so this provides some defaults but settings can also be given

    # black out pixels
    mask_x, mask_y = np.where(processed == 0)
    image[mask_x, mask_y] = 0

    return image

def ismember(A, B):
    #https://stackoverflow.com/questions/25923027/matlab-ismember-function-in-python
    return [ np.sum(a == B) for a in A ]