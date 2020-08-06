from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['AppleGothic']

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

from mpl_toolkits.axes_grid1 import make_axes_locatable

### From here below general purpose helper functions maintained in the same namespace
def create_figure_axes(number_of_axes=1):

    axes_x = number_of_axes // 2 + bool(number_of_axes % 2)

    if number_of_axes > 1:
        axes_y = 2
    else:
        axes_y = 1

    fig, axes = plt.subplots(axes_x, axes_y)

    return fig, axes

def make_cm_transparent(cmap, N=255):
    "Copy colormap and set alpha values"

    # TODO confirm corrected, add and check correct alpha scaling for diverging cmaps

    cm = cmap
    cm._init()
    cm._lut[:,-1] = np.linspace(0, 1, N+4)

    return cm

def get_image_extent(img):
    return [0, img.shape[1], img.shape[0], 0]

def add_background_image(bg_image, ax,
    origin='upper', alpha=.5):
    # overlay with image from task/setting
    # - get resolution of input image
    # - get aspect ration
    # - define heatmap binning based on aspect ratio and subsampling/scaling factor
    # - rescale to input image size using subsampling scaling factor

    # TODO refactor to set default values
    ax.imshow(bg_image, alpha=alpha, origin=origin)

def add_heat(array_like, ax, extent, alpha=1, cm=plt.cm.jet, origin="upper",
    add_contour=False, contour_mask=None, levels=1):

    heat = ax.imshow(array_like, extent=extent, origin=origin,
        alpha=alpha, cmap=cm)

    if add_contour:
        contours = ax.contour(contour_mask, 1, extent=extent, levels=levels, colors='black')
        ax.clabel(contours, inline=True, fontsize=8)

    return heat

def poly_mask(array_like, polygon_points):
    # convolve heatmap with boolean matrix of polygon cutout
    # 1. make bool array of size heatmap

    d0 = array_like.shape[0]
    d1 = array_like.shape[1]

    x, y = np.meshgrid(np.arange(d0), np.arange(d1)) # make a canvas with coordinates
    points = np.vstack((x.flatten(),y.flatten())).T

    p = Path(polygon_points) # make a polygon
    grid = p.contains_points(points)

    mask = grid.reshape(d0,d1) # now you have a mask with points inside a polygon

    return mask

def clip_img_poly_patch(ax, img, polygon_points):
    # see https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_clip_path.html for implementation

    patch = patches.Polygon(polygon_points, closed=True, transform=ax.transData)
    img.set_clip_path(patch)

def add_colorbar(img, ax):

    # colorbar
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    plt.colorbar(img, cax=cax)

### Plot formatting funcs below
def set_labelnames(ax,
    title, xlabel, ylabel):

    # parse keyword value pairs, e.g. ax.title = title
    # TODO define defaults for ticks etc.
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def add_formatted_legend(ax, **kwargs):

    # TODO add formatting after parsing kwargs
    ax.legend()

def format_axes(ax):
    # TODO reformat all the label, ticksizes, tightness, fonts, faces, lines etc.
    # https://matplotlib.org/tutorials/introductory/usage.html#figure-parts

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

    # remove bounding box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

### Plot show and export
def show():
    plt.show()

def export(savepath, filename):
    # TODO remove and change later
    plt.savefig(savepath+filename+'.png', dpi=300, bbox_inches='tight')






# TODO better make a plotbuilder pipeline/class here





## Deprecated Stuff below
# class ParaHeatPlot:
#     # think about whether this has to be a class or can just be some module with functions

#     def __init__(heat, bg_image=None):

#         # TODO might refactor if/else to usage of a dict some time
#         if heat.heatmap is None:
#             heat_to_plot = heat
#         else:
#             heat_to_plot = heat.heatmap

#         if bg_image is None:
#             bg_image = None
#             extent = []
#         else:
#             bg_image = bg_image
#             extent = extent = [0, bg_image.shape[1], bg_image.shape[0], 0]

#         # defaults
#         cm = plt.cm.jet

#         # plot defaults
#         fig_size = (16,9)


# ### functions to deprecate after integrating above
# def heat_xy(title="heat", lims=[0,1], superimpose_on_bg=False, mask=None):

#     fig, ax = plt.subplots(figsize=(fig_size))

#     if superimpose_on_bg:
#         ax.imshow(bg_image,
#             alpha=bg_image_alpha,
#             origin=origin)

#     if mask is not None:
#         contours = plt.contour(mask, levels=1, colors='black')
#         plt.clabel(contours, inline=True, fontsize=8)

#     heat_plot = ax.imshow(map,
#         extent=extent,
#         alpha=1,
#         cmap=cm,
#         origin=origin)

#     # cb = ax.contourf(xedges, yedges,
#         # map.reshape(xedges, yedges),
#         # 15, cmap=cm)

#     # remove bounding box
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     # move ticks out from axis
#     # ax.tick_params(axis="y",direction="in", pad=-100)
#     # ax.tick_params(axis="x",direction="in", pad=-30)

#     # title
#     plt.title(title)

#     # TODO add contour

#     # set limits
#     #plt.clim(lims)

#     # colorbar
#     # create an axes on the right side of ax. The width of cax will be 5%
#     # of ax and the padding between cax and ax will be fixed at 0.05 inch.
#     ax = plt.gca()
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="2.5%", pad=0.1)
#     plt.colorbar(heat_plot, cax=cax)