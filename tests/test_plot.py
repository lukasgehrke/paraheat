import numpy as np
import pandas as pd
from scipy import stats
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# import class to be tested
# from heat.plot import ParaHeatPlot
import heat.plot as ph

# generate some realistic pandas random data and write tests to generate the tests i desire :)
rng = np.random.default_rng()
df = pd.DataFrame(rng.integers(0, 100, size=(100, 2)), columns=list('XY'))
binx = np.arange(0,100,1)
biny = binx

# use function wrapped in heat module
ret = stats.binned_statistic_2d(df.X, df.Y, None, 'count', bins=[binx, biny])

# load a background image
bg_img = mpimg.imread('/Users/lukasgehrke/Documents/temp/matb.png')

def test_create_figure_axes():
    nr_axes = 16
    fig1, axes1 = ph.create_figure_axes(number_of_axes=nr_axes)
    assert axes1.shape[0] == math.ceil(nr_axes // 2), "returned number of axes as expected"

def test_get_image_extent():

    # TODO take iamge from web
    extent = ph.get_image_extent(bg_img)
    assert extent[2] == bg_img.shape[0], "image size matches extent"

def test_get_transparant_colormap():

    cmap = plt.cm.coolwarm
    my_cm = ph.make_cm_transparent(cmap)
    assert np.min(my_cm._lut) == 0, "min of alpha in colormap is 0"
    assert np.max(my_cm._lut) == 1, "max of alpha in colormap is 0"

def test_make_axes_publication_ready():
    pass

def test_make_plot():

    # plot settings and parameters
    extent = ph.get_image_extent(bg_img)
    my_cm = ph.make_cm_transparent(plt.cm.coolwarm)

    # create plot and add elements and formatting
    fig, ax = ph.create_figure_axes(1)
    ph.add_background_image(bg_img, ax)

    heat = ph.add_heat(ret.statistic, ax, extent, cm=my_cm)
    ph.add_colorbar(heat, ax)

    ph.set_labelnames(ax, title="some title", xlabel="some x label", ylabel="some y label")
    ph.format_axes(ax)
    ph.show()

def test_poly_mask():

    # cut a square
    poly = [(.5, .5), (.5, 1.5), (1.5, 1.5), (1.5, .5)]
    mask = ph.poly_mask(ret.statistic, poly)

    assert mask.shape == ret.statistic.shape, "mask and input array shape are equal"

# def test_poly_clip():
#     ph.clip_img_poly_patch(bg_img)