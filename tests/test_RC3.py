import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# import class
from heat.first_level import OLS, GLM
from heat.multcomp import MCC_MNE
import heat.modeling as modeling
import heat.plot as ph

df = pd.read_csv('/Users/lukasgehrke/Documents/temp/chatham/LG_data_crdPhase1/df_scenario1_random_sample.csv')
data = df[['X', 'Y']]
data['Z'] = data['X']

def test_fit_OLS():
    design = df[['pID' ,'Activity', 'Workload', 'Intensity', 'GTLX']]
    design = design.sort_values("pID")

    st = OLS(name="test", data=df, design_matrix=design, model="Pixel ~ Activity * Intensity")
    bins = 10
    st.create_heatmaps('pID', 2, bins, None, 'count')
    st.zero_to_nan()
    st.standardize("zscore")
    st.fit()

    # TODO st.inspect_fit()

    # prepare results
    to_plot = np.reshape(np.array(st._heatmaps_betas["Intercept"]),(bins,bins))
    mask = np.reshape(np.array(st._heatmaps_ps["Intercept"]),(bins,bins))

    # apply mcc
    mask = MCC_MNE.fdr(mask, .05)
    mask = mask[0][0]

    # plot and format it
    bg_img = mpimg.imread('/Users/lukasgehrke/Documents/temp/matb.png')
    extent = ph.get_image_extent(bg_img)
    my_cm = ph.make_cm_transparent(plt.cm.coolwarm)
    fig, ax = ph.create_figure_axes(1)
    ph.add_background_image(bg_img, ax)

    lims = [-20,20]
    sig_levels = 1
    heat = ph.add_heat(to_plot, ax, extent, cm=my_cm, lims=lims, add_contour=True, contour_mask=mask, levels=sig_levels)
    ph.add_colorbar(heat, ax)
    ph.set_labelnames(ax, title="some title", xlabel="some x label", ylabel="some y label")
    ph.format_axes(ax)
    ph.show()

def test_fit_GLM():
    design = df[['pID', 'Activity']]
    design = design.sort_values("pID") # okay since this is random data anyways

    glm = GLM(name="Activity ~ Pixel", data=df, design_matrix=design, model="Activity ~ Pixel")
    bins = 5
    glm.create_heatmaps('pID', 2, bins, None, 'count')
    glm.standardize("zscore")
    glm.zero_to_nan()
    glm.fit()
    
    # glm.inspect_fit()
    
    # apply mcc

    # plot and format it

# multilevel modeling example
def test_fit_OLS_multilevel():
    design = df[['pID' ,'Activity', 'Workload', 'Intensity', 'GTLX']]
    design = design.sort_values("pID")
    # rename pID to trialID and GLTX to RT
    design.rename(columns={"pID": "trialNr", "GTLX": "RT"}, inplace=True)
    df.rename(columns={"pID": "trialNr", "GTLX": "RT"}, inplace=True)

    # pseudocode
    # first level analysis: across trials within a single subject

    # do for each subject and retain results
    st = OLS(name="test", data=df, design_matrix=design, model="Pixel ~ Intensity + RT")
    bins = 10
    st.create_heatmaps('trialNr', 2, bins, None, 'count')
    st.standardize("zscore")
    st.zero_to_nan()
    st.fit()

    # second level analysis: across subjects
    # now using heatmaps (betas) as input!




