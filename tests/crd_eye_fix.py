# This is the example using crd eye fix data!

# %%

# read data for tests
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import class
import sys
sys.path.append('/Users/lukasgehrke/Documents/code/paraheat/heat/')

from first_level import OLS, GLM, Independent_T, Paired_T
from multcomp import MCC_MNE
from plot import Plot

%load_ext autoreload
%autoreload 2

# %% load raw data

df = pd.read_csv('/Users/lukasgehrke/Documents/temp/chatham/crd_fix_all_good_s_bike.csv')
# df = pd.read_csv('/Users/lukasgehrke/Documents/temp/chatham/crd_fix_all_good_s.csv')
data = df[['X', 'Y']]
#data.head()
design = df[['partId', 'workloadLevel', 'physicalLevel']]
#design.head()

# %% create analyses, design and resulting aggregate heatmaps

st = GLM(name="test", data=data, design_matrix=design, model="workloadLevel ~ Pixel")
bins = 100
st.create_heatmaps('partId', 2, bins, None, 'count')

# %% save

with open('/Users/lukasgehrke/Documents/code/paraheat/tests/crd_data_bike_heatmaps_100_GLM_workload.pkl', 'wb') as f:
    pickle.dump(st, f)

# %% load

# with open('/Users/lukasgehrke/Documents/code/paraheat/tests/crd_data_bike_heatmaps_100_paired_T_workload.pkl', 'rb') as f:
with open('/Users/lukasgehrke/Documents/code/paraheat/tests/crd_data_bike_heatmaps_100_GLM_workload.pkl', 'rb') as f:
    st = pickle.load(f)

# %% modelfitting and multiple comparisons

# st.zero_to_nan()
# st.standardize('zscore')
st.fit()

# %%

cond1 = np.reshape(st._heatmaps_betas['Intercept'].ravel(), (100,100))
cond2 = np.reshape(st._heatmaps_ps['Pixel'].ravel(), (100,100))
# cond1 = cond1[1].tolist()
# cond2 = cond2[1].tolist()


# %% extract data to plot
# Create new stats image with only significant clusters

T_obs_plot = np.zeros_like(st._t_obs)
for c, p_val in zip(st._clusters, st._cluster_p_values):
    if p_val <= 1:
        T_obs_plot[c] = st._t_obs[c]

cond1 = np.nanmean(st._grand_averages[0],0)
cond2 = np.nanmean(st._grand_averages[1],0)
diff = np.nanmean(st._grand_averages[2],0)
t = st._t_obs

# nans to zeros
cond1[np.isnan(cond1)] = 0.0
cond2[np.isnan(cond2)] = 0.0
diff[np.isnan(diff)] = 0.0
t[np.isnan(t)] = 0.0

# %% plot

p = Plot('test', [[cond2], [cond2], [cond2], [cond2]], [[],[],[],[]])
p.set_background_image('/Users/lukasgehrke/Documents/temp/matb.png')
p.set_subplot_axes()
# p.set_axes_labels([st._grand_averages_condition[0], st._grand_averages_condition[1], st._grand_averages_condition[2], 'T'], ['X', 'X', 'X', 'X'], ['Y', 'Y', 'Y', 'Y'])
p.set_axes_format()
# p.remove_islands()
p.set_colormaps([plt.cm.coolwarm, plt.cm.viridis, plt.cm.coolwarm, plt.cm.coolwarm], 1, [0, 1, 2, 3, 4])
p.set_colormap_limits([[0, 1], [0, 10], [-4, 4], [-1.5, 1.5]]) # make so a list of lists can be provided for each axes

p.draw_background_image() # add list determining on which axes to draw?
p.draw_heat(apply_gaussian_blur=1, sigma=1.5, draw_cbar=1, cbar_label=['# Fixations', '# Fixations', '', 't stat.'])
p.draw_contour()
p.finish_plot(.5,0)
# p.export('/Users/lukasgehrke/Documents/code/paraheat/figures_out', 'pe_reaching.pdf')

# %%
