# This is an example script to run paraheat using prediction error experiment reaching data

# %% imports

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import class
import sys
sys.path.append('/Users/lukasgehrke/Documents/code/paraheat/heat/')

from first_level import OLS, GLM, Paired_T
from multcomp import MCC_MNE
from plot import Plot

%load_ext autoreload
%autoreload 2

# %% read raw data and split into data and design

df = pd.read_csv('/Users/lukasgehrke/Documents/temp/chatham/pe_reach_all_good_s.csv')

data = df[['X', 'Z', 'FCz_ERP']] # 'Mag_Vel'
data.columns = ['X', 'Y', 'Z']

design = df[['pID' ,'TrialNr', 'Haptics', 'age', 'ipq_vibro', 'ipq_visual']]
design['Haptics'] = design['Haptics'].astype('category')

# %% create paraheat first level analyses

st = Paired_T(name="test", data=data, design_matrix=design, model="Pixel ~  Haptics")
bins = 100
edges = [[-1.5, 1.5], [-.2, 1.5]]
# st.create_heatmaps('pID', 3, bins, edges, 'count')
st.create_heatmaps('pID', 3, bins, edges, 'mean')

# %% save results

with open('/Users/lukasgehrke/Documents/code/paraheat/tests/pe_reaching_heatmaps_100_edges_paired_T_fcz1.pkl', 'wb') as f:
    pickle.dump(st, f)

# %% fit the model

# %% load results

# with open('/Users/lukasgehrke/Documents/code/paraheat/tests/pe_reaching_heatmaps_100_edges_paired_T.pkl', 'rb') as f:
with open('/Users/lukasgehrke/Documents/code/paraheat/tests/pe_reaching_heatmaps_100_edges_paired_T_fcz1.pkl', 'wb') as f:
    st = pickle.load(f)

# %%
# st.nan_to_zero()
# st.standardize('zscore')
st.fit()

# Create new stats image with only significant clusters
# T_obs_plot = np.nan * np.ones_like(st._t_obs)
# T_obs_plot = st._t_obs 

# %%

T_obs_plot = np.zeros_like(st._t_obs)
# TODO below must go into own function to prepare heatmap results for plotting
for c, p_val in zip(st._clusters, st._cluster_p_values):
    if p_val <= 0.05:
        T_obs_plot[c] = st._t_obs[c]

# call heatmap plotting function and plot grand averages for each condition
# make 4 subplots: cond 1, cond 2, diff, t-map
# get data for plot
cond1 = np.mean(st._grand_averages[0],0)
cond2 = np.mean(st._grand_averages[1],0)
diff = np.mean(st._grand_averages[2],0)
t = st._t_obs

# nans to zeros
cond1[np.isnan(cond1)] = 0.0
cond2[np.isnan(cond2)] = 0.0
diff[np.isnan(diff)] = 0.0
t[np.isnan(t)] = 0.0

# %%

p = Plot('test', [[cond1], [cond2], [diff], [t]], [[],[],[t],[t]])
p.set_background_image('/Users/lukasgehrke/Documents/chatham_internship/figures/background_pe_reaching.png')
p.set_subplot_axes()
p.set_axes_labels([st._grand_averages_condition[0], st._grand_averages_condition[1], st._grand_averages_condition[2], 'Accuracy'], ['X', 'X', 'X', 'X'], ['Y', 'Y', 'Y', 'Y'])
p.set_axes_format()
# p.remove_islands()
# p.set_colormaps([plt.cm.viridis, plt.cm.viridis, plt.cm.coolwarm, plt.cm.coolwarm], 1, [0, 1, 2, 3])
p.set_colormaps([plt.cm.coolwarm, plt.cm.coolwarm, plt.cm.coolwarm, plt.cm.viridis], 1, [2, 3])
p.set_colormap_limits([[-1, 1], [-1, 1], [-.5, .5], [0, 1]]) # make so a list of lists can be provided for each axes

p.draw_background_image() # add list determining on which axes to draw?
p.draw_heat(apply_gaussian_blur=1, sigma=1.5, draw_cbar=1, cbar_label=['EEG', 'EEG', '', '% correct'])
# p.draw_heat(apply_gaussian_blur=1, sigma=1.5, draw_cbar=1, cbar_label='Amplitude FCz')
p.draw_contour()
p.finish_plot(.5,0)
# p.export('/Users/lukasgehrke/Documents/code/paraheat/figures_out', 'pe_reaching.pdf')

# %%
