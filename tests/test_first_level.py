import pandas as pd

# import class
from heat.first_level import OLS, GLM
import heat.modeling as modeling

# # for now use crd data // later put random data
# bike_plow_whigh = pd.read_csv('/Users/lukasgehrke/Documents/temp/chatham/crd_gaze_phys-LOW_work-HIGH_equip-Bike_all_good_s.csv')
# tread_plow_whigh = pd.read_csv('/Users/lukasgehrke/Documents/temp/chatham/crd_gaze_phys-LOW_work-HIGH_equip-Treadmill_all_good_s.csv')
# df = bike_plow_whigh.append(tread_plow_whigh, ignore_index=True)

# # create the design matrix
# # get unique pIDs per locomotion
# pID_bike = pd.Series(bike_plow_whigh['pID'].unique())
# pID_tread = pd.Series(tread_plow_whigh['pID'].unique())
# pID = pID_bike.append(pID_tread)
# # define contrast
# bike = pd.Series(['bike'] * pID_bike.shape[0])
# tread = pd.Series(['tread'] * pID_tread.shape[0])
# locomotion = bike.append(tread)
# design = pd.concat([pID, locomotion], axis=1)
# design.columns = ['pID', 'Locomotion']
# design.head()

# read data -> emulate this kind of data with a random generated dataset
# df = pd.read_csv('/Users/lukasgehrke/Documents/temp/chatham/LG_data_crdPhase1/df_scenario1.csv')
df = pd.read_csv('/Users/lukasgehrke/Documents/temp/chatham/LG_data_crdPhase1/df_scenario1_random_sample.csv')
# df = df.sample(100000) # select random rows for faster debugging
# df.to_csv('/Users/lukasgehrke/Documents/temp/chatham/LG_data_crdPhase1/df_scenario1_random_sample.csv', index=False)

data = df[['X', 'Y']]
design = df[['pID' ,'Activity', 'Workload', 'Intensity', 'GTLX']]

# sort desing by pid
#design.sort_values("pID", inplace=True)


# design.head()

# define pandas columns as categorical

def test_create_ph_study():
    st = OLS(name="bike_plow_whigh", data=df, design_matrix=design, model="Pixel ~ Activity")

    assert st._predictors

def test_create_maps():
    st = OLS(name="bike_plow_whigh", data=df, design_matrix=design, model="Pixel ~ Activity + Intensity")
    st.create_heatmaps('pID', 2, 25, None, 'count')

    assert st._heatmaps.empty == 0

def test_aggregate_maps():
    st = OLS(name="bike_plow_whigh", data=df, design_matrix=design, model="Pixel ~ Activity + Intensity")
    st.create_heatmaps('pID', 2, 25, None, 'count')
    st.aggregate(["Intensity"])

    assert st._heatmaps_aggregates.empty == 0

def test_export_maps():
    st = OLS(name="bike_plow_whigh", data=df, design_matrix=design, model="Pixel ~ Activity + Intensity")
    st.create_heatmaps('pID', 2, 25, None, 'count')
    st.export_heatmaps("/Users/lukasgehrke/Desktop/")

    # TODO add assertion confirming export

def test_fit_OLS_paraheats():
    design = design.sort_values("pID") # okay since this is random data anyways
    st = OLS(name="test", data=df, design_matrix=design, model="Pixel ~ Activity * Intensity")
    bins = 10
    st.create_heatmaps('pID', 2, bins, None, 'count')
    st.standardize("zscore")
    st.zero_to_nan()
    st.fit()
    assert st._heatmaps_betas.shape[0] == bins**2, "size matches"
    # assert lm correct! -> give data of which i now lm result

def test_fit_GLM_paraheats():
    design = df[['pID', 'Activity']]
    design = design.sort_values("pID") # okay since this is random data anyways

    glm = GLM(name="Activity ~ Pixel", data=df, design_matrix=design, model="Activity ~ Pixel")
    bins = 5
    glm.create_heatmaps('pID', 2, bins, None, 'count')
    glm.standardize("zscore")
    glm.zero_to_nan()
    glm.fit()

    assert glm._heatmaps_betas.shape[0] == bins**2, "size matches"