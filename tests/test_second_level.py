import pandas as pd
import numpy as np

# import class
from heat.first_level import OLS, Independent_T, one_way_ANOVA, Paired_T
# df = pd.read_csv('/Users/lukasgehrke/Documents/temp/chatham/LG_data_crdPhase1/df_scenario1.csv')

# generate some realistic random data
participants = 60
size = 100
X = []
Y = []
pID = []
three_level_between_factor = []

for p in range(participants):
    if p < 20:
        level = ["A"]
    elif p >= 20 & p < 40:
        level = ["B"]
    else:
        level = ["C"]

    X = X+np.random.random(size).tolist()
    Y = Y+np.random.random(size).tolist()
    pID = pID+([p]*size)
    three_level_between_factor = three_level_between_factor+(level*size)

df = pd.DataFrame.from_dict({"X":X, "Y": Y, "pID": pID, "three_level_between_factor": three_level_between_factor})

data = df[["X", "Y"]]
design = df[["pID" , "three_level_between_factor"]]

def test():
    pass


def test_ttest_trimmed_mean():

    st = Independent_T(name="bike_plow_whigh", data=df, design_matrix=design, model="Pixel ~ three_level_between_factor")
    st.create_paraheats('pID', 2, 25, None, 'count')
    st.standardize("zscore")
    st.zero_to_nan()
    st.fit(.1)

    # TODO add assertion

def test_anova():
    st = one_way_ANOVA(name="bike_plow_whigh", data=df, design_matrix=design, model="Pixel ~ three_level_between_factor")
    st.create_paraheats('pID', 2, 25, None, 'count')
    st.standardize("zscore")
    st.zero_to_nan()
    st.fit()

    # TODO add assertion

def test_Paired_T():
    st = Paired_T(name="bike_plow_whigh", data=df, design_matrix=design, model="Pixel ~ three_level_between_factor")
    st.create_paraheats('pID', 2, 25, None, 'count')
    st.standardize("zscore")
    st.zero_to_nan()
    st.fit()
