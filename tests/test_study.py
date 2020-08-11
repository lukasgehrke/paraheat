import statsmodels.api as sm
import statsmodels.formula.api as smf

import pandas as pd

# import class
from heat.paraheat import ParaHeat
from heat.paraheat_study import OLS, GLM
import heat.paraheat_study as ph_study

# for now use crd data // later put random data
bike_plow_whigh = pd.read_csv('/Users/lukasgehrke/Documents/temp/chatham/crd_gaze_phys-LOW_work-HIGH_equip-Bike_all_good_s.csv')
tread_plow_whigh = pd.read_csv('/Users/lukasgehrke/Documents/temp/chatham/crd_gaze_phys-LOW_work-HIGH_equip-Treadmill_all_good_s.csv')
df = bike_plow_whigh.append(tread_plow_whigh, ignore_index=True)

# create the design matrix
# get unique pIDs per locomotion
pID_bike = pd.Series(bike_plow_whigh['pID'].unique())
pID_tread = pd.Series(tread_plow_whigh['pID'].unique())
pID = pID_bike.append(pID_tread)
# define contrast
bike = pd.Series(['bike'] * pID_bike.shape[0])
tread = pd.Series(['tread'] * pID_tread.shape[0])
locomotion = bike.append(tread)
design = pd.concat([pID, locomotion], axis=1)
design.columns = ['pID', 'Locomotion']
# design.head()

def test_create_ph_study():
    st = OLS(name="bike_plow_whigh", data=df, design_matrix=design, model="pixel ~ Locomotion", first_level=[], second_level=[])

    assert st.name == "bike_plow_whigh"

def test_create_paraheats():
    st = OLS(name="bike_plow_whigh", data=df, design_matrix=design, model="pixel ~ Locomotion", first_level=[], second_level=[])
    st.create_paraheats('pID', 25)

    assert len(st.first_level) == df["pID"].unique().shape[0]

def test_fit_OLS_paraheats():
    st = OLS(name="bike_plow_whigh", data=df, design_matrix=design, model="pixel ~ Locomotion", first_level=[], second_level=[])
    bins = 5
    st.create_paraheats('pID', bins)
    st.fit()

    assert st.second_level.shape[0] == bins**2, "size matches"
    # assert lm correct! -> give data of which i now lm result

def test_fit_GLM_paraheats():

    glm = GLM(name="Locomotion ~ Pixel", data=df, design_matrix=design, model="Locomotion ~ Pixel", first_level=[], second_level=[])
    bins = 5
    glm.create_paraheats('pID', bins)
    glm.fit()

    assert glm.second_level.shape[0] == bins**2, "size matches"