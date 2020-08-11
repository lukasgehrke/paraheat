import numpy as np
import pandas as pd
import statsmodels.api as sm

# import class
from heat.paraheat import ParaHeat
import heat.modeling as modeling

# TODO create a dummy paraheat object from dummy data for testing purposes
# rng = np.random.default_rng()
# df = pd.DataFrame(rng.integers(0, 100, size=(100, 5)), columns=list('ABCDE'))

# for now use crd data
bike_plow_whigh = pd.read_csv('/Users/lukasgehrke/Documents/temp/chatham/crd_gaze_phys-LOW_work-HIGH_equip-Bike_all_good_s.csv')
tread_plow_whigh = pd.read_csv('/Users/lukasgehrke/Documents/temp/chatham/crd_gaze_phys-LOW_work-HIGH_equip-Treadmill_all_good_s.csv')
tmp_p = tread_plow_whigh[tread_plow_whigh['pID'] == 2]
tmp_p = tmp_p[['X', 'Y']]

# paraheat object from dummy data
h = ParaHeat('2', tmp_p, None)
h.heatmap = h.binned_statistic(bins=25) # this is tested in other test class

def test_ttest_per_bin():

    cond1 = "1"
    cond2 = "2"
    c1 = pd.Series([cond1] * int(df.shape[0]/2))
    c2 = pd.Series([cond2] * int(df.shape[0]/2))
    df["conds"] = c1.append(c2, ignore_index=True)

    # make design matrix
    # design = pd.concat([df, c1.append(c2)], axis=0)
    # design = df.insert(5, 'condition', dmatrix, True)
    # design.columns = ['A', 'B', 'C', 'D', 'design']

    m = Modeling()
    res = m.ttest_per_bin(df, "conds", cond1, cond2)

def test_OLS():

    df = sm.datasets.get_rdataset("Guerry", "HistData").data # use crd data for testing !!

    vars = ['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']
    df = df[vars]
    df = df.dropna()

    pars, rsq = Modeling.fit_OLS(df, 'Lottery ~ Literacy + Wealth')

    assert pars.shape[0] == 3, "intercept plus number of regressors is returned"

def test_RLM():

    df = sm.datasets.get_rdataset("Guerry", "HistData").data # use crd data for testing !!

    vars = ['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']
    df = df[vars]
    df = df.dropna()

    pars, bse = Modeling.fit_robust_lm(df, 'Lottery ~ Literacy + Wealth')

    assert pars.shape[0] == 3, "intercept plus number of regressors is returned"

def test_fit_lm_per_bin():

    # generate random regressor
    rng = np.random.default_rng()
    # reg = pd.DataFrame(rng.integers(0, 100, size=(crd_1s.shape[0], 1)), columns=['reg'])

    # h.heatmap.statistic.ravel().shape

    crd_1s.reset_index(drop=True, inplace=True)
    data = pd.concat([crd_1s, reg], axis=1)

    pars, bse = Modeling.fit_lm_per_bin(data, 'pixel ~ reg')

