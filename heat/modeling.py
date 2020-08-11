import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

from patsy import dmatrices
from scipy import stats # remove and only use statsmodels later on

# fit different GLM, etc.
def OLS_fit(df, model):
    # wrapper for statsmodels OLS
    # prepare data using patsy designmatrix
    # fit using statsmodels OLS

    # simple statsmodels approach
    # y, X = dmatrices(model, data=df, return_type='dataframe')

    # design = sm.tools.categorical(df, drop=True)

    mod = smf.ols(formula=model, data=df)
    res = mod.fit()

    # TODO for first point with 80% of data points, plot regressions diagnostics

    return res.params, res.rsquared #adjusted rsquared # pack_results !

def robust_OLS_fit(df, model):
    # wrapper for statsmodels OLS
    # prepare data using patsy designmatrix
    # fit using statsmodels OLS

    # simple statsmodels approach
    y, X = dmatrices(model, data=df, return_type='dataframe')
    mod = sm.RLM(y, X, M=sm.robust.norms.HuberT())
    res = mod.fit()

    # TODO for first point with 80% of data points, plot regressions diagnostics

    return res.params, res.bse #adjusted rsquared # pack_results !

def binomial_GLM_fit(df, model):

    return smf.glm(formula=model, data=df, family=sm.families.Binomial()).fit().params

# maybe remove and use only fitlm
def ttest(self, group1, group2):
    # maybe change to a faster implementation?
    return stats.ttest_ind(group1,group2)

    # use statsmodels implementation statsmodels.stats.weightstats.ttest_ind

    # print("ttest column: "+str(column))

    # TODO add a logger here