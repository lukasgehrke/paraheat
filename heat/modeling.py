import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy as sp
from mne.stats import f_mway_rm, spatio_temporal_cluster_test, spatio_temporal_cluster_1samp_test

from patsy import dmatrices

# fit different GLM, etc.
def OLS_fit(df, model):
    # wrapper for statsmodels OLS

    mod = smf.ols(formula=model, data=df)
    res = mod.fit()

    # TODO for first point with 80% of data points, plot regressions diagnostics

    return res.params, res.tvalues, res.pvalues, res.rsquared #adjusted rsquared # pack_results !

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

    mod = smf.glm(formula=model, data=df, family=sm.families.Binomial())
    res = mod.fit()

    return res.params, res.tvalues, res.pvalues

def ttest_trimmed_mean(x, y, trim=None):

    if trim is not None:
        x = sp.stats.trimboth(x, proportiontocut=trim)
        y = sp.stats.trimboth(y, proportiontocut=trim)

    t, p = sp.stats.ttest_ind(x, y)
    return t, p

def mne_spatio_temporal_cluster_test(X, **kwargs):

    threshold_tfce = dict(start=0, step=0.2)
    F_obs, _, p_values, _ = spatio_temporal_cluster_test(X, n_permutations=1000,
                                        threshold=threshold_tfce, tail=1,
                                        n_jobs=1, buffer_size=None,
                                        connectivity=None)

    return F_obs, p_values

def mne_spatio_temporal_cluster_1samp_test(X, **kwargs):

    threshold_tfce = dict(start=0, step=0.2)

    # from MNE
    # X : array, shape (n_observations, n_times, n_vertices)
    #         The data to be clustered. The first dimension should correspond to the
    #         difference between paired samples (observations) in two conditions.
    # T_obs, _, p_values, _ = spatio_temporal_cluster_1samp_test(X, n_permutations=1000,
    #                                     threshold=threshold_tfce, tail=1,
    #                                     n_jobs=1, buffer_size=None,
    #                                     connectivity=None)

    T_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(X, n_permutations=1000,
                                        threshold=None, tail=1,
                                        n_jobs=1, buffer_size=None,
                                        connectivity=None, out_type='mask')

    return T_obs, clusters, cluster_p_values, H0

def two_way_rmANOVA(df, model, factor_levels, effect_labels):

    # TODO use spatio_temporal clust to pool in function to stat_fun
    f, p = f_mway_rm(data, factor_levels, effects="A*B")

    # TODO check wether names of results can be overwritten with effect_labels
    return f, p

