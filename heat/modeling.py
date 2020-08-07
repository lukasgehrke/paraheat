import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices

from scipy import stats # remove and only use statsmodels later on

# from heat import heat

class Modeling:

    # def __init__(self, analyses):
        # self.heat = analyses

        # design: np.array
        # predictors: List[str]

        #self.design = pd.DataFrame(self.design, columns=self.predictors)

    # def make to long format and retain shape of input data to later reshape before returning results

    # fit different GLM, etc.
    @staticmethod
    def fit_lm(df, model):
        # wrapper for statsmodels OLS
        # prepare data using patsy designmatrix
        # fit using statsmodels OLS

        # simple statsmodels approach
        y, X = dmatrices(model, data=df, return_type='dataframe')
        mod = sm.OLS(y, X)
        res = mod.fit()

        # TODO for first point with 80% of data points, plot regressions diagnostics

        return res.params, res.rsquared #adjusted rsquared # pack_results !

    @staticmethod
    def fit_robust_lm(df, model):
        # wrapper for statsmodels OLS
        # prepare data using patsy designmatrix
        # fit using statsmodels OLS

        # simple statsmodels approach
        y, X = dmatrices(model, data=df, return_type='dataframe')
        mod = sm.RLM(y, X, M=sm.robust.norms.HuberT())
        res = mod.fit()

        # TODO for first point with 80% of data points, plot regressions diagnostics

        return res.params, res.bse #adjusted rsquared # pack_results !

    @staticmethod
    def fit_lm_per_bin(df, model):

        dep_var_name = [model.partition("~")[0][:-1]]

        pred_var_names = model.partition("~")[2]
        pred_var_names = pred_var_names.split("+")
        pred_var_names = [x.strip(' ') for x in pred_var_names]

        data_vars = [x for x in df.columns if x not in dep_var_name+pred_var_names]

        coeffs = {}
        rsqs = {}

        for column in data_vars:
            d_tmp = df[[column]+pred_var_names]
            d_tmp = d_tmp.rename(index = {0:dep_var_name})

            # run through every data point, check if more then 12 participant have data, if yes run analyses, else set to NaN
            # TODO run data checks here before fitlm to speed up analyses !!
            # skip points and save empty results
            # document in some way

            coeff, rsq = fit_lm(d_tmp, model)

            # add the output to the dictionary
            coeffs[column] = coeff
            rsqs[column] = rsq

        return coeffs, rsqs

    # maybe remove and use only fitlm
    def ttest(self, group1, group2):
        # maybe change to a faster implementation?
        return stats.ttest_ind(group1,group2)

        # use statsmodels implementation statsmodels.stats.weightstats.ttest_ind

        # print("ttest column: "+str(column))

        # TODO add a logger here

    def ttest_per_bin(self, df, grouping_var, cond1, cond2):

        # get a list of all columns in the dataframe without the Group column
        column_list = [x for x in df.columns if x != grouping_var]
        # create an empty dictionary
        ttest_results = {}

        # loop over column_list and execute code explained above
        for column in column_list:
            group1 = df.where(df[grouping_var]== cond1).dropna()[column]
            group2 = df.where(df[grouping_var]== cond2).dropna()[column]

            # add the output to the dictionary
            ttest_results[column] = self.ttest(group1,group2)

        res = self.pack_results(ttest_results, ["tstat", "pval"])
        return res

    def pack_results(self, results, colnames):
        res = pd.DataFrame.from_dict(results,orient='Index')
        res.columns = colnames

        return res

    # def pixels where lm fit is okay, conforming to enough data and ok to lm assumptionster