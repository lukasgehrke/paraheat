import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy import stats

# from heat import heat

class Modeling:

    # def __init__(self, analyses):
        # self.heat = analyses

        # design: np.array
        # predictors: List[str]

        #self.design = pd.DataFrame(self.design, columns=self.predictors)

    # def make to long format and retain shape of input data to later reshape before returning results

    # fit different GLM, etc.
    def fitlm(self, data, design_matrix, model):
        pass

    def fitlm_per_bin(self, model, design, data):
        pass

        # save 2D data shape
        # ravel 2D data
        # run through every data point, check if more then 12 participant have data, if yes run analyses, else set to NaN

    def aov_per_bin(self):
        pass

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