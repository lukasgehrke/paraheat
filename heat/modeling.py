from typing import List

import numpy as np
import pandas as pd
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
    def fitlm_per_bin(self, model, design, data):
        pass

        # save 2D data shape
        # ravel 2D data
        # run through every data point, check if more then 12 participant have data, if yes run analyses, else set to NaN


    def aov_per_bin(self):
        pass

    def ttest_per_bin(self, df, grouping_var, cond1, cond2):

        # get a list of all columns in the dataframe without the Group column
        column_list = [x for x in df.columns if x != grouping_var]
        # create an empty dictionary
        t_test_results = {}
        # loop over column_list and execute code explained above
        for column in column_list:
            group1 = df.where(df[grouping_var]== cond1).dropna()[column]
            group2 = df.where(df[grouping_var]== cond2).dropna()[column]
            # add the output to the dictionary
            t_test_results[column] = stats.ttest_ind(group1,group2)
            print("ttest column: "+str(column))
        results_df = pd.DataFrame.from_dict(t_test_results,orient='Index')
        results_df.columns = ['statistic','pvalue']

        # TODO add a logger here

        return results_df