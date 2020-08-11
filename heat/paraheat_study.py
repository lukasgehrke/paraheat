import numpy as np
import pandas as pd
from dataclasses import dataclass
import abc

from heat.paraheat import ParaHeat
import heat.modeling as modeling

@dataclass
class LevelContainer:
# class to hold data for one level of a multilevel/hierarchical/mixed-effects analyses
# class modeling expects a heat object, this makes it much easier to implement the wrappers to statsmodels functions
# also, with a heatobject the ravel() and reshape() transformation are straightforward to implement

    name: str
    data: pd.DataFrame
    design_matrix: pd.DataFrame
    model: str
    first_level: [] # mutable
    second_level: []

    # do some check to confirm data matches design matrix
    # also allow to get a descriptive overview of data and design
    # -> aggregate summary statistics using the given model

    def summary_statistics(self):

        df = pd.concat([self.first_level, self.design_matrix.reset_index(drop=True)], axis=True, ignore_index=True)

        # TODO log some commom summary statistics about the model

    def create_paraheats(self, id_var, bins):
        # loop over unique id var

        unique_id_var = pd.Series(self.data[id_var].unique())
        for id in unique_id_var:

            id_df = self.data.where(self.data[id_var]== id).dropna() # TODO add this to logging
            id_df = id_df.drop([id_var], axis=1)

            ph_id = ParaHeat(str(id), id_df, None)
            ph_id.heatmap = ph_id.binned_statistic(bins=bins)

            self.first_level.append(pd.Series(ph_id.heatmap.statistic.ravel())) # auslagern in 2d to 1d and 3d to 2d functions

        self.first_level = pd.DataFrame(self.first_level)

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def inspect_modelfit(self):
        pass

    @abc.abstractmethod
    def mcc(self):
        pass

class OLS(LevelContainer):

    def fit(self):

        dep_name, pred_names = get_model_param_names(self.model)
        coeffs = {}
        rsqs = {}

        for column in self.first_level.columns:

            # join with design matrix
            df = pd.concat([self.first_level[column], self.design_matrix[pred_names].reset_index(drop=True)], axis=True, ignore_index=True)
            df.columns = dep_name+pred_names

            # run through every data point, check if more then 12 participant have data, if yes run analyses, else set to NaN
            # TODO run data checks here before fitlm to speed up analyses !!
            # skip points and save empty results
            # document in some way
            # TODO logic how to select the model

            coeff, rsq = modeling.OLS_fit(df, self.model)

            coeffs[column] = coeff
            rsqs[column] = rsq

        coeffs = pack_results_pd(coeffs)
        rsqs = pack_results_pd(rsqs, ['r_squared'])

        self.second_level = pd.concat([coeffs, rsqs], axis=1)
        return self.second_level

    def mcc(self):
        pass

class GLM(LevelContainer):

    def fit(self):

        dep_name, pred_names = get_model_param_names(self.model)
        coeffs = {}

        for column in self.first_level.columns:

            # join with design matrix
            df = pd.concat([self.first_level[column], self.design_matrix.reset_index(drop=True)], axis=True)
            df.rename(columns={df.columns[0]: pred_names[0]}, inplace = True)

            # run through every data point, check if more then 12 participant have data, if yes run analyses, else set to NaN
            # TODO run data checks here before fitlm to speed up analyses !!
            # skip points and save empty results
            # document in some way

            coeff = modeling.binomial_GLM_fit(df, self.model)
            coeffs[column] = coeff

        self.second_level = pack_results_pd(coeffs)
        return self.second_level

    def mcc(self):
        pass

class LMM(LevelContainer):
    pass

class rmANOVA(LevelContainer):
    pass



### Util functions
def get_model_param_names(model):

    # extract model
    dep_name = [model.partition("~")[0][:-1]]
    pred_names = model.partition("~")[2]
    pred_names = pred_names.split("+")
    pred_names = [x.strip(' ') for x in pred_names]

    return dep_name, pred_names

def reshape_1d_2d(data, bins):

    if isinstance(bins, int):
        data = np.reshape(data, (bins, bins))
    else:
        data = np.reshape(data, (bins))

    return data

def pack_results_pd(results, colnames=None):
    res = pd.DataFrame.from_dict(results,orient='Index')

    if colnames is not None:
        res.columns = colnames

    return res