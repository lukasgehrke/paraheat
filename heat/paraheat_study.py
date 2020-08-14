import numpy as np
import pandas as pd
from dataclasses import dataclass
import abc
import re
import itertools

# import heat.paraheat as paraheat
# import heat.modeling as modeling

import paraheat as paraheat
import modeling as modeling

@dataclass
class LevelContainer:
# class to hold data for one level of a multilevel/hierarchical/mixed-effects analyses
# class modeling expects a heat object, this makes it much easier to implement the wrappers to statsmodels functions
# also, with a heatobject the ravel() and reshape() transformation are straightforward to implement

    name: str
    data: pd.DataFrame
    design_matrix: pd.DataFrame
    model: str

    def __post_init__(self):
        object.__setattr__(self, '_model_keys', get_model_param_names(self.model))

        # do some check to confirm data matches design matrix
        # also allow to get a descriptive overview of data and design
        # -> aggregate summary statistics using the given model

    def summary_statistics(self):

        df = pd.concat([self._first_level, self.design_matrix.reset_index(drop=True)], axis=True, ignore_index=True)

        # TODO log some commom summary statistics about the model

    def create_paraheats(self, id_var, bins):

        # now a first level analyses is conducted, hence create results placeholed
        bins_tmp = np.arange(bins**2).tolist() # this must be adapted to different binning procedures based on edges
        object.__setattr__(self, '_first_level', pd.DataFrame(columns=bins_tmp+self._model_keys[1]+[id_var]))

        for id in pd.Series(self.data[id_var].unique()):

            print("now making map for s " + str(id))

            id_df = self.data.where(self.data[id_var]== id).dropna() # TODO add this to logging

            # get factor combinations
            model_keys_list = []
            for m in self._model_keys[1]:
                model_keys_list.append(id_df[m].unique().tolist())
            model_values_list = list(itertools.product(*model_keys_list))

            # loop through data based on model specification and make a map for all combinations
            for values in model_values_list:

                # select pandas data
                key_combo = dict(zip(self._model_keys[1], values))
                key_combo_mask = pd.DataFrame([id_df[key] == val for key, val in key_combo.items()]).T.all(axis=1)

                # TODO add 2d to 3d switch depending on dimension analysed
                # throw out data, or directly acccess binned_statistic to not waste memory
                tmp_ph = paraheat.ParaHeat2D(str(id), id_df[key_combo_mask], None)
                tmp_ph.heatmap = tmp_ph.binned_statistic(bins=bins)

                pd_out = pd.concat([pd.DataFrame(tmp_ph.heatmap.statistic.ravel()).T,
                    pd.DataFrame([key_combo], columns=key_combo.keys())], axis=1)
                    # pd.DataFrame.from_dict({id_var:id}
                # pd_out.columns = self._first_level.columns
                pd_out[id_var] = id

                self._first_level = self._first_level.append(pd_out)

        object.__setattr__(self, '_design_matrix_first_level', self._first_level[self._model_keys[1]+[id_var]])
        self._first_level.drop(self._model_keys[1]+[id_var], axis=1, inplace=True)

    def select_data_from_model(self, id):
        pass

        # check if interaction

    def get_param_combinations(self):
        pass

    def columnwise_zscore(self):

        self._first_level.apply(zscore, axis=1)

    @abc.abstractmethod
    def fit(self):
        pass

    # @abc.abstractmethod
    # def normalize(self):
    #     pass

    @abc.abstractmethod
    def inspect_modelfit(self):
        pass

class OLS(LevelContainer):

    def normalize(self):
        pass

    def fit(self):

        coeffs = {}
        rsqs = {}

        for column in self._first_level.columns:

            print("now fitting column " + str(column))

            # join with design matrix
            df = pd.concat([self._first_level[column], self._design_matrix_first_level], axis=1)
            df.rename(columns={df.columns[0]: self._model_keys[0][0]}, inplace=True) # univariate case, must be changed for multivariate

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

        object.__setattr__(self, '_second_level', pd.concat([coeffs, rsqs], axis=1))
        return self._second_level

    def mcc(self):
        pass

class GLM(LevelContainer):

    def fit(self):

        dep_name, pred_names = get_model_param_names(self.model)
        coeffs = {}

        for column in self._first_level.columns:

            # join with design matrix
            df = pd.concat([self._first_level[column], self.design_matrix.reset_index(drop=True)], axis=True)
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
    pred_names = re.split(r'[+*]', pred_names)
    # pred_names = pred_names.split("+")
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

def zscore(pd_row):
    return pd_row - pd_row.mean()/pd_row.std(ddof=0)