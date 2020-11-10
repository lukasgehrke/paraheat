from dataclasses import dataclass
import abc, re, itertools, warnings

import numpy as np
import scipy as sp
import pandas as pd

import modeling as modeling
#import heat.modeling as modeling

REG_MIN_SUBJECTS = 24
TTEST_MIN_SUBJECTS = 12

@dataclass
class LevelContainer:
    """class to hold data for one level of a multilevel/hierarchical/mixed-effects analyses. 
    class modeling expects a heat object, this makes it much easier to implement the wrappers to statsmodels functions 
    also, with a heatobject the ravel() and reshape() transformation are straightforward to implement
    """

    name: str
    data: pd.DataFrame
    design_matrix: pd.DataFrame
    model: str

    def __post_init__(self):
        dv_name, predictor_names = self.parse_model()
        setattr(self, '_dv', dv_name)
        setattr(self, '_predictors', predictor_names)

        cat_predictors = self.parse_categorical_predictors(predictor_names+dv_name)
        setattr(self, '_cat_predictors', cat_predictors)

        # do some check to confirm data matches design matrix
        # also allow to get a descriptive overview of data and design
        # -> aggregate summary statistics using the given model

    def parse_model(self):

        dv_name = [self.model.partition("~")[0][:-1]]
        pred_names = self.model.partition("~")[2]
        pred_names = re.split(r'[+*]', pred_names)
        pred_names = [x.strip(' ') for x in pred_names]

        return dv_name, pred_names

    def parse_categorical_predictors(self, pred_names):

        if self.design_matrix.select_dtypes(exclude=["number"]).empty:
            warnings.warn("no categorical predictor present in the design: "+self.model)
            return {}

        pred_names[:] = [x for x in pred_names if "Pixel" not in x]
        predictors = {}
        for p in pred_names:
            # predictors[p] = self.data.select_dtypes(exclude=["number"])[p].unique().tolist()
            predictors[p] = self.design_matrix.select_dtypes(exclude=["number"])[p].unique().tolist()
        return predictors


### First level (within subjects) interface and implementations
class FirstLevel(LevelContainer):

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def inspect_fit(self):
        pass

    def create_heatmaps(self, id_var, dims, bins, range, statistic):
        """TODO this should automatically run when the constructor is called and flag level_of_analyses is "first" otherwise
        it should not run automatically

        Args:
            id_var ([type]): [description]
            dims ([type]): [description]
            bins ([type]): [description]
            range ([type]): [description]
            statistic ([type]): [description]
        """

        # TODO refactor so it is more elegant for all kinds of designs
        setattr(self, '_bins', bins)
        setattr(self, '_range', range)

        if not self._cat_predictors:
            this_design = [id_var]
        else:
            this_design = list(self._cat_predictors.keys())+[id_var]
        # variable names for pandas df containing heatmaps
        # this must be adapted to different binning procedures based on edges
        bins_tmp = np.arange(bins**2).tolist()
        setattr(self, '_heatmaps', pd.DataFrame(columns=bins_tmp+this_design))

        # for id in pd.Series(self.data[id_var].unique()):
        for id in pd.Series(self.design_matrix[id_var].unique()):

            print("now making map for s " + str(id)) # TODO remove later
            # id_df = self.data.where(self.data[id_var]==id).dropna() # TODO add this to logging
            id_df = self.data.where(self.design_matrix[id_var]==id).dropna() # TODO add this to logging

            # loop through data based on model specification and make a map for all combinations
            if not self._cat_predictors:
                if id_df.empty:
                    warnings.warn("id has no data for this combination of factors in the design: " + str(self._cat_predictors))
                else:
                    if dims == 2:
                        binned_stat = sp.stats.binned_statistic_2d(id_df.X, id_df.Y, None, statistic, bins, range)
                    elif dims == 3:
                        binned_stat = sp.stats.binned_statistic_2d(id_df.X, id_df.Y, id_df.Z, statistic, bins, range)
                    pd_out = pd.concat([pd.DataFrame(binned_stat.statistic.ravel()).T, pd.DataFrame([id], columns=[id_var])], axis=1)
                    pd_out[id_var] = id
                    self._heatmaps = self._heatmaps.append(pd_out)

            else:
                for values in list(itertools.product(*list(self._cat_predictors.values()))):

                    # select pandas data
                    # TODO improve the speed here!!!
                    key_combo = dict(zip(list(self._cat_predictors.keys()), values))
                    key_combo_mask = pd.DataFrame([self.design_matrix[key] == val for key, val in key_combo.items()]).T.all(axis=1)

                    if id_df[key_combo_mask].empty:
                        warnings.warn("id has not data for this combination of factors in the design: " + str(values))
                    else:
                        if dims == 2:
                            binned_stat = sp.stats.binned_statistic_2d(id_df[key_combo_mask].X, id_df[key_combo_mask].Y, None, statistic, bins, range)
                        elif dims == 3:
                            binned_stat = sp.stats.binned_statistic_2d(id_df[key_combo_mask].X, id_df[key_combo_mask].Y, id_df[key_combo_mask].Z, statistic, bins, range)
                        pd_out = pd.concat([pd.DataFrame(binned_stat.statistic.ravel()).T, pd.DataFrame([key_combo], columns=key_combo.keys())], axis=1)
                        pd_out[id_var] = id
                        self._heatmaps = self._heatmaps.append(pd_out)

        # split data and design
        setattr(self, '_heatmaps_dmatrix', self._heatmaps[this_design])
        self._heatmaps.drop(this_design, axis=1, inplace=True)

    def aggregate(self, variables_to_aggregate):

        if self._heatmaps_dmatrix[variables_to_aggregate].empty:
            warnings.warn("variables to aggregate don't exist in design matrix: " + str(variables_to_aggregate))

        df = pd.concat([self._heatmaps, self._heatmaps_dmatrix], axis=1)
        df = df.groupby(variables_to_aggregate)

        # TODO report this to the user!
        setattr(self, '_heatmaps_aggregates', df.mean())
        return df.mean()

    def zero_to_nan(self):
        self._heatmaps.replace(0, np.nan, inplace=True)

    def nan_to_zero(self):
        self._heatmaps.replace(np.nan, 0, inplace=True)

    def standardize(self, method):
        if method == "zscore":
            self._heatmaps.apply(zscore, axis=1) # TODO add data selector

    def export_heatmaps(self, path):
        self._heatmaps.to_csv(path+'_heatmaps.csv', index=False)
        self._heatmaps_dmatrix.to_csv(path+'_design_matrix.csv', index=False)


### First level (within subjects) implementations
class OLS(FirstLevel):

    def fit(self, id_var):

        betas = {}
        ts = {}
        ps = {}
        rsqs = {}

        for column in self._heatmaps.columns:

            print("now fitting column " + str(column)) # TODO remove later

            df = pd.concat([self._heatmaps[column], self._heatmaps_dmatrix], axis=1)
            df.rename(columns={df.columns[0]: self._dv[0]}, inplace=True) # univariate case, must be changed for multivariate

            df_grouped = df.groupby(id_var).mean()
            if (df[id_var].unique().shape[0] - df_grouped[self._dv[0]].isna().sum()) > REG_MIN_SUBJECTS:
                beta, t, p, rsq = modeling.OLS_fit(df, self.model)
            else:
                beta = np.nan
                t = np.nan
                p = np.nan
                rsq = np.nan

            betas[column] = beta
            ts[column] = t
            ps[column] = p
            rsqs[column] = rsq

        setattr(self, '_heatmaps_betas', pd.DataFrame(betas).T)
        setattr(self, '_heatmaps_ts', pd.DataFrame(ts).T)
        setattr(self, '_heatmaps_ps', pd.DataFrame(ps).T)
        setattr(self, '_heatmaps_rsqs', pd.Series(rsqs).to_frame("r_squared"))

    def inspect_fit(self):
        pass

    def multcompare(self):
        pass

class GLM(FirstLevel):

    def fit(self):

        betas = {}
        ts = {}
        ps = {}

        for column in self._heatmaps.columns:
            df = pd.concat([self._heatmaps[column], self._heatmaps_dmatrix], axis = 1)
            # df.rename(columns={df.columns[0]: self._predictors[0]}, inplace = True)
            df.rename(columns={df.columns[0]: "Pixel"}, inplace = True)

            # run through every data point, check if more then 12 participant have data, if yes run analyses, else set to NaN
            # TODO run data checks here before fitlm to speed up analyses !!
            # skip points and save empty results
            # document in some way

            df_grouped = df.groupby("partId").mean()
            # if (df["pID"].unique().shape[0] - df_grouped[self._dv[0]].isna().sum()) > REG_MIN_SUBJECTS:
            beta, t, p = modeling.binomial_GLM_fit(df, self.model)
            # else:
            #     beta = np.nan
            #     t = np.nan
            #     p = np.nan
            #     rsq = np.nan

            betas[column] = beta
            ts[column] = t
            ps[column] = p

        setattr(self, '_heatmaps_betas', pd.DataFrame(betas).T)
        setattr(self, '_heatmaps_ts', pd.DataFrame(ts).T)
        setattr(self, '_heatmaps_ps', pd.DataFrame(ps).T)

    def inspect_fit(self):
        pass

    def multcompare(self):
        pass


### Second level (between subjects) implementations
# Robust method for second level (across subjects) inference,
# these classes direct the flow of analyses using functions from module "modeling" as well as "multcomp"
class Independent_T(FirstLevel):
    """
    Given heatmaps of two independent groups, compute an independent ttest using trimmed means

    Args:
        LevelContainer ([type]): [description]
    """

    def fit(self, trim=None):
        """cam only be called when create_paraheats has been run

        Args:
            main_effect ([type]): [description]
        """

        t = []
        p = []

        for column in self._heatmaps.columns:

            df = pd.concat([self._heatmaps[column], self._heatmaps_dmatrix], axis=1).dropna().groupby(self._predictors)

            if not df.count().empty and (df.count()[column] > TTEST_MIN_SUBJECTS).all():
                x = df.get_group(self._cat_predictors[self._predictors[0]][0])[column]
                y = df.get_group(self._cat_predictors[self._predictors[0]][1])[column]
                column_t, column_p = modeling.ttest_trimmed_mean(x, y, trim)
                t.append(column_t)
                p.append(column_p)
            else:
                t.append(np.nan)
                p.append(np.nan)

        setattr(self, '_t', pd.DataFrame(t).T)
        setattr(self, '_p', pd.DataFrame(p).T)

    def inspect_fit(self):
        pass
        # maybe plot distributions at some randomly selected points etc.
        # confirm ttest assumptions

    def multcompare(self):
        pass

class one_way_ANOVA(FirstLevel):
    """Given heatmaps of more than 2 independent groups, compute an (one-way) ANOVA

    Args:
        FirstLevel ([type]): [description]
    """

    def fit(self):

        F = []
        p = []
        X = []

        # create N (number of groups in predictor) arrays with reshaped heatmaps
        for level in self._cat_predictors[self._predictors[0]]:
            ix = self._heatmaps_dmatrix.index[self._heatmaps_dmatrix[self._predictors[0]] == level].tolist()
            X.append()


        # TODO this is wrong and can pipe all columns at once as a "time-freq" representation into mne functions!
        for column in self._heatmaps.columns:

            df = pd.concat([self._heatmaps[column], self._heatmaps_dmatrix], axis=1).dropna().groupby(self._predictors)

            if not df.count().empty and (df.count()[column] > TTEST_MIN_SUBJECTS).all():

                for factor_level in self._cat_predictors[self._predictors[0]]:
                    X.append(df.get_group(factor_level))

                column_f, column_p = modeling.mne_spatio_temporal_cluster_test(X)

                F.append(column_f)
                p.append(column_p)
            else:
                F.append(np.nan)
                p.append(np.nan)

        setattr(self, '_F', pd.DataFrame(F).T)
        setattr(self, '_p', pd.DataFrame(p).T)

    def inspect_fit(self):
        pass

    def multcompare(self):
        pass

class rmANOVA(FirstLevel):

    def fit(self):
        pass

    def inspect_fit(self):
        pass

    def multcompare(self):
        pass

class Paired_T(FirstLevel):

    def fit(self):
        self._grand_averages = []
        self._grand_averages_condition = []

        for factor_level in self._cat_predictors[self._predictors[0]]:
            self._grand_averages.append(reshape(self._heatmaps.loc[self._heatmaps_dmatrix[self._predictors[0]] == factor_level], self._bins, self._bins))
            self._grand_averages_condition.append(self._predictors[0] + '_' + str(factor_level))

        self._grand_averages.append(self._grand_averages[0] - self._grand_averages[1])
        self._grand_averages_condition.append('Difference')

        # TODO check how many zero elements and decide whether to run test or to return 0s
        # self._grand_averages[-1][self._grand_averages[-1]==0] = np.nan

        self._t_obs, self._clusters, self._cluster_p_values, self._H0 = modeling.mne_spatio_temporal_cluster_1samp_test(self._grand_averages[-1])

    def inspect_fit(self):
        pass




class SecondLevel(LevelContainer):
    # this takes heatmaps as inputs instead of timeseries data and then can call the same functions as firstlevel containers
    pass





### Util functions
def reshape_1d_2d(data, bins):

    if isinstance(bins, int):
        data = np.reshape(data, (bins, bins))
    else:
        data = np.reshape(data, (bins))

    return data

def zscore(pd_row):
    return pd_row - pd_row.mean()/pd_row.std(ddof=0)

def reshape(heatmap_long, binX, binY):
    return np.reshape(np.array(heatmap_long), (heatmap_long.shape[0], binX, binY))


# def pack_results_pd(results, colnames=None):
#     # res = pd.DataFrame.from_dict(results,orient='Index')
#     res = pd.DataFrame(results).T

#     if colnames is not None:
#         res.columns = colnames

#     return res