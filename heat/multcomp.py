import abc
class MCC(abc.ABC):
    # add here all the functionality of MCC in case moving away from MNE at some point in the future

    @abc.abstractmethod
    def fdr(self):
        pass

import numpy as np
import mne.stats
class MCC_MNE(MCC):

    def threshold(p_map, p_accept):
        return np.where(p_values < p_accept)[0]

    def bonferroni(p_map, p_alpha):
        return [mne.stats.bonferroni_correction(p_map, alpha=p_alpha)]

    @staticmethod
    def fdr(p_map, p_alpha):
        return [mne.stats.fdr_correction(p_map, alpha=p_alpha, method='indep')]