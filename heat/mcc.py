import abc

# from heat import Analyses

class MCC(abc.ABC):
    # add here all the functionality of MCC in case moving away from MNE at some point in the future

    @abc.abstractmethod
    def permutation_ttest(self, heat: heat.Analyses) -> mask:
        pass

    @abc.abstractmethod
    def fdr(self):
        pass

# add concrete implementation of the MCC interface
import mne.stats as mnes
class MCC_MNE(MCC):

    def permutation_ttest(self):
        print('needs to be implemented')

    def fdr(self, p_map):
        return mnes.fdr_correction(p_map, .05, 'indep')
