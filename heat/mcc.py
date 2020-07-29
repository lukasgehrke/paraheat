import abc

from heat import Analyses

class MCC(abc.ABC):

    @abc.abstractmethod
    def permutation_ttest(self, heat: heat.Analyses) -> mask:
        pass


# add concrete implementation of the MCC interface
class MCC_MNE(MCC):

    def permutation_ttest(self):
        print('needs to be implemented')