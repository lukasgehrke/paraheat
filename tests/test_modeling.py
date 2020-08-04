import numpy as np
import pandas as pd

# import class
from heat.modeling import Modeling

# generate random data
rng = np.random.default_rng()
df = pd.DataFrame(rng.integers(0, 100, size=(100, 5)), columns=list('ABCDE'))

def test_ttest_per_bin():

    c1 = pd.Series(["1"] * int(df.shape[0]/2))
    c2 = pd.Series(["2"] * int(df.shape[0]/2))
    df["conds"] = c1.append(c2, ignore_index=True)

    # make design matrix
    # design = pd.concat([df, c1.append(c2)], axis=0)
    # design = df.insert(5, 'condition', dmatrix, True)
    # design.columns = ['A', 'B', 'C', 'D', 'design']

    m = Modeling()
    res = m.ttest_per_bin(df, 'conds')

def test_compute_binned_statistic():
    bins = 100
    h = Heat('workhigh', s5, None)
    h.heatmap = h.binned_statistic(bins=bins)
    assert h.heatmap.statistic.size == bins**2