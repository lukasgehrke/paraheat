import numpy as np
import pandas as pd

# import class
from heat.modeling import Modeling

# generate random data
rng = np.random.default_rng()
df = pd.DataFrame(rng.integers(0, 100, size=(100, 5)), columns=list('ABCDE'))

def test_ttest_per_bin():

    cond1 = "1"
    cond2 = "2"
    c1 = pd.Series([cond1] * int(df.shape[0]/2))
    c2 = pd.Series([cond2] * int(df.shape[0]/2))
    df["conds"] = c1.append(c2, ignore_index=True)

    # make design matrix
    # design = pd.concat([df, c1.append(c2)], axis=0)
    # design = df.insert(5, 'condition', dmatrix, True)
    # design.columns = ['A', 'B', 'C', 'D', 'design']

    m = Modeling()
    res = m.ttest_per_bin(df, "conds", cond1, cond2)

def test_OLS():

    import numpy as np
    # import statsmodels.api as sm

    spector_data = sm.datasets.spector.load(as_pandas=False)
    spector_data.exog = sm.add_constant(spector_data.exog, prepend=False)