import numpy as np
import pandas as pd

# import class to be tested
from heat.plot import Plot

# generate some realistic pandas random data and write tests to generate the tests i desire :)
rng = np.random.default_rng()
df = pd.DataFrame(rng.integers(0, 100, size=(100, 3)), columns=list('ABC'))

def test_plot_builder():
    pass