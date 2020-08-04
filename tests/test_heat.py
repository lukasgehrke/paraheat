import pandas as pd

# import class
from heat.heat import Heat

# TODO generate random data

df = pd.read_csv('/Users/lukasgehrke/Documents/temp/chatham/crd_gaze_phys-LOW_work-HIGH_all_good_s.csv')
# select a subject
s5 = df[df['pID'] == 5]
s5 = s5[['X', 'Y']]

def test_create_heat():
    h = Heat('workhigh', s5, None)
    assert h.name == 'workhigh'

def test_compute_binned_statistic():
    bins = 100
    h = Heat('workhigh', s5, None)
    h.heatmap = h.binned_statistic(bins=bins)
    assert h.heatmap.statistic.size == bins**2