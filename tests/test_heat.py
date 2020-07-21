import pandas as pd
import matplotlib.image as mpimg

from heat.heat import Heat

# read data
df = pd.read_csv('/Users/lukasgehrke/Documents/temp/chatham/crd_gaze_phys-LOW_work-HIGH_all_good_s.csv')
# select a subject
s1 = df[df['pID'] == 5]
s1[['X', 'Y']]
img = mpimg.imread('/Users/lukasgehrke/Documents/temp/matb.png')

# todo instantiate rand data and img

def test_create_heat():
    h = Heat('workhigh', s1, None, img)
    assert h.name == 'workhigh'

def test_compute_2d_histogram():
    h = Heat('workhigh', s1, None, img)
    h.heatmap = h.histogram2d()
    # assert h.heatmap.shape == 2 dimensions