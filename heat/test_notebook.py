# %%

# read data for tests
import pandas as pd
df = pd.read_csv('/Users/lukasgehrke/Documents/temp/chatham/LG_data_crdPhase1/df_scenario1_random_sample.csv')
# df = df.sample(100000) # select random rows for faster debugging
# df.to_csv('/Users/lukasgehrke/Documents/temp/chatham/LG_data_crdPhase1/df_scenario1_random_sample.csv', index=False)

data = df[['X', 'Y']]
design = df[['pID' ,'Activity', 'Workload', 'Intensity', 'GTLX']]
design.head()

# %%
import pandas as pd
import numpy as np

participants = 20
size = 100
X = []
Y = []
pID = []
some_cat_between_factor = []

for p in range(participants):
    if p < participants/2:
        level = ["A"]
    else:
        level = ["B"]
    X = X+np.random.random(size).tolist()
    Y = Y+np.random.random(size).tolist()
    pID = pID+([p]*size)
    some_cat_between_factor = some_cat_between_factor+(level*size)

data = {"X":X, "Y": Y, "pID":pID, "some_cat_between_factor": some_cat_between_factor}
df = pd.DataFrame.from_dict(data)




this_df = pd.DataFrame.from_dict(d)
# %%
import pandas as pd
import numpy as np

size = 100
d = {'X': np.random.random(size), 'Y': np.random.random(size), 'pID':[4]*size, 'some_cat_between_factor': ["A"]*size}
this_df = pd.DataFrame.from_dict(d)

# %%
