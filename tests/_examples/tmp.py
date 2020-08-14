# from scipy.stats import gaussian_kde
# import matplotlib.pyplot as plt

# # fit an array of size [Ndim, Nsamples]
# mean = [0, 0]
# cov = [[1, 1], [1, 2]]
# x, y = np.random.multivariate_normal(mean, cov, 10000).T
# data = np.vstack([x, y])
# kde = gaussian_kde(data)

# # evaluate on a regular grid
# xgrid = np.linspace(-3.5, 3.5, 40)
# ygrid = np.linspace(-6, 6, 40)
# Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
# Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

# # Plot the result as an image
# plt.imshow(Z.reshape(Xgrid.shape),
#            origin='lower', aspect='auto',
#            extent=[-3.5, 3.5, -6, 6],
#            cmap='Blues')
# cb = plt.colorbar()
# cb.set_label("density")

from patsy import dmatrices, dmatrix, demo_data

# read data -> emulate this kind of data with a random generated dataset
df = pd.read_csv('/Users/lukasgehrke/Documents/temp/chatham/LG_data_crdPhase1/df_scenario1.csv')
data = df[['X', 'Y']]
design = df[['pID' ,'Activity', 'Workload', 'Intensity', 'GTLX']]
design.head()
# design.drop_duplicates(inplace=True)
df_gtlx = df.groupby(['X', 'Y', 'pID', 'Activity', 'Workload']).mean()
df_gtlx.head()

p = dmatrices("X, Y ~ Activity * Workload", data)