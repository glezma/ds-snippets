# Descriptives without pandas profiling

# %%
import pandas as pd
import numpy as np

housing = pd.read_csv('data/housing.csv')
housing.info()
print('Nulls en total_bedrooms: ',housing['total_bedrooms'].isnull().sum())
housing.head()
# %%
housing.describe()

# %%

# %matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# %%

from pandas.plotting import scatter_matrix
attributes = [ 'housing_median_age','median_house_value']
scatter_matrix(housing[attributes], figsize=(12,8),hist_kwds={'bins':20}, alpha=0.1)


# %%
from scipy.stats import pearsonr
import seaborn as sns
attributes = [ 'spread','importe']
def reg_coef(x,y,label=None,color=None,**kwargs):
    ax = plt.gca()
    r,p = pearsonr(x,y)
    ax.annotate('r = {:.2f}'.format(r), xy=(0.5,0.5), xycoords='axes fraction', ha='center')
    ax.set_axis_off()

x = housing[attributes]
g = sns.PairGrid(x)
g.map_diag(sns.distplot)
g.map_lower(sns.regplot)
g.map_upper(reg_coef)

# %%
