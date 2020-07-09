# %% Import libraries
import pandas as pd
import numpy as np
from pathlib import Path
# %% Import data
filename = 'housing.csv'
filepath = Path('.' ) / 'data' / filename
df = pd.read_csv(filepath)
# %% Total histogram
df["median_income"].hist()
# %% Data basic info
df.info()
# %% Data describe
df.describe()
# %% Binned histogram value counts

# df["income_cat"] = pd.cut(df["median_income"],
#                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
#                                labels=[1, 2, 3, 4, 5])
# Transform numerical to cat
df["income_cat"] = pd.qcut(df['median_income'],q=[0, 0.25, 0.5,0.75, 1],labels=['q1','q2','q3','q4'])
df["income_cat"].value_counts()
# Concatenating cross variables combinations
combine_list = ['ocean_proximity', 'income_cat']
combine_list.reverse()
def combine(df,cl):
    ''' cl should have at least 2 elements
    '''
    r = df[cl[0]]
    for elem in cl[1:]:
        r = r.str.cat(df[elem],sep=',')
    return r

df['combination'] =  combine(df,combine_list)


# %% Pure cat histogram
df["income_cat"].hist()

# %%  Stratified sampling relative frequencies
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

strat_test_set["income_cat"].value_counts() / len(strat_test_set)
# %%  Total relative frequencies
df["income_cat"].value_counts() / len(df)# %%

# %% Pure random sampling test and comparison
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)



compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(df),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
compare_props

# %%
