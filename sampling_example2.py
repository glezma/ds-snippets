# %% Import libraries
import pandas as pd
import numpy as np
from pathlib import Path
# %% Import data
filename = 'leads420.csv'
filepath = Path('.' ) / 'data' / filename
df = pd.read_csv(filepath)
df.head()
# %% Data basic info
df.info()
# %% Data describe
df.describe()
# %% Total histogram
df["SCORE_BHV"].hist()

# %% Total histogram
df["MTOFINALOFERTADOSOL"].hist(bins=20)

# %% Binned histogram value counts
cut_div = np.append(np.arange(0,220_000,20_000),[np.inf]).tolist()
labels = [str(x) for x in cut_div[0:-1] ]
df["monto_bucket"] = pd.cut(df["MTOFINALOFERTADOSOL"],
                               bins=cut_div,
                               labels=labels)
df["monto_bucket"].value_counts()
# %% Transform numerical to cat

# df["income_cat"] = pd.qcut(df['median_income'],q=[0, 0.25, 0.5,0.75, 1],labels=['q1','q2','q3','q4'])
# df["income_cat"].value_counts()

# %% Concatenating cross variables combinations
combine_list = ['monto_bucket', 'RNG_SCOREBHV']
combine_list.reverse()
def combine(df,cl):
    ''' cl should have at least 2 elements
    '''
    r = df[cl[0]]
    for elem in cl[1:]:
        r = r.str.cat(df[elem],sep=',')
    return r

df['combination'] =  combine(df,combine_list)
df.head()



# %%  Stratified sampling relative frequencies
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
for train_index, test_index in split.split(df, df["combination"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

strat_test_set["combination"].value_counts() / len(strat_test_set)

# %% Strat sampling 2
strat_train_set=strat_train_set.reset_index()
split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=43)
for train_index2, test_index2 in split.split(strat_train_set, strat_train_set["combination"]):
    strat_train_set2 = strat_train_set.loc[train_index2]
    strat_test_set2 = strat_train_set.loc[test_index2]

strat_test_set2["combination"].value_counts() / len(strat_test_set2)
# %%  Total relative frequencies
df["combination"].value_counts() / len(df)# %%

# %% Pure random sampling test and comparison
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size=0.33, random_state=42)

def comb_prop(data):
    return data["combination"].value_counts() / len(data)



compare_props = pd.DataFrame({
    "Overall": comb_prop(df),
    "Stratified": comb_prop(strat_test_set),
    "Stratified2": comb_prop(strat_test_set2),
    "Stratified3": comb_prop(strat_train_set2),
    "Random": comb_prop(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. Split 1 %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
compare_props["Strat. Split 2 %error"] = 100 * compare_props["Stratified2"] / compare_props["Overall"] - 100
compare_props["Strat. Split 3 %error"] = 100 * compare_props["Stratified3"] / compare_props["Overall"] - 100

compare_props

# %%
strat_test_set['split'] = 1
strat_test_set2['split'] = 2
strat_train_set2['split'] = 3

# %%
list_concat =[strat_test_set, strat_test_set2,strat_train_set2]
df_f = pd.concat(list_concat)


# %%
group1 = df_f[['split','RNG_SCOREBHV','monto_bucket','CODCLAVECIC']].groupby(['split','RNG_SCOREBHV']).count()
state_pcts = group1.groupby(level=0).apply(lambda x:  100 * x / float(x.count()))

# %%
df_f.to_csv('output.csv')

# %%
