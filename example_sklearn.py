# %%
import numpy as np
import pandas as pd

df = pd.read_csv('example_data.csv')
df.head()
# %%
df.info()
# %%
df['CODSUBSEGMENTO'].value_counts()

# %%
df.describe()

# %%
df.hist(bins=100, figsize=(20,15))

# %% imputar mediana a los valores missing
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(df_num)

# %% Partir muestra entre entrenamiento y prueba
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

train_data , test_data = train_test_split(df,test_size=0.2,random_state=42)


# %% Making a continuous value cathegorical using pandas cut function

df['CAT_MONETARY'] = pd.cut(df['MONETARY'],bins=[0,1000,5000,50000,np.inf],labels=[1,2,3,4])
df['CAT_MONETARY'].hist()

# %% Using stratifiedShuffleSplit to get stratified sampling 
# using the cathegorized series

split = StratifiedShuffleSplit(n_splits=1,test_size=0.3,random_state=42)

for train_index, test_index in split.split(df, df['CAT_MONETARY']):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]
    


# %% Manualy doing sampling

def sample(data,testProp):
    n = len(data)
    permutedIndexes = np.random.permutation(n)
    partitionSize = int(testProp*n)
    trainIndex = permutedIndexes[:partitionSize]
    testIndex = permutedIndexes[partitionSize:]
    return data.iloc[trainIndex], data.iloc[testIndex]


# %%
