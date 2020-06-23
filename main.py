# %% 
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
file_name = Path('.') / 'data' / 'country_stats.csv'

df_raw = pd.read_csv(file_name)
df_raw.head()
# %%

df_raw.info()

# %%

df_raw.describe()

# %% 

from sklearn.model_selection import train_test_split
df_trn, df_tst = train_test_split(df_raw, test_size=0.2, random_state=42)

df = df_trn.drop(['GDP per capita','Country'],axis=1)  # regressor
df_label = df_trn['GDP per capita'].copy()

# %%

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
imputer.fit(df.values)
df_tr= imputer.transform(df.values)


# %%
