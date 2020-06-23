
# %% 
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
full_file = Path('.') / 'data' / 'more_data.csv'
df_raw = pd.read_csv(full_file)
df_raw.head()
# %% Summary of nulls
df_raw.info()
print('\n List of nulls')
df_raw.isnull().sum()
# %% Describe
print(df_raw.describe())
print('\nCategorical')
for item in df_raw.select_dtypes(include=[np.object]).columns: print(df_raw[item].value_counts())
# %%
df_raw.hist()
# %% Train and test sample split - Separation of target from data
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
df_trn , df_tst = train_test_split(df_raw,test_size=0.2,random_state=42)

df = df_trn.drop(["LifeSat"], axis=1)
df_label = df_trn.loc[:,["LifeSat"]].copy()
print('df')
print(df.columns)
print('df_label')
df_label.columns

# %% Missing (Manual w/o pipelines)
from sklearn.impute import SimpleImputer
df_num = df.select_dtypes(include=[np.number])
df_cat = df.select_dtypes(include=[np.object])

imputer = SimpleImputer(strategy="median")
imputer.fit(df_num)  # 1) Estimacion
imputer.statistics_  # same as df_num.median().values
imputer.strategy
data_num_tr = imputer.transform(df_num)  # 2) Transform returns np.array

df_num = pd.DataFrame(data=data_num_tr,
                            columns=df_num.columns,
                            index=df_num.index)
# Summary of nulls

print('\n List of nulls')
df_num.isnull().sum() 
# %% Standarization w/o pipelines
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
npa_num_trnf = scaler.fit_transform(df_num.values)

df_num_trnsf = pd.DataFrame(data=npa_num_trnf,
                            columns=df_num.columns,
                            index=df_num.index)

df_num_trnsf
plt.plot(df_num_trnsf.index,df_num_trnsf['HealthI'],'o',df_num['HealthI'],'o')


# %% imputer + standardizer with pipelines for numeric values
from sklearn import pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()

pipeline_num = pipeline.Pipeline([('imputer', imputer),
                                ('scal', scaler)])
df_trans = pipeline_num.fit_transform(df_num.values)

# %% transformer for categorical without pipeline
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
df_cat_1hot = cat_encoder.fit_transform(df_cat)
cat_encoder.categories_
df_cat_1hot.toarray()
# pipeline.Pipeline([('cat_treat', OneHotEncoder())])

# %% Column transformer

from sklearn.compose import ColumnTransformer

num_attribs = list(df_num)
cat_attribs = list(df_cat)

full_pipeline = ColumnTransformer([
        ("num", pipeline_num, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

df_prepared = full_pipeline.fit_transform(df)
df_prepared


# %% Pipeline
import sklearn as skl
# from sklearn.metrics import mean_squared_error
from sklearn import pipeline

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

model = LinearRegression()
model = KNeighborsRegressor(n_neighbors=3)
model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(df_prepared, df_label.values)
predicted = model.predict(df_prepared)

plt.plot(predicted,'o')
plt.plot(df_label.values,'.')
plt.xlabel("obs")
plt.legend(['estimado','real'])
plt.show()

from sklearn.metrics import mean_squared_error
lin_mse = mean_squared_error(predicted, df_label.values)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

# %%

# %% 

from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(model, df_prepared, df_label,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
display_scores(forest_rmse_scores)

# %% GRID SEARCH (this might take a while)

from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 9 (3×3) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]


# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(model, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True, n_jobs=4)
grid_search.fit(df_prepared, df_label)



# %%
grid_search.best_params_

# %%
# %% List showing each hyperparam with score
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# %%
pd.DataFrame(grid_search.cv_results_)

# %%
