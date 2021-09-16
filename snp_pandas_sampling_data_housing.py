# 1) FULL DATA EXPLORATION
# %% Import data

import numpy as np
np.set_printoptions( suppress=True)
import pandas as pd
from pathlib import Path
filepath = Path('.') / 'data'/ 'housing.csv'
df_raw = pd.read_csv(filepath)
df_raw.head()

# %% Summary of nulls

df_raw.info()
print('\n List of nulls')
df_raw.isnull().sum()

# %% Describe

df_raw.describe()

# %% Get list of categorical variables

s = (df_raw.dtypes == 'object')
cat_vars = list(s[s].index)

print("Categorical variables:")
print(cat_vars)
for var in cat_vars:
    print('\n Abs freqs')
    print(df_raw[var].value_counts())
    print('\n Relative freqs')
    print(df_raw[var].value_counts()/len(df_raw[var]))

# %% Histogram (Not necessary to drop cat vars)

df_raw.hist(bins=100, figsize=(20,15))

# 2) SEPARAR TRAINING AND TESTING DATASET
# %% Pure random sampling method

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
df_trn_rs , df_tst_rs = train_test_split(df_raw,test_size=0.2,random_state=42)


# %% Stratified sampling preparation
import matplotlib.pyplot as plt
target = 'median_income'
target_cat = 'income_cat'
df_raw[target_cat] = pd.cut(df_raw[target],
                        bins=[0 , 1.5 , 3, 4.5, 6, np.inf],
                        labels=[1, 2, 3, 4, 5])
fig = df_raw[target_cat].hist()
plt.title(target_cat)
# %% Stratified Sampling
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index, test_index in split.split(df_raw, df_raw[target_cat]):
    df_trn = df_raw.loc[train_index]
    df_tst = df_raw.loc[test_index]

# %% Pure random sampling method

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
df_trn_rs , df_tst_rs = train_test_split(df_raw,test_size=0.2,random_state=42)



# %% comparison
def income_cat_proportions(data):
    return data[target_cat].value_counts() / len(data)


compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(df_raw),
    "Stratified": income_cat_proportions(df_trn),
    "Random": income_cat_proportions(df_trn_rs),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
compare_props
# %% Drop target cat variable
for set_ in (df_trn, df_tst):
    set_.drop(target_cat, axis=1, inplace=True)

# %% Visualization

df = df_trn.copy()
df.head()

# %% Function for saving figure to png
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join('image', fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# %% Simple geolocation graph

df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
save_fig("better_visualization_plot")


# %% color geolocation graph 

df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=df["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
save_fig("housing_prices_scatterplot")

# %%  Correlation 
corr_matrix = df.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)

# %% Corr matrix
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
from pandas.plotting import scatter_matrix
scatter_matrix(df[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")

# %% Exploring new "per" variables

df["rooms_per_household"] = df["total_rooms"]/df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"]/df["total_rooms"]
df["population_per_household"]=df["population"]/df["households"]
corr_matrix = df.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# %% PREPARING DATA FOR ML

df = df_trn.drop("median_house_value", axis=1) # drop labels for training set
df_labels = df_trn["median_house_value"].copy()

# %% Treatment of missing values


# sample_incomplete_rows = df[df.isnull().any(axis=1)].head()  
# df.dropna(subset=["total_bedrooms"])    # option 1
# # df.drop("total_bedrooms", axis=1)       # option 2
# median = df["total_bedrooms"].median()    # option 3
# df["total_bedrooms"].fillna(median,inplace=True)

# imputar mediana a los valores missing

from sklearn.impute import SimpleImputer
df_num = df.select_dtypes(include=[np.number])
df_cat = df.select_dtypes(include=[np.object])

imputer = SimpleImputer(strategy="mean")
imputer.fit(df_num)  # 1) Estimacion
imputer.statistics_  # df_num.median().values
imputer.strategy
data_num_tr = imputer.transform(df_num)  # 2) Transform yields np.array

df_num = pd.DataFrame(data=data_num_tr,
                            columns=df_num.columns,
                            index=df_num.index)
df_num

# %% Ordinal encoders

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
df_cat_encoded = ordinal_encoder.fit_transform(df_cat)
print(df_cat_encoded[:10])
ordinal_encoder.categories_
# %% OH Encoder

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
df_cat_1hot = cat_encoder.fit_transform(df_cat)
cat_encoder.categories_
df_cat_1hot.toarray()


# %% Custom Transformers

from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do
        
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(df.values)


# %% Transformation Pipelines

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

df_num_tr = num_pipeline.fit_transform(df_num)


# %% Column transformer

from sklearn.compose import ColumnTransformer

num_attribs = list(df_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

df_prepared = full_pipeline.fit_transform(df)

# %%

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(df_prepared, df_labels)


# %%

# let's try the full preprocessing pipeline on a few training instances
some_data = df.iloc[:5]
some_labels = df_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))

print("Labels:", list(some_labels))
# %%

from sklearn.metrics import mean_squared_error

df_predictions = lin_reg.predict(df_prepared)
lin_mse = mean_squared_error(df_labels, df_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

# %%

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(df_prepared, df_labels)

df_predictions = tree_reg.predict(df_prepared)
tree_mse = mean_squared_error(df_labels, df_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# %%
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, df_prepared, df_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)

# %%

lin_scores = cross_val_score(lin_reg, df_prepared, df_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

# %%

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(df_prepared, df_labels)
df_predictions = forest_reg.predict(df_prepared)
forest_mse = mean_squared_error(df_labels, df_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse



# %% 

from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, df_prepared, df_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

# %% GRID SEARCH (this might take a while)

from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(df_prepared, df_labels)


# %% best hyperparameter combination found
grid_search.best_params_

# %% Best estimator object

grid_search.best_estimator_

# %% List showing each hyperparam with score
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# %%  Dataframe showing each split in the cv

pd.DataFrame(grid_search.cv_results_)


# %% Finding feature importance
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
#cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
# %% Best model rmse


final_model = grid_search.best_estimator_

X_test = df_tst.drop("median_house_value", axis=1)
y_test = df_tst["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse

# %% Confidence interval
from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))
m = len(squared_errors)
mean = squared_errors.mean()
tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)

# %%
