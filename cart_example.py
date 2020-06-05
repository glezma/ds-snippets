# %% Carga bibliotecas
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
# %% Partir muestra entre entrenamiento y prueba
m = 10000
X1=6*np.random.rand(m)-3
X2=6*np.random.rand(m)-3
Y = 0.5*X1**2+X1+X2+2+np.random.randn(m)

df = pd.DataFrame()
df['X1'] = X1
df['X2'] = X2
df['Y'] = Y
df
# %% Partir muestra entre entrenamiento y prueba

train_data , test_data = train_test_split(df,test_size=0.2,random_state=42)

tree_reg = DecisionTreeRegressor(max_depth=4)
x_tr = train_data[['X1','X2']]
y_tr = train_data['Y']
tree_reg.fit(x_tr,y_tr)
x_tst = test_data[['X1','X2']]
y_tst = test_data['Y']

# %%
from sklearn.tree import export_graphviz

export_graphviz(
    tree_reg,
    out_file='reg_tree4.dot',
    feature_names=x_tr.columns,
    rounded=True,
    filled=True
)

# dot -Tpng reg_tree.dot -o reg_tree.png

# %% Prediccion
pred = tree_reg.predict(test_data[['X1','X2']])
scoretr =tree_reg.score(x_tr,y_tr) 
tree_reg.feature_importances_
# %%
