# Data science snippets 
## Groupby in pandas

```python
import pandas as pd

df = pd.DataFrame({'kind': ['cat', 'dog', 'cat', 'dog'],
                   'height': [9.1, 6.0, 9.5, 34.0],
                   'weight': [7.9, 7.5, 9.9, 198.0]})
   

df.groupby('kind').agg(min_height=('height', 'min'), 
                               max_weight=('weight', 'max'))
```

## Fanchart function usage
```python
fig = ut.plot_fanchart(df, title= 'Titulo de grafico',dot_name='Punto',line_name='Linea', 
                    xLabel='Eje x', yLabel='y',
                    colorListRGB=colorListRGB)
```
![Fan chart](image/FanChart.png?raw=true)

## Tree plots with graphviz

```python

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
    out_file='reg_tree_spread.dot',
    feature_names=x_tr.columns,
    rounded=True,
    filled=True
)

# dot -Tpng reg_tree.dot -o reg_tree.png
```

![Fan chart](image/reg_tree.png?raw=true)

## Profiling of code with dot files

```python
import cProfile

code_to_profile = """
fc.plot_fanchart(df, title= 'Titulo de grafico',dot_name='Punto',line_name='Linea', 
                    xLabel='Eje x', yLabel='y',
                    colorListRGB=colorListRGB)
"""
cProfile.run(code_to_profile,'profiling/code.prof')

## requires  graphviz and gprof2dot
# pip install graphviz
# pip install gprof2dot


import os
command = 'gprof2dot -f pstats profiling/code.prof | dot -Tpng -o profiling/code.png '
os.system(command)
```
![Fan chart](profiling/code.png?raw=true)

