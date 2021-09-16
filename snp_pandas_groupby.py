#%%
import pandas as pd

df = pd.DataFrame({'kind': ['cat', 'dog', 'cat', 'dog'],
                   'height': [9.1, 6.0, 9.5, 34.0],
                   'weight': [7.9, 7.5, 9.9, 198.0]})
   
df
#%%
df.groupby('kind').agg(min_height=('height', 'min'), 
                               max_weight=('weight', 'max'))


# %%
df.groupby("kind").agg(f_height=('height', lambda x: 0),
                       f_weight=('weight', lambda x: 1))
# %%
