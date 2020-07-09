# %%
import numpy as np
import pandas as pd
from pathlib import Path
df = pd.read_excel( Path('.') /'data'/ '2018_Sales_Total.xlsx')

# %%
var_cats = df.select_dtypes(include=np.object).columns.to_list()
df1=df[ [var_cats[0], 'quantity']].groupby(var_cats).agg(
                [('media',np.mean),
                ('suma',np.sum), 
                ('q90',lambda x : np.quantile(x,0.1))]).style.format('{:.1f}').background_gradient(subset=['quantity'], cmap='BuGn')

# %%
