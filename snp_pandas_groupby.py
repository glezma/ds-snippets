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
# %% Loading Data from cognos & GS

from cmath import nan
from typing import Iterator
import pandas as pd
import numpy as np
from pathlib import Path
import itertools as it
import utils.planpy as pp

input_path = Path('.') / 'input_data' 


file_to_load_tl = 'Topline_Q4G.xlsx'

df_tot_mx = pd.read_excel(input_path / file_to_load_tl ,sheet_name ="cognos_gms_2py")
df_gs = pd.read_excel(input_path /  file_to_load_tl,sheet_name="gs_gms_2py")

df_exch_rate = pd.read_excel(input_path /  file_to_load_tl,sheet_name="exch_rate")
exch_rate = df_exch_rate['Value'][0]


# % Unpivoting dfs

id_vars= df_gs.columns[0:6].to_list()
value_vars = df_gs.columns[6:].to_list()
df_gs = pd.melt(df_gs,id_vars=id_vars, value_vars= value_vars,
         var_name='month', value_name='value' )
df_tot_mx = pd.melt(df_tot_mx,id_vars=id_vars, value_vars= value_vars,
         var_name='month', value_name='value' )

# % converting GS value to MXN

df_gs['value']= df_gs['value']
# Concatenating dfs
df = pd.concat([ df_tot_mx, df_gs])

# pivot
index = df.columns[1:-1].to_list()
columns = 'account'
values = 'value'
df['value']=np.abs(df['value'])
df_upvt= df.pivot(index=index, columns=columns, values=values).reset_index()

df_upvt.UNITS=df_upvt.UNITS.fillna(df_upvt.GMS*0)

df_upvt['ASP'] = df_upvt.GMS.divide(df_upvt.UNITS,fill_value=np.nan)


# % converting text to datetime format '%Y.%m'

df['time']= pd.to_datetime(df['time'].astype(str)).dt.strftime('%Y.%m')

