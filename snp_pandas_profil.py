# %% Importing libraries
import pandas as pd
import matplotlib
from pathlib import Path

import pandas_profiling
pd.options.display.float_format = '{:,.1f}'.format


# %% Loading data
filename = 'Tesoreria_venta.csv'
filename_output = 'report.csv'
filepath = Path('.' ) / 'data' / filename
filepath_output = Path('.' ) / 'data' / filename_output
df = pd.read_csv(filepath)

# %% Basic manual descriptive info
df.info()
# %%  %% Basic manual descriptive statistics
dfNum=df[['RECENCY','FRECUENCY','MONETARY','AVGMONETARY']]

dfCat=df[['CODSUBSEGMENTO','BANCA_GYP']]

dfNum.describe()

# %% Pandas descriptive profiling can take a few hours

pandas_profiling.ProfileReport(dfNum).to_file(filepath_output)

