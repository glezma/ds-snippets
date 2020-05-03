# %% Importing libraries
import pandas as pd
import matplotlib
import pandas_profiling
pd.options.display.float_format = '{:,.1f}'.format


# %% Loading data

df = pd.read_csv('Tesoreria_venta.csv')

# %% Basic manual descriptive info
df.info()
# %%  %% Basic manual descriptive statistics
dfNum=df[['RECENCY','FRECUENCY','MONETARY','AVGMONETARY']]

dfCat=df[['CODSUBSEGMENTO','BANCA_GYP']]

dfNum.describe()

# %% Pandas descriptive profiling can take a few hours

pandas_profiling.ProfileReport(dfNum).to_file('report.html')

