# %% Importing libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import utils.fchart as ut
pd.options.display.float_format = '{:,.1f}'.format

# %% Generating data
# Data to plot as line
mu = 0.0
sigma = 0.09
n = 100
epsilon = np.random.normal(mu, sigma, n)
y = np.zeros(shape=(n,1))
y[0]=0
for ii in range(1,n):
    y[ii] = 1+0.9*y[ii-1]+epsilon[ii]

# Data to plot as dots
# y1 = np.array([ 1+0.9*y[ii-1]+epsilon[ii] for  ii in range(1,n)])
y1 = y + np.random.normal(0,sigma*6,len(y)).reshape(-1,1)


# Preparing Dataframe to plot as Fanchart
df = pd.DataFrame()
x = list(range(0,n))
df['x'] = x
df['y_real'] = y1.reshape(-1)
df['y_pred'] = y.reshape(-1)
df['CI:2.5-97'] = (y-3)
df['CI:2.5-97_2'] = (y+3).reshape(-1)
df['CI:5-95'] = (y-2)
df['CI:5-95_2'] = (y+2).reshape(-1)
df['CI:25-75'] = (y-1).reshape(-1)
df['CI:25-75_2'] = (y+1).reshape(-1)

# Matrix of colors (RGB format) to color the fan chart
colorListRGB= [ [200,222,255],
                    [153,212,255],
                    [0,99,174],
                    ]

# Using the fan chart command
fig = ut.plot_fanchart(df, title= 'Titulo de grafico',dot_name='Punto',line_name='Linea', 
                    xLabel='Eje x', yLabel='y',
                    colorListRGB=colorListRGB)
fig.show()


# %%

