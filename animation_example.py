# %% Importing libraries
import pandas as pd
import numpy as np
import numpy.random as npr
import matplotlib
import pandas_profiling
import plotly.graph_objects as go
import utils as ut
pd.options.display.float_format = '{:,.1f}'.format

# %% Generating data
# Data to plot as line
import plotly.express as px
df = px.data.gapminder()
px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
           size="pop", color="continent", hover_name="country",
           log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])


# %%
df

# %%
