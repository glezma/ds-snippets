# %%
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
# from statsmodels.tsa.arima_process import ar_generate_sample

def create_data(n):
    x1 =  np.random.standard_normal(size=(n,1)) * 3 +10
    x2 = np.zeros((n,1))
    x3 = np.random.random_integers(1,3,n).reshape(-1,1)
    for i in range(1,n):
        x2[i] = 0.95 * x2[i-1] + np.random.normal() +3
    y = 2*x1 + 4* x2 + 4*x3 +np.random.standard_normal(size=(n,1))*0.1
    return np.c_[y,x1,x2,x3]
data = create_data(30)
features = ['LifeSat','HealthI','GDP','Contin']
df = pd.DataFrame(data=data, columns= features)
full_file = Path('.') / 'data' / 'more_data.csv'
df.to_csv(full_file, index=False)

# %%
