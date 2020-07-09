# %%
import numpy as np
import pandas as pd

np.random.seed(seed=42)
# %%
a =  np.random.randn(3,1)
a
# %%
b = np.random.randint(100, size=(4, 2))

# %%
class Leon:
    def __init__(self, input_name):
        self.name = input_name
    def __str__(self):
        return 'Mi nombre es {}'.format(self.name)
    def __repr__(self):
        return 'Mi nombre es {}'.format(self.name)

# %%


# %%
