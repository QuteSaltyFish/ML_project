#%%
import numpy as np
import torch as t 
import pandas as pd 


#%%
dataframe = pd.read_csv('data/train_val.csv')


# %%
data = dataframe.values

# %%
data[:,1]
# %%
