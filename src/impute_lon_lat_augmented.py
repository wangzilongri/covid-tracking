#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from multiprocessing import Pool, cpu_count
import dask.dataframe as dd


# ## Obtain County Lon Lat Data

# In[2]:


block_path = "../data/block_windowsize=2/block_500.csv"
block_df = (dd.read_csv(block_path, assume_missing=True)).compute()
augmented_data_path = "../data/augmented_us-counties-states_latest.csv"
augmented_df = (dd.read_csv(augmented_data_path, assume_missing=True)).compute()


# ### Impute Virgin Islands, St Thomas `LON, LAT` as `-64.916667, 18.333333`

# In[3]:


geo_data = augmented_df[["state","county","fips","LON","LAT"]]
#geo_data = geo_data.drop_duplicates()
geo_data_NAN_mask = geo_data.isna()
#print(geo_data[geo_data_NAN_mask.any(axis=1)])
# St Thomas FIPS = 78030
geo_data.loc[geo_data["fips"]==78030, "LON"] = -64.916667
geo_data.loc[geo_data["fips"]==78030, "LAT"] = 18.333333
# St John FIPS = 78020
geo_data.loc[geo_data["fips"]==78020, "LON"] = -64.7930
geo_data.loc[geo_data["fips"]==78020, "LAT"] = 18.3315
# St Croix FIPS = 78010
geo_data.loc[geo_data["fips"]==78010, "LON"] = -64.7033
geo_data.loc[geo_data["fips"]==78010, "LAT"] = 17.7460
# Tinian FIPS = 69120
geo_data.loc[geo_data["fips"]==69120, "LON"] = 145.6197
geo_data.loc[geo_data["fips"]==69120, "LAT"] = 14.9997
# Saipan FIPS = 69120
geo_data.loc[geo_data["fips"]==69110, "LON"] = 145.7350
geo_data.loc[geo_data["fips"]==69110, "LAT"] = 15.1750
# Rota FIPS = 69120
geo_data.loc[geo_data["fips"]==69100, "LON"] = 145.1492
geo_data.loc[geo_data["fips"]==69100, "LAT"] = 14.1533
# Yakutat plus Hoonah-Angoon FIPS=2998, using Yakutat's
geo_data.loc[geo_data["fips"]==2998, "LON"] = -139.7272
geo_data.loc[geo_data["fips"]==2998, "LAT"] = 59.5467
# Bristol Bay plus Lake and Peninsula FIPS = 2997, using Bristol Bay's
geo_data.loc[geo_data["fips"]==2997, "LON"] = -158.5050
geo_data.loc[geo_data["fips"]==2997, "LAT"] = 59.0433
print(geo_data)
augmented_df.update(geo_data)


# In[4]:


augmented_df.to_csv(augmented_data_path, index=False)


# In[ ]:




