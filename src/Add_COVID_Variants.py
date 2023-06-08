#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import requests

from itertools import product


# ### Overall Code Logic
# 
# Data comes in as `bi_weekly_dates x hhs_region x variant`, with the corresponding `share`
# 
# End goal is to transform this into:
# `daily_dates x fips` with `variant_1, variant_2, ..., variant_n` as features with their corresponding `shares` adding up to `1`
# 
# Where the `dates` will range from the beginning of the TLGRF project till now (when data gets updated)

# ### Import CDC's CSV File of COVID 19 Proportions 

# In[2]:


url = "https://data.cdc.gov/api/views/jr58-6ysp/rows.csv?accessType=DOWNLOAD"
data_path = "../data/SARS-CoV-2_Variant_Proportions.csv"
UPDATE_DATA = True

if (UPDATE_DATA):
    response = requests.get(url)
    with open(data_path, "wb") as f:
        f.write(response.content)

df = pd.read_csv(data_path, low_memory=False)
df["week_ending"] = pd.to_datetime(df["week_ending"])
df["published_date"] = pd.to_datetime(df["published_date"])

df["variant"] = df["variant"].apply(str)
#df["share"] = df["share"].astype('int64')


df.sort_values(by=["week_ending","usa_or_hhsregion","variant"], inplace=True, ascending=[True,True,True])
df.head(30)


# In[ ]:


unique_dates = pd.unique(df["week_ending"])
unique_dates


# In[ ]:


unique_regions = pd.unique(df["usa_or_hhsregion"])
unique_regions


# In[ ]:


unique_variants = sorted(pd.unique(df["variant"]))
unique_variants


# ### Multiple Published Reports of `share` per `date`, `hhs_region`, `variant`. Keep by latest `publication_date`

# In[ ]:


df_latest = df.sort_values('published_date', ascending=False).groupby(["week_ending","usa_or_hhsregion","variant"]).first().reset_index()
df_latest


# ### Impute missing variants for each date and region, setting their share to 0

# In[ ]:


#df_imputed = pd.DataFrame(columns=["date", "hhs_region", "variant"])

df_weekly_dates = pd.DataFrame({"date":unique_dates})
df_regions = pd.DataFrame({"hhs_region": unique_regions})
df_variants = pd.DataFrame({"variant": unique_variants})

df_imputed = pd.merge(left=df_weekly_dates, right=df_regions, how="cross")
df_imputed = pd.merge(left=df_imputed, right=df_variants, how="cross")

#for date, region, variant in product(unique_dates, unique_regions, unique_variants):
#    df_imputed = df_imputed.append({"date": date, "hhs_region": region, "variant": variant}, ignore_index=True)

df_imputed = df_imputed.merge(df_latest, how="left", left_on=["date", "hhs_region", "variant"], right_on=["week_ending","usa_or_hhsregion", "variant"])
df_imputed.fillna(0, inplace=True)
df_imputed


# ### Normalize the shares

# In[ ]:


df_imputed_select = df_imputed[["date", "hhs_region", "variant", "share"]]
def normalize(x):
    return (x) / (x.sum())
df_grouped = df_imputed_select.groupby(["date","hhs_region"])
df_normalized = df_grouped["share"].transform(normalize)
df_imputed_select["normalized_share"] = df_normalized
df_imputed_select = df_imputed_select[~(df_imputed_select["hhs_region"] == "USA")]
df_imputed_select["hhs_region"] = df_imputed_select["hhs_region"].astype(int)
df_imputed_select


# ### Verify that for each date, all the present variants sum up to 100%

# In[ ]:


for date in unique_dates:
    df_segmented_by_date = df_imputed[df_imputed["date"] == date]
    for region in pd.unique(df_segmented_by_date["hhs_region"]):
        df_segmented_by_date_and_region = df_segmented_by_date[df_segmented_by_date["hhs_region"]==region]
        sum_of_variant_shares = df_segmented_by_date_and_region["share"].sum()
        print("Date: {}, Region: {}, Total Sum of Normalized Shares {}".format(date, region, sum_of_variant_shares))
    print("\n")


# ### For each date, for each HHS region, take the latest `published_date` if there are duplicates of a variant

# In[ ]:


for date in unique_dates:
    df_segmented_by_date = df_imputed_select[df_imputed_select["date"] == date]
    for region in pd.unique(df_segmented_by_date["hhs_region"]):
        df_segmented_by_date_and_region = df_segmented_by_date[df_segmented_by_date["hhs_region"]==region]
        sum_of_variant_shares = df_segmented_by_date_and_region["normalized_share"].sum()
        print("Date: {}, Region: {}, Total Sum of Normalized Shares {}".format(date, region, sum_of_variant_shares))
    print("\n")


# ### HHS to State Mapping

# In[ ]:


df_imputed_select


# In[ ]:


hhs_data_path = "../data/hhs_regions.csv"
hhs_df = pd.read_csv(hhs_data_path, low_memory=False)
hhs_region_state_df = hhs_df[["region_number", "state_or_territory"]]


# In[ ]:


df_variant_share_by_state = pd.merge(left=df_imputed_select, right=hhs_region_state_df, how='left', left_on="hhs_region", right_on="region_number")
df_variant_share_by_state = df_variant_share_by_state[["date","hhs_region", "state_or_territory", "variant", "normalized_share"]]
df_variant_share_by_state = df_variant_share_by_state.rename(columns={"state_or_territory":"state", "normalized_share":"normalized_variant_share"})
df_variant_share_by_state["variant"] = "Variant % " + df_variant_share_by_state["variant"]
df_variant_share_by_state = df_variant_share_by_state.pivot_table(index=["date","hhs_region","state"], columns="variant", values="normalized_variant_share").reset_index()
df_variant_share_by_state.to_csv("../data/normalized_variant_share_by_date_and_state.csv", index=False)


# In[ ]:


df_variant_share_by_state


# ### State to County Mapping

# In[ ]:


fips_list_path = "../data/fips-list.csv"
fips_list = pd.read_csv(fips_list_path, low_memory=False)
fips_list


# In[ ]:


df_variant_share_by_fips = pd.merge(left=fips_list, right=df_variant_share_by_state, how='left', on="state")
df_variant_share_by_fips.to_csv("../data/df_variant_share_by_fips.csv", index=False)
df_variant_share_by_fips


# ### Fill in the missing days in between every 2 weeks with the previous week's values

# In[ ]:


date_range = pd.date_range(unique_dates[0],unique_dates[-1])
unique_fips = pd.unique(df_variant_share_by_fips["fips"])


# In[ ]:


df_dates = pd.DataFrame({"date":date_range})
df_fips = pd.DataFrame({"fips":unique_fips})
df_all_dates_fips = pd.merge(left=df_dates, right=df_fips, how="cross")
df_all_dates_fips_imputted = pd.merge(left=df_all_dates_fips, right=df_variant_share_by_fips, how="left", on=["date","fips"])
df_all_dates_fips_imputted = df_all_dates_fips_imputted.sort_values(by=["date", "fips"], ascending=[True,True])

# Group by FIPS, while being sorted by dates, and then forward fill
COVID_Variants_Normalized_Share_All_Dates_FIPS = df_all_dates_fips_imputted.fillna(df_all_dates_fips_imputted.groupby(["fips"]).ffill())

COVID_Variants_Normalized_Share_All_Dates_FIPS.to_csv("../data/COVID_Variants_Normalized_Share_All_Dates_FIPS.csv", index=False)
COVID_Variants_Normalized_Share_All_Dates_FIPS


# ### Obtain Dates of Project

# In[ ]:


dataF_path = "../data/augmented_us-counties-states_latest.csv"
dataF = pd.read_csv(dataF_path, low_memory=False)
dataF["date"] = pd.to_datetime(dataF["date"])

newer_augmented_path = "../data/augmented_us-counties-states_latest_variants.csv"


# ### Back Fill from Beginning of Variants Dataset to Start of Project Date
# 

# In[ ]:


project_start_date = dataF["date"].min()
COVID_variants_end_date = date_range[-1]
print("project_start_date: {}, COVID_variants_end_date: {}".format(project_start_date, COVID_variants_end_date))

backfill_date_range = pd.date_range(project_start_date, COVID_variants_end_date)
backfill_date_range


# In[ ]:


df_backfill_dates = pd.DataFrame({"date":backfill_date_range})
df_fips = pd.DataFrame({"fips":unique_fips})
df_backfill_dates_fips = pd.merge(left=df_backfill_dates, right=df_fips, how="cross")
df_backfilled_COVID_Variants_Normalized_Share_All_Dates_FIPS = pd.merge(left=df_backfill_dates_fips, right=COVID_Variants_Normalized_Share_All_Dates_FIPS, how="left", on=["date","fips"])
df_backfilled_COVID_Variants_Normalized_Share_All_Dates_FIPS = df_backfilled_COVID_Variants_Normalized_Share_All_Dates_FIPS.sort_values(by=["date", "fips"], ascending=[True,True])

df_backfilled_COVID_Variants_Normalized_Share_All_Dates_FIPS = df_backfilled_COVID_Variants_Normalized_Share_All_Dates_FIPS.fillna(df_backfilled_COVID_Variants_Normalized_Share_All_Dates_FIPS.groupby(["fips"]).bfill())
df_backfilled_COVID_Variants_Normalized_Share_All_Dates_FIPS


# ### Forward Fill from End of Variants Dataset to Current Date

# In[ ]:


current_date = dataF["date"].max()
print("Project Start Date: {}, Current Date for Update: {}".format(project_start_date, current_date))

forwardfill_date_range = pd.date_range(project_start_date, current_date)
forwardfill_date_range


# In[ ]:


df_forwardfill_dates = pd.DataFrame({"date":forwardfill_date_range})
df_fips = pd.DataFrame({"fips":unique_fips})
df_forwardfill_dates_fips = pd.merge(left=df_forwardfill_dates, right=df_fips, how="cross")
df_ffill_bfill_COVID_Variants_Normalized_Share_All_Dates_FIPS = pd.merge(left=df_forwardfill_dates_fips, right=df_backfilled_COVID_Variants_Normalized_Share_All_Dates_FIPS, how="left", on=["date","fips"])
df_ffill_bfill_COVID_Variants_Normalized_Share_All_Dates_FIPS = df_ffill_bfill_COVID_Variants_Normalized_Share_All_Dates_FIPS.sort_values(by=["date", "fips"], ascending=[True,True])

df_ffill_bfill_COVID_Variants_Normalized_Share_All_Dates_FIPS = df_ffill_bfill_COVID_Variants_Normalized_Share_All_Dates_FIPS.fillna(df_ffill_bfill_COVID_Variants_Normalized_Share_All_Dates_FIPS.groupby(["fips"]).ffill())
df_ffill_bfill_COVID_Variants_Normalized_Share_All_Dates_FIPS.to_csv("../data/df_ffill_bfill_COVID_Variants_Normalized_Share_All_Dates_FIPS.csv", index=False)
df_ffill_bfill_COVID_Variants_Normalized_Share_All_Dates_FIPS


# ### Merge with `augmented_us-counties-states_latest.csv`

# In[ ]:


augmented_us = pd.merge(left=dataF, right=df_ffill_bfill_COVID_Variants_Normalized_Share_All_Dates_FIPS, on=["date","fips","county","state"], how="left")
augmented_us.to_csv(newer_augmented_path, index=False)
augmented_us


# In[ ]:




