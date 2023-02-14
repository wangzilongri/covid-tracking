#!/bin/bash

Rscript "Preprocess_us-counties-states.R"
Rscript "Add_CUSP.R"
Rscript "join_dataT_act_dataF.R"
jupyter nbconvert --to script ../scratch/Add_COVID_Variants.ipynb

Rscript "Specific_Cases.R"
Rscript "Block_Prepare.R"

#5
Rscript "State_Forests.R"

#6
Rscript "Preprocess_us-counties-states_minus7.R"

#7
Rscript "14_day_trend.R"

#8
Rscript "30_day_check.R"

#9
Rscript "Post-Process-Block-Forests.R"

#10
Rscript "RMSE_Plot.R"

#11
Rscript "MAPE_Plot.R"

#12
Rscript "county_plot.R"

git add ../data
git commit -m "Update $(date +'%Y-%m-%d')"
git push

#killall R
