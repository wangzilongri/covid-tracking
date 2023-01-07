list.of.packages <- c("ggplot2", "Rcpp", "grf", "caret", "mltools", "rpart", "minpack.lm", "doParallel", "rattle", "anytime")
list.of.packages <- c(list.of.packages,"plyr", "zoo","usmap","readxl","lubridate","tidyverse","data.table")


new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, lib='/home/zwang937/local/R_libs', repos="http://cran.us.r-project.org", dependencies = TRUE, INSTALL_opts = '--no-lock')
# Will need to add custom installation folder for servers without admin access
lapply(list.of.packages, require, character.only = TRUE) 


dataT = as.data.frame(fread("../data/dataT.csv"))
act_dataF = as.data.frame(fread("../data/act_dataF.csv"))

dataF<-join(x=dataT, y=act_dataF, by = c("fips","date"), type="left")

end_file = paste("../data/augmented_us-counties-states_latest",".csv",sep="")

fwrite(dataF, end_file, row.names=FALSE)

closeAllConnections()
