# grab old packages names
old_packages <- installed.packages(lib.loc = "/usr/share/R/library")
old_packages <- as.data.frame(old_packages)
list.of.packages <- unlist(old_packages$Package)

# remove old packages 
remove.packages( installed.packages( priority = "NA" )[,1] )

# reinstall all packages 
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages,function(x){library(x,character.only=TRUE)})
