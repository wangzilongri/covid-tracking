
closeAllConnections()
list.of.packages <- c("ggplot2", "Rcpp", "grf", "caret", "mltools", "rpart", "minpack.lm", "doParallel", "rattle", "anytime","rlist")
list.of.packages <- c(list.of.packages, "zoo", "dtw", "foreach", "evaluate","rlist","data.table","plyr","tidyverse", "data.table")
list.of.packages<-c(list.of.packages,"rlist","varhandle","stargazer","dotwhisker","ggstatsplot","Rcpp")
list.of.packages<-c(list.of.packages,"breakDown","here","DiagrammeR","reshape","data.table","xtable","dplyr","lme4","arm","readr")

new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
#if(length(new.packages)) install.packages(new.packages, lib='/home/zwang937/local/R_libs', repos="http://cran.us.r-project.org", dependencies = TRUE, INSTALL_opts = '--no-lock')


lapply(list.of.packages, require, character.only = TRUE)




# Set Working Directory to File source directory
#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


registerDoParallel(cores=min(c(detectCores(),40)))
print(paste0("There are ",toString(detectCores())," cores"))
#library("pacman")

#install.packages("ggplot2",repos = c("http://rstudio.org/_packages","http://cran.rstudio.com"))
#> You may also find it useful to restart R,
#> In RStudio, that's the menu Session >> Restart R

source("var_imp_plot.R")
pngfolder = "./analysis-plots"
dir.create(pngfolder,showWarnings=FALSE)

VTOP40_folder = "./VTOP40"
dir.create(VTOP40_folder, showWarnings=FALSE)

windowsize=2

cutoff.to.use <- 53
cutoff <- cutoff.to.use
# Cut out block
block.files.path = paste("block_windowsize=",toString(windowsize),sep="")
block.folder = file.path("../data/",block.files.path)


cutoff.list <- 1:cutoff.to.use
################################################
# Check for the first block
###############################################

first.block.cutoff <- Inf


# Check for the first block file
for (cutoff in cutoff.list){
  # See if block is already in there
  # Block is numbered by last day in it
  fname <- paste("block_",toString(cutoff),".csv",sep="")
  full.path <- file.path(block.folder,fname)
  if (file.exists(full.path)){
    print(paste(fname," exists",sep=""))
    if (first.block.cutoff > cutoff){
      first.block.cutoff <- cutoff
    }
    break
  }
}

cutoff.list <- first.block.cutoff:cutoff.to.use
###############################################################
# Load the data
# Given my current cutoff, which block numbers should I use?
###############################################################
fname <- paste("block_",toString(cutoff),".csv",sep="")
full.path <- file.path(block.folder,fname)

# Concatenate every 7 days until no more
# e.g. 51 is the start
# Then on 63, we have 63,56
shift <- (cutoff.to.use - first.block.cutoff)%%windowsize 
data.cutoff.list <- c(seq(first.block.cutoff + shift, cutoff.to.use, windowsize))
#data.cutoff.list <- c(seq(first.block.cutoff, cutoff, 1))
print(data.cutoff.list)

block.fullpath.list <- c()
for (block.number in data.cutoff.list){
  block.fullpath.list <- c(block.fullpath.list, file.path(block.folder, paste("block_",toString(block.number),".csv",sep="")))
}

df.list <- lapply(block.fullpath.list,read.csv)
df <- do.call(rbind,df.list)


##############################################
# GENERATE THE TREATMENT AND OUTCOME SLICES
##############################################
treatment <- df$shifted_time
outcome <- df$shifted_log_rolled_cases

#exclusion <- c("shifted_log_rolled_cases","fips","State_FIPS_Code","county","state","datetime","log_rolled_cases.x","shifted_time")
exclusion <- c("shifted_log_rolled_cases","new_rolled_cases","datetime","State_FIPS_Code","county","state","log_rolled_cases.x","shifted_time")

covariates <- (df[,-which(names(df) %in% exclusion)])

exclusion.test <- c("shifted_log_rolled_cases","new_rolled_cases","datetime","State_FIPS_Code","county","state","shifted_time")

current.block <- read.csv(file.path(block.folder, paste("block_",toString(cutoff),".csv",sep="")))
current.block <- subset(current.block, shifted_time==(windowsize-1))
covariates.test <- current.block[,-which(names(current.block) %in% exclusion.test)]
covariates.test.unique <- unique(covariates.test)

final.day.cases <- covariates.test.unique$log_rolled_cases.x
covariates.test.unique <- covariates.test.unique[,-which(names(covariates.test.unique) %in% c("log_rolled_cases.x"))]

################################################
# LOAD THE GRF OBJECT
################################################

#cutoff.to.use.list <- c(346,578)
#cutoff.to.use.list <- seq(68,593,15)
cutoff.to.use.list <- 53:1075
#cutoff.to.use.list <- 1074:1075
foreach(cutoff.to.use = cutoff.to.use.list) %dopar% {
#for (cutoff.to.use in cutoff.to.use.list){
  #grf.string <- paste("./grf_2000/grf_stateforest_cutoff=",toString(cutoff.to.use),".rds",sep="")
  grf.string <- sprintf("../data/output/grf_windowsize=2_numtrees=200/grf_stateforest_cutoff=%d.rds",cutoff.to.use)
  print(paste0("Reading ",grf.string," object"))
  grf.object <- readRDS(grf.string)
  
  #tree <- get_tree(grf.object,1)
  # Obtain names of original training data
  Vnames = as.vector(names(grf.object$X.orig))
  names(Vnames) <- "Vnames"
  
  VI = grf::variable_importance(grf.object)
  names(VI) <- "Importance"
  #print(VI)
  tVI = as.vector(t(VI))
  #print(tVI)
  # Generate Vresults
  print(paste0("Generating Vresults for cutoff.to.use=",toString(cutoff.to.use))) 
  Vresults <- melt(data.frame(tVI,Vnames))
  #print(Vresults)
  Vresults$variable <- NULL
  names(Vresults)[names(Vresults)=="value"] <- "Importance"
  
  Vresults<-Vresults[order(-Vresults$Importance),]
  Vresults$Vnames <- factor(Vresults$Vnames, levels = Vresults$Vnames[order(-Vresults$Importance)])
  #print(Vresults)
  rownames(Vresults) <-NULL
  
  #Vresults = data.frame(t(VI))
  print(Vresults)
  #names(Vresults) <-Vnames
  ################
  # Save the VTOP
  ################

  #################################################################################################
  # PLOT ALL VARIABLES
  #################################################################################################
  #GRFVALL.path <- file.path(pngfolder,"GRFVALL.png")
  #png(filename=GRFVALL.path,width=1080,height=720,type="cairo")
  #plot(Vresults, main=paste("GRF Variable Importance cutoff = ",toString(cutoff.to.use),sep=""),xlab=paste("Vnames: Num Var=",toString(length(Vnames),sep="")) )
  #print(p)
  #dev.off()
  
  #################################################################################################
  # PLOT ALL Non-zero
  #################################################################################################
  Vnonzero <- Vresults[which(Vresults$Importance > 0),]
  num.nonzero <- length(Vnonzero$Vnames)
  Vnonzero$Vnames <- factor(Vnonzero$Vnames)
  
  GRFNONZERO.path <- file.path(pngfolder,"GRFNONZERO.png")
  #png(filename=GRFNONZERO.path,width=1080,height=720,type="cairo")
  #plot(Vnonzero, main=paste("GRF Nonzero Variable Importance cutoff = ",toString(cutoff.to.use),sep=""),xlab=paste("Vnames: Num Var=",toString(num.nonzero),sep="") )
  #print(p)
  #dev.off()
  
  #################################################################################################
  # PLOT TOP N
  #################################################################################################
  N=40
  print(paste0("Generating VTOPN_", toString(cutoff.to.use)))
  
  VTOPN <- Vnonzero[(1):(N),]
  VTOPN$Vnames <- factor(VTOPN$Vnames)
  
  
  
  ###################
  # LATEX OUTPUT
  ###################
  #LATEX.VTOPN <- VTOPN
  #LATEX.VTOPN$"Importance (%)" <- VTOPN$Importance * 100
  #LATEX.VTOPN$Importance <- NULL
  #print(xtable(LATEX.VTOPN,digits=c(0,0,4)),include.rownames=FALSE)
  
  
  ######################
  # VAR IMPORTANCE PLOT
  ######################
  # Calculate variable importance of all features
  # (from print.R)
  print(paste0("Writing VTOP40_",toString(cutoff.to.use),".csv")) 
  VTOP40 <- Vnonzero#[(1):(40),]
  VTOP40$Vnames <- factor(VTOP40$Vnames)
  VTOP40$"Importance (%)" <- VTOP40$Importance * 100
  
  #Write as CSV
  VTOP40_path = file.path(VTOP40_folder, paste0("VTOP40_",toString(cutoff.to.use),".csv"))
  write_csv(VTOP40, VTOP40_path)
  
  #GRFTOP40.path <- file.path(pngfolder,sprintf("GRFTOP_%d_%d.png",cutoff.to.use,40))
  #png(GRFTOP40.path,width=1080,height=720,type="cairo")
  #p<-ggplot(VTOP40, aes(x=Vnames, y=`Importance (%)`)) +geom_bar(stat='identity') +coord_flip() 
  #print(p)
  #dev.off()
}
#CODE IS OK UP TO HERE

break
# Look at latest data
#all_data <- read_csv("./augmented_us-counties-states_latest_utf-8.txt")
#all_data <-  all_data[,colSums(is.na(all_data))<nrow(all_data)]
# Get last 14 days worth
#last.cutoff <- max(all_data$days_from_start)
#all_data_last_14 <- all_data%>% filter( between(days_from_start, last.cutoff-13, last.cutoff) )
# Omit NAs from state FIPS
#all.data.last14.omit <- all_data_last_14[!is.na(all_data_last_14$State_FIPS_Code),]
#fits <- lm(rolled_cases ~ . -county -state -datetime -State_FIPS_Code  , na.action=na.exclude, data=all.data.last14.omit)
#fitted_models <- all.data.last14.omit %>% group_by(state) %>% do(model = lm(rolled_cases ~  . , data = .))

############################
# Compare with linear model
############################
# Only keep columns with 40 most important variables
top10names <- unfactor(VTOP40$Vnames)[1:10]
auxilliary.var<- c("fips","State_FIPS_Code","county","state","datetime","log_rolled_cases.y")
df.important <- df[list.append(unfactor(VTOP40$Vnames),auxilliary.var)]
df.important[VTOP40$Vnames]%>%mutate_if(is.numeric,scale)

df.important.top10 <- df[list.append(top10names,auxilliary.var)]
df.important.top10[top10names]%>%mutate_if(is.numeric,scale)

#df.important[VTOP40$Vnames] <- lapply(df.important[VTOP40$Vnames],scale)
#write_csv(x=df.important,"./all_blocks_top40.csv")

fits.all <- lm(log_rolled_cases.y ~ . -State_FIPS_Code -county -state -datetime -fips, na.action=na.exclude , data=df.important)
fits.time.state <- lmList(log_rolled_cases.y ~ cutoff | state, na.action=na.exclude , data=df.important)
fits.all.state <- lmList(log_rolled_cases.y ~ . -State_FIPS_Code -county -datetime -fips | state, na.action=na.exclude , data=df.important)

fits.all.top10 <- lm(log_rolled_cases.y ~ . -State_FIPS_Code -county -state -datetime -fips, na.action=na.exclude , data=df.important.top10)
#export.table<-stargazer(fits.all, fits.time, title="Results Linear Model", align=TRUE)
#fits.time <- lmList(log_rolled_cases.y ~ cutoff | state, na.action=na.exclude , data=df.important)
#print(export.table,file=file.path("RegressionTables.txt"))
#writeLines(export.table,file.path("RegressionTables.txt"))

#dwplot(fits.all)
#https://indrajeetpatil.github.io/ggstatsplot/reference/ggcoefstats.html
ggstatsplot::ggcoefstats(x = fits.all, output = "plot")

ggstatsplot::ggcoefstats(x = fits.all.top10, statistic = "t",
                         exclude.intercept = TRUE,
                         meta.analytic.effect = TRUE,
                         k = 3)
