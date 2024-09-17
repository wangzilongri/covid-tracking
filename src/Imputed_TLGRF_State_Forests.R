closeAllConnections()
list.of.packages <- c("ggplot2", "Rcpp", "grf", "caret", "mltools", "rpart", "minpack.lm", "doParallel", "rattle", "anytime","rlist")
list.of.packages <- c(list.of.packages, "zoo", "dtw", "foreach", "evaluate","rlist","data.table","plyr","dplyr")


new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, lib='/home/zwang937/local/R_libs', repos="http://cran.us.r-project.org", dependencies = TRUE, INSTALL_opts = '--no-lock')


lapply(list.of.packages, require, character.only = TRUE)




# Set Working Directory to File source directory
#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


registerDoParallel(cores=min(detectCores(), 20))


# Obtain the latest data to see how many dates there are

print("Beginning LLF_TLGRF_State_Forests.R")

destfile = paste("../analysis/benchmark_LLF/imputed_augmented_us-counties-states_latest_variants",".csv",sep="")


county_data <- as.data.frame(fread(file = destfile))


earliest_start = min(county_data$days_from_start)
latest_date = max(county_data$days_from_start)




windowsize = 2

print(paste0("windowsize = ",toString(windowsize)))



block.folder = paste("../data/imputed_block_windowsize=",toString(windowsize),sep="")


cutoff.list <- earliest_start:latest_date


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
num_trees=200
cutoff.list <- first.block.cutoff:latest_date
#cutoff.list <- latest_date:latest_date
# Main loop, parallelize later

print(paste0("Imputed_TLGRF_State_Forests.R latest_date is ",toString(latest_date)))


mainDir = "../data/output"
subDir = paste("backtest_Imputed_TLGRF_state_forests_windowsize=",toString(windowsize),sep="")
outputfolder = file.path(mainDir, subDir)


dir.create(outputfolder, showWarnings = FALSE)

grf.subfolder = paste("LLF_TLGRF_windowsize=",toString(windowsize),"_numtrees=",toString(num_trees),sep="")
grf.outputfolder = file.path(mainDir,grf.subfolder)
dir.create(grf.outputfolder, showWarnings = FALSE)

#cutoff.list <- 120:120


counter <- 1

# Step 1: Determine the range in the logarithmic scale
log_min <- -2
log_max <- 1

# Step 2: Create a sequence in the logarithmic scale
lambda_exponent_list <- seq(log_min, log_max)  # 100 values for example


# Step 1: Read the CSV file and extract the top 10 feature names
sorted_feature_importance <- read.csv("../analysis/TLGRF_Feature_Importance/Sorted_Feature_Importance.csv", header = TRUE)

# Assuming the unnamed column with feature names is the first column:
sorted_features <- sorted_feature_importance[, 1]

# SET START AND END
args <- commandArgs(trailingOnly = TRUE)

start_day = min(county_data$days_from_start)
end_day = max(county_data$days_from_start)

# Set default values for start and end if less than 2 arguments are provided
if (length(args) >= 2) {
  start_day <- as.integer(args[1])
  end_day <- as.integer(args[2])
  print(c(start_day,end_day))
  cutoff.list <- start_day:end_day
}



foreach(cutoff = (cutoff.list)) %dopar%{
#for(cutoff in cutoff.list){
  #################################
  # Skip file if it exists  
  #check.file.name <- paste0("block_results_",toString(cutoff),".csv")
  #check.file.full.name <- file.path(outputfolder, check.file.name) 
  check.file.name <- paste0("LLF_TLGRF_stateforest_cutoff=", toString(cutoff), ".rds")
  check.file.full.name <- file.path(grf.outputfolder, check.file.name)
    
  output.fname = paste("Imputed_TLGRF_block_results_",toString(cutoff),".csv",sep="")
  destfolder = file.path(outputfolder,output.fname)

  if (file.exists(check.file.full.name) & file.exists(destfolder)){
	print(paste0(destfolder, " exists, skipping"))
    next
  }
  #################################
  # See if block is already in there
  # Block is numbered by last day in it
  print(paste0("Computing Imputed TLGRF for ", toString(cutoff)))
  try({
    start_time <- Sys.time()
    # Given my current cutoff, which block numbers should I use?
    fname <- paste("block_",toString(cutoff),".csv",sep="")
    full.path <- file.path(block.folder,fname)
    
    
	tau.forest.fname <- paste("LLF_TLGRF_stateforest_cutoff=",toString(cutoff),".rds",sep="")
    tau.forest.path <- file.path(grf.outputfolder,tau.forest.fname)
	state.tau.forest <- readRDS(tau.forest.path)   	
 
    exclusion.test <- c("shifted_log_rolled_cases","new_rolled_cases","datetime","State_FIPS_Code","county","state","shifted_time")
    
    current.block <- read.csv(file.path(block.folder, paste("block_",toString(cutoff),".csv",sep="")))
    current.block <- subset(current.block, shifted_time==(windowsize-1))
    covariates.test <- current.block[,-which(names(current.block) %in% exclusion.test)]
    covariates.test.unique <- unique(covariates.test)
    
    final.day.cases <- covariates.test.unique$log_rolled_cases.x
    covariates.test.unique <- covariates.test.unique[,-which(names(covariates.test.unique) %in% c("log_rolled_cases.x"))]
	covariates.test.unique <- mutate_all(covariates.test.unique, function(x) as.numeric(as.character(x)))                                
    
    state.tau.hat <- predict(state.tau.forest, covariates.test.unique, estimate.variance = TRUE)$predictions
    #state.tau.hat <- unlist(state.tau.hat)
    #print(state.tau.hat)
    
    identifiers <- unique(covariates.test.unique[c("fips","log_rolled_cases.y")])
    E.log_rolled_cases <- c()
    E.shifted_time <- c()
    nrows <- dim(identifiers)[1]
    for (i in 1:nrows){
      fips <- identifiers[i,1]
      cases <- identifiers[i,2]
      current.county.block <- subset(current.block, fips == fips & log_rolled_cases.y == cases)
      E.log_rolled_cases <- c(E.log_rolled_cases,mean(current.county.block$log_rolled_cases.x))
      # Doesn't matter that order for time is reversed
      E.shifted_time <- c(E.shifted_time,mean(cutoff-6+current.county.block$shifted_time))
    }
    state.t0.hat <- (E.log_rolled_cases - state.tau.hat*E.shifted_time)/(-state.tau.hat)
    
    #print(state.t0.hat)
   
    # Write down results
    results <- data.frame("fips"=identifiers[1],"log_rolled_cases.y"=identifiers[2],"days_from_start"=cutoff)
    results <- merge(x=results,y=current.block[which(current.block$shifted_time==windowsize-1),],by="fips")
    results <- results[, c("fips","county","state","days_from_start","datetime","log_rolled_cases.x")]
    results <- unique(results)
    results$t0.hat <- state.t0.hat
    results$tau.hat <- state.tau.hat
    results$predicted.grf.future <- (state.tau.hat*((cutoff + 7) - state.t0.hat ))
    results$predicted.grf.future.0 <- state.tau.hat*(windowsize-1 + 7)+ covariates.test.unique$log_rolled_cases.y
    results$predicted.grf.future.last <- state.tau.hat*(7)+ final.day.cases
    results$Predicted_Double_Days <- log(2,exp(1))/state.tau.hat
    #results$date <- unique(current.block[which(current.block$shifted_time==6),"datetime"])
    #results$log_rolled_cases.x <- (current.block[which(current.block$shifted_time==6),"log_rolled_cases.x"])
    
    output.fname = paste("Imputed_TLGRF_block_results_",toString(cutoff),".csv",sep="")
    destfolder = file.path(outputfolder,output.fname)
    write.csv(results, destfolder, row.names=FALSE)
    
    end_time <- Sys.time()
    
    time_taken <- end_time - start_time
    
    print(paste("Time taken for cutoff=",toString(cutoff)," is ",toString(time_taken),sep=""))
    rm(results)
	rm(outcome)
	rm(treatment)
	rm(state.tau.forest)
	rm(covariates)
	gc() 
  })
  #break
}
closeAllConnections()
