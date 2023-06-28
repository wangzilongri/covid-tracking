closeAllConnections()
list.of.packages <- c("ggplot2", "Rcpp", "grf", "caret", "mltools", "rpart", "minpack.lm", "doParallel", "rattle", "anytime","rlist")
list.of.packages <- c(list.of.packages, "zoo", "dtw", "foreach", "evaluate","rlist","data.table","plyr","tidyverse", "data.table")


new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, lib='/home/zwang937/local/R_libs', repos="http://cran.us.r-project.org", dependencies = TRUE, INSTALL_opts = '--no-lock')


lapply(list.of.packages, require, character.only = TRUE)




# Set Working Directory to File source directory
#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


#registerDoParallel(cores=min(c(detectCores(),40)))
registerDoParallel(cores=min(c(detectCores(),60)))
print(paste0("There are ",toString(detectCores())," cores"))
#break
# Obtain the latest data to see how many dates there are


destfile = paste("../data/augmented_us-counties_latest",".csv",sep="")


county_data <- as.data.frame(fread(file = destfile))


earliest_start = min(county_data$days_from_start)
latest_date = max(county_data$days_from_start)

rm(county_data)
gc()

windowsize = 2
block.folder = paste("../data/block_windowsize=",toString(windowsize),sep="")


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
num_trees=100
cutoff.list <- first.block.cutoff:(latest_date)
#cutoff.list <- 801:latest_date
cutoff.list <- 247:800
#cutoff.list <- latest_date:latest_date
# Main loop, parallelize later




mainDir = "../data/output"
subDir = paste("individual_county_backtest_state_forests_windowsize=",toString(windowsize),sep="")
outputfolder = file.path(mainDir, subDir)


dir.create(outputfolder, showWarnings = FALSE)


#cutoff.list <- 120:120


counter <- 1




#cutofflist=earliest_start:latest_date
#cutofflist = (latest_date):(latest_date)


#cutoff.list

# Get list of fips from master fips
county_fips_master <- fread("../data/county_fips_master.csv")
fips_list = unique(county_fips_master$fips)

# Read all the blocks as data
# Then we subset by cutoff and fips in each process
# Subsetting and sequencing of blocks happens in each process
all_block_fnames_full_path = sort(list.files(path=block.folder, pattern="*.csv", full.names=TRUE))
# Concatenate every 7 days until no more
# e.g. 51 is the start
# Then on 63, we have 63,56
print("lapply fread-ing all the blocks")

#df.list <- lapply(all_block_fnames_full_path,fread)
df.list <- mclapply(all_block_fnames_full_path, fread)
# DO NOT RBIND YET
# We keep this as a list dataframes. We construct the sequence and then rbind
#df <- do.call(rbind,df.list)
cutoff_to_index_dict = setNames(seq_along(cutoff.list), cutoff.list)
#print(cutoff_to_index_dict)
print("Computing GRF for each fips x cutoff pair")
foreach(cutoff = (cutoff.list)) %:%
  foreach(fips_code = fips_list) %dopar% {
#for(cutoff in cutoff.list){
  #################################
  # Skip file if it exists  
  check.file.name <- paste0("block_results_fips=",toString(fips_code),"_",toString(cutoff),".csv")
  check.file.full.name <- file.path(outputfolder, check.file.name) 
  if (file.exists(check.file.full.name)){
	print(paste0(check.file.name," exists skipping"))
	next
  }
  #################################
  # See if block is already in there
  # Block is numbered by last day in it
  
  tryCatch(expr={
    start_time <- Sys.time()
    # Given my current cutoff, which block numbers should I use?
	print(paste0("Computing GRF for cutoff=", toString(cutoff)," fips=",toString(fips_code)))
	shift <- (cutoff - first.block.cutoff)%%windowsize 
	starting_block_index = max(first.block.cutoff, cutoff-300)
	data.cutoff.list <- c(seq(starting_block_index + shift, cutoff, windowsize))

	#print(data.cutoff.list)
	indices = data.cutoff.list - starting_block_index + 1
	#print(indices)

	#cutoff_subsetted_df_list <- sapply(indices, function(x) df.list[[x]]) 
	# REPLACE WITH PARALLEL APPLY
	#cutoff_subsetted_df_list <- lapply(df.list, "[[", indices)
	cutoff_subsetted_df = tibble()
	for (index in indices){
	  cutoff_subsetted_df = rbind(cutoff_subsetted_df, df.list[[index]])
	}
	#cutoff_subsetted_df <- do.call(rbind, cutoff_subsetted_df_list)
	#print(typeof(cutoff_subsetted_df$fips))
	#print(paste0("Dimensions of cutoff_subsetted_df for fips=",toString(fips_code)," cutoff=",toString(cutoff)," is ",toString(dim(cutoff_subsetted_df))))
	fips_cutoff_subsetted_df <- as.data.frame(dplyr::filter(cutoff_subsetted_df, fips==fips_code))
	#print(str(fips_cutoff_subsetted_df))
	#print(paste0("Dimensions of fips_cutoff_subsetted_df for fips=",toString(fips_code)," cutoff=",toString(cutoff)," is ",toString(dim(fips_cutoff_subsetted_df))))
	# Parallel Partition each block by fips in parallel 
    #fips_list = unique(covariates$fips)	
	treatment <- fips_cutoff_subsetted_df$shifted_time
	outcome <- fips_cutoff_subsetted_df$shifted_log_rolled_cases

	exclusion <- c("shifted_log_rolled_cases","new_rolled_cases","datetime","State_FIPS_Code","county","state","log_rolled_cases.x","shifted_time")

	covariates <- (fips_cutoff_subsetted_df[,-which(names(fips_cutoff_subsetted_df) %in% exclusion)])
	covariates <- mutate_all(covariates, function(x) as.numeric(as.character(x)))
	covariates <- as.data.frame(covariates)	
	
	state.tau.forest <- grf::causal_forest(X=covariates, Y=outcome, W=treatment, num.trees = num_trees)
	
	exclusion.test <- c("shifted_log_rolled_cases","new_rolled_cases","datetime","State_FIPS_Code","county","state","shifted_time")
	
	#current.block <- read_csv(file.path(block.folder, paste("block_",toString(cutoff),".csv",sep="")))
	current.block <- subset(fips_cutoff_subsetted_df, shifted_time==(windowsize-1))
	covariates.test <- current.block[,-which(names(current.block) %in% exclusion.test)]
	covariates.test.unique <- unique(covariates.test)
	
	final.day.cases <- covariates.test.unique$log_rolled_cases.x
	covariates.test.unique <- covariates.test.unique[,-which(names(covariates.test.unique) %in% c("log_rolled_cases.x"))]
	# Subset by fips
	covariates.test.unique <- subset(covariates.test.unique, fips==fips_code)	
	covariates.test.unique <- mutate_all(covariates.test.unique, function(x) as.numeric(as.character(x)))
	state.tau.hat <- predict(state.tau.forest, covariates.test.unique, estimate.variance = TRUE)$predictions
	#state.tau.hat <- unlist(state.tau.hat)
	#print(state.tau.hat)
	
	identifiers <- unique(covariates.test.unique[c("fips","log_rolled_cases.y")])
	E.log_rolled_cases <- c()
	E.shifted_time <- c()
	nrows <- dim(identifiers)[1]
	for (i in 1:nrows){
	  fips_identifier <- identifiers[i,1]
	  cases <- identifiers[i,2]
	  current.county.block <- subset(current.block, fips == fips_identifier & log_rolled_cases.y == cases)
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
	
	output.fname = paste("block_results_fips=",toString(fips_code),"_",toString(cutoff),".csv",sep="")
	destfolder = file.path(outputfolder,output.fname)
	write.csv(results, destfolder, row.names=FALSE)
	
	end_time <- Sys.time()
	
	time_taken <- end_time - start_time
	
	print(paste("Time taken for fips=",toString(fips_code)," cutoff=",toString(cutoff)," is ",toString(time_taken),sep=""))
	#rm(results)
	#rm(state.tau.forest)	
	#gc()
	}, # expr close
	finally={
	rm(results)
	rm(state.tau.forest)
	rm(covariates)
	rm(treatment)
	rm(outcome)
	rm(fips_cutoff_subsetted_df)
	rm(cutoff_subsetted_df)
	gc()
	} # finally close
	) # tryCatch close
  } # foreach close
closeAllConnections()
