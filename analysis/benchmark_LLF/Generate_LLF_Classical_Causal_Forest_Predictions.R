list.of.packages <- c("ggplot2", "Rcpp", "grf", "caret", "mltools", "rpart", "minpack.lm", "doParallel", "rattle", "anytime","rlist")
list.of.packages <- c(list.of.packages, "zoo", "dtw", "foreach", "evaluate","rlist","data.table","plyr","dplyr")


new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, lib='/home/zwang937/local/R_libs', repos="http://cran.us.r-project.org", dependencies = TRUE, INSTALL_opts = '--no-lock')


lapply(list.of.packages, require, character.only = TRUE)

# Register Parallel Backend for dopar
registerDoParallel(cores=min(detectCores(), 4))


# SET PARAMS
windowsize=2
num_trees = 200



# CREATE OUTPUT FOLDERS
mainDir = "./classical_causal_forest_results"
dir.create(mainDir, showWarnings = FALSE)

subDir = paste("llf_backtest_state_forests_windowsize=",toString(windowsize),"_numtrees=",toString(num_trees),sep="")
outputfolder = file.path(mainDir, subDir)


dir.create(outputfolder, showWarnings = FALSE)

grf.subfolder = paste("llf_windowsize=",toString(windowsize),"_numtrees=",toString(num_trees),sep="")
grf.outputfolder = file.path(mainDir,grf.subfolder)
dir.create(grf.outputfolder, showWarnings = FALSE)

# READ DATA
print("Reading in data")
destfile <- paste("imputed_augmented_us-counties-states_latest_variants.csv")
augmented_panel_data <- as.data.frame(fread(file = destfile))
# Time variant
#hhs_X_w_clusters_fpath = "../benchmark_tcv_kmeans_code/hhs_X_w_clusters.csv"
#hhs_X_w_clusters = as.data.frame(fread(file = hhs_X_w_clusters_fpath))

print("Transforming augmented_panel_data")
augmented_panel_data <- augmented_panel_data[order(augmented_panel_data$fips, augmented_panel_data$days_from_start), ]
augmented_panel_data <- augmented_panel_data[augmented_panel_data$rolled_cases >= 20,]
augmented_panel_data$log_rolled_cases <- log(augmented_panel_data$rolled_cases + 1.1)
augmented_panel_data <- augmented_panel_data %>%
  group_by(fips) %>%
    mutate(log_rolled_cases_ratios = c(0, diff(log_rolled_cases)))
augmented_panel_data <- augmented_panel_data %>%
  group_by(fips) %>%
    mutate(shifted_days_from_start = days_from_start - first(days_from_start))
augmented_panel_data <- augmented_panel_data[, colSums(!is.na(augmented_panel_data)) > 0]

# Obtain the time invariant data specific to each fips
#X_time_invariant <- (hhs_X_w_clusters[,intersect(names(augmented_panel_data), names(hhs_X_w_clusters))])
#X_time_invariant <- X_time_invariant[,!names(X_time_invariant) %in% c("county","state")]

# Check the number of command-line arguments
args <- commandArgs(trailingOnly = TRUE)



# SET START AND END
start_day = min(augmented_panel_data$days_from_start)
end_day = max(augmented_panel_data$days_from_start)

# Set default values for start and end if less than 2 arguments are provided
if (length(args) >= 2) {
  start_day <- as.integer(args[1])
  end_day <- as.integer(args[2])
}

cutoff_list <- start_day:end_day
#cutoff_list <- rev(cutoff_list)
print(paste0("Beginning time variant for LLF from ", toString(start_day), " to ", toString(end_day)))


# Step 1: Determine the range in the logarithmic scale
# -5 to 1
log_min <- -5
log_max <- -3

# Step 2: Create a sequence in the logarithmic scale
lambda_exponent_list <- seq(log_min, log_max)  # 100 values for example


# Step 1: Read the CSV file and extract the top 10 feature names
sorted_feature_importance <- read.csv("../TLGRF_Feature_Importance/Sorted_Feature_Importance.csv", header = TRUE)

# Assuming the unnamed column with feature names is the first column:
sorted_features <- sorted_feature_importance[, 1]



foreach(cutoff = (cutoff_list)) %dopar%{
#for(cutoff in (cutoff_list)){
for (lambda_exponent in lambda_exponent_list){
    # Check if result already exists
    lambda_choice = 10^lambda_exponent
    check.file.name <- paste0("time_variant_llf_cutoff=", toString(cutoff), ".rds")
    check.file.full.name <- file.path(grf.outputfolder, check.file.name)
    
    backtest.check.file.name <- paste0("time_variant_llf_block_results_", toString(cutoff),"_lambda_exponent_",toString(lambda_exponent), ".csv")
    backtest.check.file.full.name <- file.path(outputfolder, backtest.check.file.name)

    if (file.exists(check.file.full.name) && file.exists(backtest.check.file.full.name)){
        print(paste0(check.file.name, " and ",  backtest.check.file.name, " exists, skipping"))
        next
    }
    print(paste0("Generate LL Prediction from GRF ", toString(cutoff), " lambda_exponent=", toString(lambda_exponent) ))
    try({ # START TRY
        start_time <- Sys.time()     
        # SLICE THE DATA ACCORDINGLY
        early_data <- augmented_panel_data[augmented_panel_data$days_from_start <= cutoff, ]
        early_data <- early_data[early_data$days_from_start >= max(cutoff-200, 0), ]

        case_number_columns <- c("fips","date","county","state","days_from_start","log_rolled_cases")
        early_data_case_numbers <- early_data[, names(early_data) %in% case_number_columns]
        # Format data to be fed to GRF
        #WYX <- merge(early_data_case_numbers, X_time_invariant, by="fips", all.x=TRUE)
        WYX <- early_data
        WYX <- WYX[order(WYX$fips, WYX$days_from_start),]
        
        Y <- WYX$log_rolled_cases
        W <- WYX$days_from_start
        X <- WYX[, !names(WYX) %in% case_number_columns]
        X <- X %>%
          select_if(is.numeric)
        
        #Load the Base GRF
        llf <- readRDS(check.file.full.name)
        # GENERATE r_LLF estimates
        WYX_test <- WYX %>%
          group_by(fips) %>%
          filter(days_from_start == cutoff)
        X_test <- WYX_test[, !names(WYX_test) %in% case_number_columns]
        X_test <- X_test %>%
          select_if(is.numeric)
        # Step 2: Get the column names from X_test
        X_test_columns <- colnames(X_test)

        # Step 3: Initialize an empty vector to store the valid indices
        valid_indices <- integer(0)

        # Step 4: Loop through sorted features and keep matching until we get 10 valid matches
        for (feature in sorted_features) {
          matched_index <- match(feature, X_test_columns)
          if (!is.na(matched_index)) {
            valid_indices <- c(valid_indices, matched_index)
          }
          if (length(valid_indices) >= 10) {
            break
          }
        }

        # Step 5: Limit to at most 10 valid indices (if necessary)
        valid_indices <- valid_indices[1:10]

        r_LLF <- predict(llf, X_test, linear.correction.variables = valid_indices, ll.lambda=lambda_choice, estimate.variance=TRUE)
        
        # Generate predictions
        indexing_columns <- c("fips","county","state","date", "days_from_start", "rolled_cases", "log_rolled_cases")
        indexing <- WYX_test[, names(WYX) %in% indexing_columns]
        indexing$r_LLF <- r_LLF$predictions
        indexing$var_r_LFF <- r_LLF$variance.estimates
        indexing$LLF_predicted_log_rolled_cases <- indexing$r_LLF*7 + indexing$log_rolled_cases
        
        #output.fname = paste("time_variant_llf_block_results_",toString(cutoff),".csv",sep="")
        #destpath = file.path(outputfolder,output.fname)
        fwrite(indexing, backtest.check.file.full.name, row.names=FALSE)
        end_time <- Sys.time()
    
        time_taken <- end_time - start_time

        print(paste("Time taken for cutoff=",toString(cutoff), " lambda_exponent=", toString(lambda_exponent)," is ",toString(time_taken),sep=""))
        rm(WYX)
        rm(X)
        rm(W)
        rm(cf)
        rm(Y)
        rm(early_data)
        rm(WYX_test)
        rm(X_test)
        rm(indexing)
        gc()
    } # END TRY
    ) # END TRY
}
}



