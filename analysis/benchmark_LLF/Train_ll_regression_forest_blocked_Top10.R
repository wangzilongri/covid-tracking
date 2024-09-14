# Trains the Base GRFs for LLF prediction

list.of.packages <- c("ggplot2", "Rcpp", "grf", "caret", "mltools", "rpart", "minpack.lm", "doParallel", "rattle", "anytime","rlist")
list.of.packages <- c(list.of.packages, "zoo", "dtw", "foreach", "evaluate","rlist","data.table","plyr","dplyr")


new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, lib='/home/zwang937/local/R_libs', repos="http://cran.us.r-project.org", dependencies = TRUE, INSTALL_opts = '--no-lock')


lapply(list.of.packages, require, character.only = TRUE)

# Register Parallel Backend for dopar
registerDoParallel(cores=min(detectCores(), 5))


# SET PARAMS
windowsize=2
num_trees = 200



# CREATE OUTPUT FOLDERS
mainDir = "./ll_regression_forest_blocked_Top10"
dir.create(mainDir, showWarnings = FALSE)

subDir = paste("llf_backtest_state_forests_windowsize=",toString(windowsize),"_numtrees=",toString(num_trees),sep="")
outputfolder = file.path(mainDir, subDir)


dir.create(outputfolder, showWarnings = FALSE)

grf.subfolder = paste("llf_windowsize=",toString(windowsize),"_numtrees=",toString(num_trees),sep="")
grf.outputfolder = file.path(mainDir,grf.subfolder)
dir.create(grf.outputfolder, showWarnings = FALSE)

# READ DATA
print("Reading in data")
destfile <- paste("imputed_shifted_block.csv")
augmented_panel_data <- as.data.frame(fread(file = destfile))
# Time variant
#hhs_X_w_clusters_fpath = "../benchmark_tcv_kmeans_code/hhs_X_w_clusters.csv"
#hhs_X_w_clusters = as.data.frame(fread(file = hhs_X_w_clusters_fpath))

print("Transforming augmented_panel_data")
# Check the number of command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Step 1: Read the CSV file and extract the top 10 feature names
sorted_feature_importance <- read.csv("../TLGRF_Feature_Importance/Sorted_Feature_Importance.csv", header = TRUE)

# Assuming the unnamed column with feature names is the first column:
sorted_features <- sorted_feature_importance[, 1]


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

foreach(cutoff = (cutoff_list)) %dopar%{
#for(cutoff in (cutoff_list)){
    # Check if result already exists
    check.file.name <- paste0("ll_regression_forest_cutoff=", toString(cutoff), ".rds")
    check.file.full.name <- file.path(grf.outputfolder, check.file.name)
    
    backtest.check.file.name <- paste0("ll_regression_forest_block_results_", toString(cutoff), ".csv")
    backtest.check.file.full.name <- file.path(outputfolder, backtest.check.file.name)

    if (file.exists(backtest.check.file.full.name)){
        print(paste0(backtest.check.file.name, " exists, skipping!"))
        next
    }
    print(paste0("Computing LLF for ", toString(cutoff)))
    try({ # START TRY
        start_time <- Sys.time()     
        # SLICE THE DATA ACCORDINGLY
        early_data <- augmented_panel_data[augmented_panel_data$days_from_start <= cutoff, ]
        early_data <- early_data[early_data$days_from_start >= max(cutoff-400, 0), ]
        
        cutoff_parity <- cutoff %% 2

        # Filter the dataframe based on the parity of t
        early_data <- early_data %>%
          filter((days_from_start %% 2) == cutoff_parity)

        
        case_number_columns <- c("fips","date", "datetime","county","state","days_from_start", "shifted_log_rolled_cases","log_rolled_cases", "rolled_cases")
        early_data_case_numbers <- early_data[, names(early_data) %in% case_number_columns]
        # Format data to be fed to ll_regression_forest
        #WYX <- merge(early_data_case_numbers, X_time_invariant, by="fips", all.x=TRUE)
        
        if (file.exists(check.file.full.name)){
            print(paste0(check.file.name, " exists, loading instead of training!"))
            llf <- readRDS(check.file.full.name)
        }
        else{
            XY <- early_data
            XY <- XY[order(XY$fips, XY$days_from_start),]

            Y <- XY$log_shifted_rolled_cases
            X <- XY[, !names(XY) %in% case_number_columns]
            X <- X %>%
              select_if(is.numeric)
            
            # Step 2: Get the column names from X_test
            X_columns <- colnames(X)

            # Step 3: Initialize an empty vector to store the valid indices
            valid_indices <- integer(0)

            # Step 4: Loop through sorted features and keep matching until we get 10 valid matches
            for (feature in sorted_features) {
              matched_index <- match(feature, X_columns)
              if (!is.na(matched_index)) {
                valid_indices <- c(valid_indices, matched_index)
              }
              if (length(valid_indices) >= 10) {
                break
              }
            }

            # Step 5: Limit to at most 10 valid indices (if necessary)
            valid_indices <- valid_indices[1:10]
            #print(covariates[,valid_indices])
            
            
            llf <- ll_regression_forest(X[ ,valid_indices],Y, num.trees=num_trees)
            print(paste0("Saving ", check.file.full.name))
            saveRDS(llf, check.file.full.name)
        }
        
        
        XY_test = early_data[early_data$days_from_start == cutoff, ]
        XY_test <- XY_test[order(XY_test$fips, XY_test$days_from_start),]
        
        X_test <- XY_test[, !names(XY_test) %in% case_number_columns]
        X_test <- X_test %>%
          select_if(is.numeric)
        
        r_LLF <- predict(llf, X_test[ ,valid_indices], linear.correction.variables = 1:ncol(X_test[ ,valid_indices]), estimate.variance=TRUE)

        indexing_columns <- c("fips","county","state","date", "days_from_start", "rolled_cases", "log_rolled_cases")
        indexing <- XY_test[, names(XY_test) %in% indexing_columns]
        indexing$r_LLF <- r_LLF$predictions
        indexing$var_r_LFF <- r_LLF$variance.estimates
        indexing$LLF_predicted_log_rolled_cases <- indexing$r_LLF*7 + indexing$log_rolled_cases
        
        fwrite(indexing, backtest.check.file.full.name, row.names=FALSE)
        end_time <- Sys.time()
    
        time_taken <- end_time - start_time
        print(paste("Time taken for cutoff=",toString(cutoff)," is ",toString(time_taken),sep=""))
        rm(XY)
        rm(X)
        rm(llf)
        rm(Y)
        rm(early_data)
        rm(XY_test)
        rm(X_test)
        rm(Y_test)
        rm(indexing)
        gc()
    } # END TRY
    ) # END TRY
}



