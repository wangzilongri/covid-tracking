list.of.packages <- c("ggplot2", "Rcpp", "grf", "caret", "mltools", "rpart", "minpack.lm", "doParallel", "rattle", "anytime","rlist")
list.of.packages <- c(list.of.packages, "zoo", "dtw", "foreach", "evaluate","rlist","data.table","plyr","dplyr", "fs")


new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, lib='/home/zwang937/local/R_libs', repos="http://cran.us.r-project.org", dependencies = TRUE, INSTALL_opts = '--no-lock')


lapply(list.of.packages, require, character.only = TRUE)

# Register the cluster
registerDoParallel(cores=detectCores())


print("Setting up directory path")
Individual_RDS_directory <- "../../../data/output/individual_county_grf_windowsize=2_numtrees=100"

# Get a list of all subfolder names (integer names)
subfolder_names <- dir_ls(Individual_RDS_directory, type = "dir", recurse = FALSE, full_path = FALSE)

# Create an empty list to store the loaded objects
#loaded_objects <- list()
print("Loading Objects")
loaded_objects <- foreach(subfolder_name = subfolder_names, .errorhandling="pass") %dopar% {
  # Get the full path to the subfolder
  directory_name <- basename(strsplit(subfolder_name, "/")[[1]])
  fips_string <- directory_name[length(directory_name)]
  # Get the file names of the RDS objects in the subfolder
  #subfolder_feature_importance <- list()
  subfolder_feature_importance <- foreach(cutoff =seq(100, 1000, by = 100), .errorhandling="pass") %do% {
      fname <- paste0("grf_individual_county_fips=",fips_string,"_cutoff=",toString(cutoff),".rds")
      #print(fname)
      rds_file_path <- file.path(subfolder_name, fname)
      # Try reading the RDS file and extract feature importance
      tryCatch({
        loaded_object <- readRDS(rds_file_path)
        feature_importance <- t(grf::variable_importance(loaded_object))
        feature_names <- colnames(loaded_object$X.orig)
        colnames(feature_importance) <- feature_names
        feature_importance
      }, error = function(e) {
        # Return NULL in case of error
        NULL
      })
  }
  #print(subfolder_feature_importance)
  # Return the loaded objects for this subfolder
  #loaded_objects[[subfolder_name]] <- subfolder_feature_importance
  compact(subfolder_feature_importance)
}

print("Flattening list of list of matrices")
loaded_objects <- loaded_objects[lengths(loaded_objects) > 0]
unlisted_loaded_objects<-unlist(loaded_objects, recursive = FALSE)


print("Extracting Feature Importance")
result_matrix <- matrix(NA, nrow = length(unlisted_loaded_objects), ncol = lengths(unlisted_loaded_objects)[1])
for (i in 1:length(unlisted_loaded_objects)) {
  result_matrix[i, ] <- as.vector(unlisted_loaded_objects[[i]])
}

colnames(result_matrix) <- colnames(unlisted_loaded_objects[[1]])
my_datatable <- data.table(result_matrix)

print("Writing my_datatable")
fwrite(my_datatable, "individual_grf_feature_importance.csv", row.names=FALSE)