list.of.packages <- c("ggplot2", "Rcpp", "grf", "caret", "mltools", "rpart", "minpack.lm", "doParallel", "rattle", "anytime","rlist")
list.of.packages <- c(list.of.packages, "zoo", "dtw", "foreach", "evaluate","rlist","data.table","plyr","dplyr")


new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, lib='/home/zwang937/local/R_libs', repos="http://cran.us.r-project.org", dependencies = TRUE, INSTALL_opts = '--no-lock')


lapply(list.of.packages, require, character.only = TRUE)


# Initialize a cluster for parallel processing
#cl <- makeCluster(detectCores())

# Register the cluster
registerDoParallel(core=detectCores())

TLGRF_RDS_directory <- "../time_variant_grf_results/time_variant_grf_windowsize=2_numtrees=100"
file_pattern <- "time_variant_grf_stateforest_cutoff=%d.rds"
days_from_start_list <-seq(100, 1000, by=100)
# Define a function to load RDS objects
load_rds <- function(day) {
  rds_file <- sprintf(file_pattern, day)
  print(paste0("Loading ", rds_file))
  readRDS(file.path(TLGRF_RDS_directory, rds_file))
}

# Load RDS objects in parallel
results <- foreach(day = days_from_start_list, .packages = "utils") %dopar% {
  print(paste0("Loading day=", as.character(day)))
  grf_object <- load_rds(day)
  grf::variable_importance(grf_object)
}




print("Loading one file")
grf_object <- load_rds(100)  # Assuming you have already loaded the RDS object

# Extract the names of X.orig
names <- colnames(grf_object$X.orig)
names

result_matrix <- matrix(NA, nrow = length(days_from_start_list), ncol = length(names))
for (i in 1:length(results)) {
  result_matrix[i, ] <- as.vector(results[[i]])
}
colnames(result_matrix) <- names


fwrite(result_matrix, "time_variant_feature_importance.csv", row.names=FALSE)


closeAllConnections()
