list.of.packages <- c("ggplot2", "Rcpp", "grf", "caret", "mltools", "rpart", "minpack.lm", "doParallel", "rattle", "anytime","rlist")
list.of.packages <- c(list.of.packages, "zoo", "dtw", "foreach", "evaluate","rlist","data.table","plyr","dplyr", "fs")


new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, lib='/home/zwang937/local/R_libs', repos="http://cran.us.r-project.org", dependencies = TRUE, INSTALL_opts = '--no-lock')


lapply(list.of.packages, require, character.only = TRUE)

# Register the cluster
registerDoParallel(cores=detectCores())


dfs_depth <- function(tree_nodes_list, node_index = 1, current_depth = 1) {
  # Base case: if the node is a leaf, return the current depth
  if (tree_nodes_list[[node_index]]$is_leaf) {
    return(current_depth)
  }
  
  # Recursive case: calculate depth of the child nodes
  left_child_index <- tree_nodes_list[[node_index]]$left_child
  right_child_index <- tree_nodes_list[[node_index]]$right_child
  
  left_depth <- dfs_depth(tree_nodes_list, left_child_index, current_depth + 1)
  right_depth <- dfs_depth(tree_nodes_list, right_child_index, current_depth + 1)
  
  # Return the maximum depth among the child nodes
  return(max(left_depth, right_depth))
}

forest_depths <- function(grf_forest){
    forest_depths_vector <- foreach(i = 1:grf_forest$`_num_trees`) %dopar%{
        test_tree <- (grf::get_tree(grf_forest,i))
        test_tree_nodes <- test_tree$nodes
        dfs_depth(test_tree_nodes)
    }
    return(forest_depths_vector)
}


print("Setting up directory path")
Individual_RDS_directory <- "../../../data/output/individual_county_grf_windowsize=2_numtrees=100"

# Get a list of all subfolder names (integer names)
subfolder_names <- dir_ls(Individual_RDS_directory, type = "dir", recurse = FALSE, full_path = FALSE)

print("Loading Objects")
loaded_objects <- foreach(subfolder_name = subfolder_names, .errorhandling="pass") %dopar% {
  # Get the full path to the subfolder
  directory_name <- basename(strsplit(subfolder_name, "/")[[1]])
  fips_string <- directory_name[length(directory_name)]
  # Get the file names of the RDS objects in the subfolder
  loaded_objects_sub <- foreach(cutoff =seq(100, 1100, by = 30), .errorhandling="pass") %do% {
      fname <- paste0("grf_individual_county_fips=",fips_string,"_cutoff=",toString(cutoff),".rds")
      print(fname)
      rds_file_path <- file.path(subfolder_name, fname)
      # Try reading the RDS file and extract depth
      tryCatch({
        loaded_object <- readRDS(rds_file_path)
        depths_vector <- forest_depths(loaded_object)
        print(paste0("Writing depths for ", fname))
        fwrite(depths_vector, paste0("./depths/depths_fips=",fips_string,"_cutoff=",cutoff,".csv"))
        print(paste0("Getting feature importance for ", fname))
        feature_importance <- t(grf::variable_importance(loaded_object))
        feature_names <- colnames(loaded_object$X.orig)
        colnames(feature_importance) <- feature_names
        feature_importance
      }, error = function(e) {
        NULL
      })
  }
  compact(loaded_objects_sub)
}
print("Done!")

print("Flattening list of list of matrices")
loaded_objects <- loaded_objects[lengths(loaded_objects) > 0]
unlisted_loaded_objects<-unlist(loaded_objects, recursive = FALSE)
unlisted_loaded_objects[lengths(unlisted_loaded_objects) != 467]

print("Extracting Feature Importance")
result_matrix <- matrix(NA, nrow = length(unlisted_loaded_objects), ncol = lengths(unlisted_loaded_objects)[1])
for (i in 1:length(unlisted_loaded_objects)) {
  result_matrix[i, ] <- as.vector(unlisted_loaded_objects[[i]])
}

colnames(result_matrix) <- colnames(unlisted_loaded_objects[[1]])
my_datatable <- data.table(result_matrix)

print("Writing my_datatable")
fwrite(my_datatable, "individual_grf_feature_importance.csv", row.names=FALSE)