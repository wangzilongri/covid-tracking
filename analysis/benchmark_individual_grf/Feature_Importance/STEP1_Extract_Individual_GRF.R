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

get_leaf_sizes <- function(test_forest, test_data){
      results_list <- foreach(i = 1:test_forest$`_num_trees`, .combine = "rbind") %dopar% {
      test_tree <- get_tree(test_forest, i)
      node_index <- get_leaf_node(test_tree, test_data)[1]
      leaf_size <- length((test_tree$node[[node_index]])$samples)
      tree_num_samples <- length(test_tree$drawn_samples)
      ratio <- leaf_size / tree_num_samples

      # Return the values as a named vector
      c("Leaf Size" = leaf_size, "Tree Num Samples" = tree_num_samples, "Ratio" = ratio)
    }
    return(results_list)
}
# Convert the results list to a matrix
#fwrite(results_list, "test_leaf.csv", row.names=FALSE)


print("Setting up directory path")
Individual_RDS_directory <- "../../../data/output/individual_county_grf_windowsize=2_numtrees=100"

# Get a list of all subfolder names (integer names)
subfolder_names <- dir_ls(Individual_RDS_directory, type = "dir", recurse = FALSE, full_path = FALSE)

if (!file.exists("./depths")) {
  dir.create("./depths")
}

if (!file.exists("./leafs")) {
  dir.create("./leafs")
}

if (!file.exists("./features")) {
  dir.create("./features")
}

print("Loading Objects")
#loaded_objects <- foreach(subfolder_name = subfolder_names, .errorhandling="pass") %dopar% {
foreach(subfolder_name = subfolder_names, .errorhandling="pass") %dopar% {
  # Get the full path to the subfolder
  directory_name <- basename(strsplit(subfolder_name, "/")[[1]])
  fips_string <- directory_name[length(directory_name)]
  # Get the file names of the RDS objects in the subfolder
  loaded_objects_sub <- foreach(cutoff =seq(70, 1100, by = 30), .errorhandling="pass") %do% {
      fname <- paste0("grf_individual_county_fips=",fips_string,"_cutoff=",toString(cutoff),".rds")
      print(fname)
      rds_file_path <- file.path(subfolder_name, fname)
      # Try reading the RDS file and extract depth
      tryCatch({
        loaded_object <- readRDS(rds_file_path)
        # Check if depths data already written
        depths_vector_fpath <- paste0("./depths/depths_fips=",fips_string,"_cutoff=",cutoff,".csv")
        if (!file.exists(depths_vector_fpath)){
          print(paste0("Writing depths for ", fname, " to ", depths_vector_fpath))
          depths_vector <- forest_depths(loaded_object)
          fwrite(depths_vector, depths_vector_fpath)
        }
        # Check if leaf data already written
        leaf_matrix_fpath <- paste0("./leafs/leafs_fips=",fips_string,"_cutoff=",cutoff,".csv")
        if (!file.exists(leaf_matrix_fpath)){
          print(paste0("Writing leafs for ", fname, " to ", leaf_matrix_fpath))
          test_forest <- loaded_object
          orig_training_X = test_forest$`X.orig`
          test_data <- orig_training_X[orig_training_X$cutoff == max(orig_training_X$cutoff),]
          results_list <- get_leaf_sizes(test_forest, test_data)

          fwrite(results_list, leaf_matrix_fpath, row.names=FALSE)
        }
        # Check if feature importance already written
        features_vector_fpath <- paste0("./features/feature_importance_fips=",fips_string,"_cutoff=",cutoff,".csv")
        if (!file.exists(features_vector_fpath)){
          print(paste0("Writing feature importances for ", fname, " to ", features_vector_fpath))
          feature_importance <- t(grf::variable_importance(loaded_object))
          feature_names <- colnames(loaded_object$X.orig)
          colnames(feature_importance) <- feature_names

          fwrite(feature_importance, features_vector_fpath, row.names=FALSE)
        }        
        #feature_importance
      }, error = function(e) {
        NULL
      })
  }
  #compact(loaded_objects_sub)
}
print("Done!")
break
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