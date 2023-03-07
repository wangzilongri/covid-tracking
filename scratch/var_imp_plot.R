var_imp_plot <- function(forest, decay.exponent = 2L, max.depth = 4L) {
  
  split.freq <- split_frequencies(forest, max.depth)
  split.freq <- split.freq / pmax(1L, rowSums(split.freq))
  weight <- seq_len(nrow(split.freq)) ^ -decay.exponent
  var.importance <- t(split.freq) %*% weight / sum(weight)
  
  # Format data frame
  #require(dplyr)
  if (is(forest, 'regression_forest') || is(forest, 'quantile_forest')) {
    p <- ncol(forest$X.orig) - 1L
  } else if (is(forest, 'causal_forest')) {
    p <- ncol(forest$X.orig) - 2L
  } else if (is(forest, 'instrumental_forest')) {
    p <- ncol(forest$X.orig) - 3L
  }
  var.names <- colnames(forest$X.orig)[seq_len(p)]
  if (is.null(var.names)) {
    var.names <- paste0('x', seq_len(p))
  }
  df <- data_frame(Variable = var.names,
                   Importance = as.numeric(var.importance)) %>%
    arrange(Importance) %>% 
    mutate(Variable = factor(Variable, levels = unique(Variable)))
  
  # Plot results
  require(ggplot2)
  p <- ggplot(df, aes(Variable, Importance)) + 
    geom_bar(stat = 'identity') + 
    coord_flip() + 
    ggtitle('Variable Importance Plot') + 
    theme_bw() + 
    theme(plot.title = element_text(hjust = 0.5))
  #print(p)
  
}
