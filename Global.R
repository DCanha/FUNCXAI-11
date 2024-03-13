# Load necessary libraries for XAI methods and data manipulation
library(iml)         # For Shapley explanations
library(shapper)     # For SHAP explanations
library(ciu)         # For CIU explanations
library(caret)       # For model training and prediction
library(ggplot2)     # For plotting

# The goal of this script is to compute global explanations for specific Explainable AI (XAI) methods.
# These methods include CIU, Shapley, and SHAP values. The script can also
# generate global importance plots for each feature, based on their contribution to the model's predictions.

# Function to plot global feature importances
plot_features <- function(globExp) {
  # Ensure globExp has rownames as a column for plotting
  globExp$Feature <- rownames(globExp)
  
  # Ensure the first column in globExp is the global explanation value
  # Ensure the second column in globExp is the sd value
  
  # Create the ggplot
  p <- ggplot(globExp, aes(x = Feature, y = globExp[,1])) +
    geom_point(aes(y = globExp[,1]), color = "cadetblue", size = 3) +  # Add dots for global importance
    geom_errorbar(aes(ymin = globExp[,1] - globExp[,2], ymax = globExp[,1] + globExp[,2]),   # Add error bars
                  width = 0.3, color = "cadetblue3") +
    coord_flip() +  # Flip coordinates to make the plot horizontal
    labs(title = "Global Feature Importance", x = "Feature", y = paste0("Importance (", colnames(globExp)[1], ")")) +
    theme_minimal() +  # Use a minimal theme
    theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Adjust text angle for x-axis labels
  
  print(p)
}
plot_features <- function(globExp) {
  # Ensure globExp has rownames as a column for plotting
  globExp$Feature <- rownames(globExp)
  
  # Create the ggplot
  p <- ggplot(globExp, aes(x = Feature, y = Global_Importance)) +
    geom_point(aes(y = Global_Importance), color = "blue", size = 3) +  # Add dots for global importance
    geom_errorbar(aes(ymin = Global_Importance - SD, ymax = Global_Importance + SD),   # Add error bars
                  width = 0.2, color = "red") +
    coord_flip() +  # Flip coordinates to make the plot horizontal
    labs(title = "Global Feature Importance", x = "Feature", y = "Importance") +
    theme_minimal() +  # Use a minimal theme
    theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Adjust text angle for x-axis labels
  
  print(p)
}

# These take quite a long time if for entire training data

# Function to compute global 
globalCIU <- function(data, explainer, ninst = 200, class.idx = 1) {
  # Computes global Contextual Importance (CI) values 
  # Args:
  #   data: Data frame of training data without the label column
  #   explainer: CIU explainer object
  #   ninst: Number of instances to include from data, defaults to 200
  #   class.idx: Index of the class to study, defaults to 1 (for regression)
  # Returns:
  #   A data frame with mean CI values for all features across instances and corresponding standard deviation
  
  ninst <- min(ninst, nrow(data))
  nimps <- ncol(data)
  gCI <- matrix(0, nrow = ninst, ncol = nimps)
  instinds <- round(runif(ninst, 1, nrow(data)))
  
  for (inst in 1:ninst) {
    ciu.meta <- explainer$meta.explain(data[instinds[inst],])
    for (inp in 1:nimps) {
      gCI[inst,inp] <- ciu.list.to.frame(ciu.meta$ciuvals, out.ind = class.idx)[inp,]$CI
    }
    # Normalizes the importance values so their sum equals 1
    gCI[inst,] <- gCI[inst,]/sum(gCI[inst,])
  }
  
  cmeans <- colMeans(gCI)
  gCIU <- data.frame(CI=cmeans, sd= apply(gCI, 2, sd))
  rownames(gCIU) <- colnames(data)
  return(gCIU)
}

# Function to compute global Shapley values
globalShapley <- function(data, predictor, ninst = 200, class.idx = 1) {
  # Computes global Shapley values 
  # Args:
  #   data: Data frame of training data without the label column
  #   predictor: Predictor object used to compute model probabilities (from iml)
  #   ninst: Number of instances to include from data, defaults to 200
  #   class.idx: Index of the class to study, defaults to 1 (for regression)
  # Returns:
  #   A data frame with mean Shapley values for all features across instances and corresponding standard deviation
  
  ninst <- min(ninst, nrow(data))
  nimps <- ncol(data) 
  shapvals <- matrix(0, nrow = ninst, ncol = nimps)
  instinds <- round(runif(ninst, 1, nrow(data)))
  
  for (inst in 1:ninst) {
    s <- Shapley$new(predictor, x.interest = data[instinds[inst],])
    shapvals[inst,] <- s$results[(ncol(data)*(class.idx-1)+1):(ncol(data)*class.idx),]$phi
    
    # Normalizes the Shapley values so their sum equals 1
    shapvals[inst,] <- abs(shapvals[inst,])/sum(abs(shapvals[inst,]))
  }
  
  cmeans <- colMeans(shapvals)
  gShapley <- data.frame(phi=cmeans, sd= apply(shapvals, 2, sd))
  rownames(gShapley) <- colnames(data)
  return(gShapley)
}

# Function to compute global SHAP values
globalShap <- function(model, data, predictor, ninst = 200, class.idx = 1) {
  # Computes global SHAP values 
  # Args:
  #   model: Machine learning model for which SHAP values are computed
  #   data: Data frame of training data without the label column
  #   predictor: Function used to compute model predictions
  #   ninst: Number of instances to include from data, defaults to 200
  #   class.idx: Index of the class to study, defaults to 1 (for regression)
  # Returns:
  #   A data frame with mean SHAP values for all features across instances and corresponding standard deviation
  
  ninst <- min(ninst, nrow(data))
  nimps <- ncol(data)
  shapvals <- matrix(0, nrow = ninst, ncol = nimps)
  instinds <- round(runif(ninst, 1, nrow(data)))
  
  for (inst in 1:ninst) {
    s <- individual_variable_effect(model, data = data, predict_function = predictor, new_observation = data[instinds[inst],], nsamples=100)
    shapvals[inst,] <- unlist(s[["_attribution_"]][,1])
    
    # Normalizes the SHAP values so their sum equals 1
    shapvals[inst,] <- abs(shapvals[inst,])/sum(abs(shapvals[inst,]))
  }
  
  cmeans <- colMeans(shapvals)
  gShap <- data.frame(phi=cmeans, sd=apply(shapvals, 2, sd))
  rownames(gShap) <- colnames(data)
  return(gShap)
}

# Note: The script assumes that the necessary predictor and explainer objects/functions have been previously created and trained.
# The functions are designed to work with regression models or classifications models with class label in column 1 by default
# but can be adapted for classification by specifying the class.idx parameter.