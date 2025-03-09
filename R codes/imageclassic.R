# Set working directory (modify as needed)
setwd("/Users/jomarrabajante/Downloads/255 code/MNIST_CSV")

# Install required packages if not already installed
required_packages <- c("e1071", "randomForest", "caret", "moments")
install_if_missing <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]
if (length(install_if_missing)) install.packages(install_if_missing)

# Load necessary libraries
library(e1071)        # Support Vector Machine (SVM)
library(randomForest) # Random Forest classifier
library(caret)        # Model training and evaluation utilities
library(moments)      # Statistical moments for shape feature extraction

# Load the MNIST dataset (CSV format)
mnist <- read.csv("mnist_test.csv", header = FALSE)
#Rows: Each row in the CSV file represents a single image.​
#Columns: The first column contains the label (digit 0–9), indicating the digit represented in the image. The subsequent 784 columns correspond to the pixel values of the 28x28 image, listed in row-major order.
#Pixel Values: Each pixel value ranges from 0 to 255, representing the grayscale intensity of that pixel.

# Ensure the first column is the label (digits 0-9)
colnames(mnist)[1] <- "label"  # Rename first column to "label" (ground truth)
mnist$label <- as.factor(mnist$label)  # Convert label to categorical factor

### Feature Extraction Function ###
# Extracts texture and shape features from each 28x28 image
extract_features <- function(image_vector) {
  # Convert the 1D vector (784 pixels) to a 28x28 matrix and normalize to [0,1]
  image_matrix <- matrix(image_vector, nrow = 28, ncol = 28) / 255  
  
  ## --- Texture Extraction: Sobel Edge Detection ---
  # Sobel filters for detecting horizontal and vertical edges
  sobel_x <- matrix(c(-1, 0, 1, -2, 0, 2, -1, 0, 1), nrow = 3, ncol = 3)
  sobel_y <- t(sobel_x)
  
  # Apply Sobel filters
  edge_x <- filter(image_matrix, sobel_x, circular = TRUE)
  edge_y <- filter(image_matrix, sobel_y, circular = TRUE)
  
  # Compute edge magnitude (gradient intensity)
  edge_magnitude <- sqrt(edge_x^2 + edge_y^2)
  texture_feature <- mean(edge_magnitude)  # Average edge strength as texture descriptor
  
  ## --- Shape Features: Statistical Moments ---
  moment1 <- moment(image_vector, order = 1)  # Mean intensity
  moment2 <- moment(image_vector, order = 2)  # Variance (spread of intensity)
  moment3 <- moment(image_vector, order = 3)  # Skewness (asymmetry)
  moment4 <- moment(image_vector, order = 4)  # Kurtosis (sharpness of peaks)
  
  # Return extracted features as a vector
  return(c(texture_feature, moment1, moment2, moment3, moment4))
}

# Apply feature extraction to all images
feature_matrix <- t(apply(mnist[,-1], 1, extract_features))  
feature_data <- data.frame(feature_matrix, label = mnist$label)

### Split Data into Training and Testing ###
set.seed(123)  # For reproducibility
trainIndex <- createDataPartition(feature_data$label, p = 0.7, list = FALSE)  
train_data <- feature_data[trainIndex, ]  
test_data <- feature_data[-trainIndex, ]  

### Train Classifiers ###
# Support Vector Machine (SVM) with Linear Kernel
svm_model <- svm(label ~ ., data = train_data, kernel = "linear")

# Random Forest (RF) with 100 decision trees
rf_model <- randomForest(label ~ ., data = train_data, ntree = 100)

### Predict on Test Set ###
svm_pred <- predict(svm_model, test_data)  # SVM predictions
rf_pred <- predict(rf_model, test_data)  # Random Forest predictions

### Model Accuracy Calculation ###
svm_accuracy <- sum(svm_pred == test_data$label) / nrow(test_data)
rf_accuracy <- sum(rf_pred == test_data$label) / nrow(test_data)

### Display Accuracy ###
print(paste("SVM Accuracy:", round(svm_accuracy * 100, 2), "%"))
print(paste("Random Forest Accuracy:", round(rf_accuracy * 100, 2), "%"))

### Create a Table of Predictions ###
# Combine actual labels and predicted labels for visualization
results_table <- data.frame(
  Actual = test_data$label,
  SVM_Predicted = svm_pred,
  RF_Predicted = rf_pred
)

### External Dataset Prediction (New Data) ###
# Load new dataset (assumed to have the same format as MNIST)
new_data <- read.csv("new_data_image.csv", header = FALSE)

# Ensure correct column structure (no labels, only pixel data)
new_features <- t(apply(new_data, 1, extract_features))
new_features_df <- data.frame(new_features)

# Make predictions using trained models
new_svm_pred <- predict(svm_model, new_features_df)
new_rf_pred <- predict(rf_model, new_features_df)

# Store predictions in a table
new_predictions <- data.frame(
  SVM_Predicted = new_svm_pred,
  RF_Predicted = new_rf_pred
)
