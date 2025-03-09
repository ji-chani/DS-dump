# Set working directory (modify as needed)
setwd("/Users/jomarrabajante/Downloads/255 code/MNIST_CSV")

# Install required packages if not already installed
required_packages <- c("e1071", "randomForest", "caret", "moments", "OpenImageR", "fourierin", "pracma", "magick")
install.packages(setdiff(required_packages, installed.packages()[,"Package"]))

# Load necessary libraries
library(e1071)        # Support Vector Machine (SVM)
library(randomForest) # Random Forest classifier
library(caret)        # Model training utilities
library(moments)      # Statistical moments
library(OpenImageR)   # Image processing
library(fourierin)    # Fourier descriptors
library(pracma)       # Mathematical utilities
library(magick)       # Image processing alternative

# FUNCTIONS

## Compute Hu Moments (Rotation Invariant Features)
compute_hu_moments <- function(image_matrix) {
  x_mean <- sum(row(image_matrix) * image_matrix) / sum(image_matrix)
  y_mean <- sum(col(image_matrix) * image_matrix) / sum(image_matrix)
  
  mu <- function(p, q) sum(((row(image_matrix) - x_mean) ^ p) * ((col(image_matrix) - y_mean) ^ q) * image_matrix)
  
  eta <- function(p, q) mu(p, q) / (mu(0, 0) ^ (((p + q) / 2) + 1))
  
  hu_moments <- c(
    eta(2, 0) + eta(0, 2),
    (eta(2, 0) - eta(0, 2))^2 + (2 * eta(1, 1))^2,
    (eta(3, 0) - 3 * eta(1, 2))^2 + (3 * eta(2, 1) - eta(0, 3))^2,
    (eta(3, 0) + eta(1, 2))^2 + (eta(2, 1) + eta(0, 3))^2,
    (eta(3, 0) - 3 * eta(1, 2)) * (eta(3, 0) + eta(1, 2)) * ((eta(3, 0) + eta(1, 2))^2 - 3 * (eta(2, 1) + eta(0, 3))^2),
    (eta(2, 0) - eta(0, 2)) * ((eta(3, 0) + eta(1, 2))^2 - (eta(2, 1) + eta(0, 3))^2),
    (3 * eta(2, 1) - eta(0, 3)) * (eta(3, 0) + eta(1, 2)) * ((eta(3, 0) + eta(1, 2))^2 - 3 * (eta(2, 1) + eta(0, 3))^2)
  )
  
  return(hu_moments)
}

## Compute Zernike Moments (Invariant Shape Features)
compute_zernike_moments <- function(image_matrix, order = 8) {
  center_x <- 14
  center_y <- 14
  radius <- 14
  
  # Normalize coordinate system
  x_grid <- matrix(rep(1:28, each = 28), ncol = 28) - center_x
  y_grid <- matrix(rep(1:28, times = 28), ncol = 28) - center_y
  r <- sqrt(x_grid^2 + y_grid^2) / radius
  theta <- atan2(y_grid, x_grid)
  
  # Mask pixels outside the circular region
  valid_mask <- r <= 1
  image_matrix[!valid_mask] <- 0
  
  # Compute Zernike moments (simplified)
  zernike_values <- unlist(lapply(0:order, function(n) {
    lapply(seq(-n, n, 2), function(m) {
      sum((r^n * cos(m * theta)) * image_matrix) / sum(valid_mask)
    })
  }))
  
  return(as.numeric(zernike_values[1:10]))  # Ensure numeric output
}

## Harris Corner Detection (Alternative)
compute_corner_density <- function(image_matrix) {
  image_matrix <- image_matrix / max(image_matrix)
  harris_kernel <- matrix(c(-1, -1, -1, -1, 8, -1, -1, -1, -1), nrow = 3)
  corner_response <- filter(image_matrix, harris_kernel, circular = TRUE)
  return(sum(corner_response > quantile(corner_response, 0.95)) / length(image_matrix))
}

## Compute Fourier Descriptors
compute_fourier_descriptors <- function(image_matrix) {
  fourier_coeffs <- fft(image_matrix)
  return(c(mean(Mod(fourier_coeffs)), sd(Mod(fourier_coeffs))))
}

# Load the MNIST dataset (CSV format)
mnist <- read.csv("mnist_test.csv", header = FALSE)
colnames(mnist)[1] <- "label"
mnist$label <- as.factor(mnist$label)

## Feature Extraction Function
extract_features <- function(image_vector) {
  image_matrix <- matrix(image_vector, nrow = 28, ncol = 28) / 255
  
  sobel_x <- matrix(c(-1, 0, 1, -2, 0, 2, -1, 0, 1), nrow = 3)
  sobel_y <- t(sobel_x)
  edge_x <- filter(image_matrix, sobel_x, circular = TRUE)
  edge_y <- filter(image_matrix, sobel_y, circular = TRUE)
  texture_feature <- mean(sqrt(edge_x^2 + edge_y^2))
  
  moment_features <- c(moment(image_vector, order = 1:4))
  hu_values <- compute_hu_moments(image_matrix)
  zernike_values <- compute_zernike_moments(image_matrix)
  corner_density <- compute_corner_density(image_matrix)
  fourier_values <- compute_fourier_descriptors(image_matrix)
  
  return(c(texture_feature, moment_features, hu_values, corner_density, fourier_values, zernike_values))
}

# Apply feature extraction
feature_matrix <- t(apply(mnist[,-1], 1, extract_features))
feature_data <- data.frame(feature_matrix, label = mnist$label)

# Split dataset
set.seed(123)
trainIndex <- createDataPartition(feature_data$label, p = 0.7, list = FALSE)
train_data <- feature_data[trainIndex, ]
test_data <- feature_data[-trainIndex, ]

# Train classifiers
svm_model <- svm(label ~ ., data = train_data, kernel = "linear")
rf_model <- randomForest(label ~ ., data = train_data, ntree = 100, importance = TRUE)

# Predictions
svm_pred <- predict(svm_model, test_data)
rf_pred <- predict(rf_model, test_data)

# Accuracy
svm_accuracy <- mean(svm_pred == test_data$label)
rf_accuracy <- mean(rf_pred == test_data$label)

print(paste("SVM Accuracy:", round(svm_accuracy * 100, 2), "%"))
print(paste("Random Forest Accuracy:", round(rf_accuracy * 100, 2), "%"))

# External Data Prediction
new_data <- read.csv("new_data_image.csv", header = FALSE)
new_features <- t(apply(new_data, 1, extract_features))
new_predictions <- data.frame(
  SVM_Predicted = predict(svm_model, data.frame(new_features)),
  RF_Predicted = predict(rf_model, data.frame(new_features))
)

# Compute Feature Importance
importance_values <- importance(rf_model)
feature_importance <- data.frame(
  Feature = rownames(importance_values),
  Importance = importance_values[, 1]  # Mean Decrease in Accuracy
)

# Sort by Importance
feature_importance <- feature_importance[order(-feature_importance$Importance), ]

# Visualize Feature Importance
ggplot(feature_importance, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance (Random Forest)", x = "Feature", y = "Importance Score") +
  theme_minimal()

# Display Feature Importance Values
print(feature_importance)

# Feature Descriptions (Total Features: 25)
#
# 1. **Texture Feature (Sobel Edge Detection) → 1 Feature (X1)**
#    - X1: Edge Magnitude (average stroke thickness)
#    - Detects stroke thickness & structure of digits
#    - Helps differentiate '1' (thin) from '0' (round)
#
# 2. **Statistical Moments → 4 Features (X2 to X5)**
#    - X2: Mean (average brightness)
#    - X3: Variance (spread of intensity values)
#    - X4: Skewness (asymmetry in intensity)
#    - X5: Kurtosis (peakedness/sharpness of edges)
#    - Captures **brightness, contrast, tilt, and edge sharpness**.
#
# 3. **Hu Moments (Rotation-Invariant Shape Features) → 7 Features (X6 to X12)**
#    - X6 to X12: Hu Moments 1 to 7 (Invariant to rotation, scale, and translation)
#    - Helps recognize rotated digits (e.g., '6' vs. '9')
#    - Captures **global shape properties** of digits.
#
# 4. **Zernike Moments (Curvature & Symmetry) → 10 Features (X13 to X22)**
#    - X13 to X22: Zernike Moments 1 to 10
#    - Differentiates symmetrical numbers (e.g., '8' vs. '3')
#    - Works better for **round vs. angular digits**.
#
# 5. **Harris Corner Detection (Corner Density) → 1 Feature (X23)**
#    - X23: Density of corners in the image
#    - Counts the number of sharp bends
#    - Useful for distinguishing '4' (many corners) vs. '0' (few corners).
#
# 6. **Fourier Descriptors (Frequency-Based Shape Analysis) → 2 Features (X24, X25)**
#    - X24: Mean of frequency components
#    - X25: Standard deviation of frequency components
#    - Extracts global shape patterns
#    - Captures **differences in digit strokes** (e.g., '1' vs. '0').
#
# **Total Features Used in Classification: 25**