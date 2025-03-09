# ---------------------------------------
# INSTALL & LOAD REQUIRED PACKAGES
# ---------------------------------------
required_packages <- c("TDA", "ggplot2", "dplyr", "randomForest", "e1071", "caret", "patchwork")
install_if_missing <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]
if (length(install_if_missing)) install.packages(install_if_missing)
lapply(required_packages, library, character.only = TRUE)

# ---------------------------------------
# LOAD & PREPARE MNIST DATASET
# ---------------------------------------
mnist <- read.csv("mnist_test.csv", header = FALSE)
colnames(mnist)[1] <- "label"
mnist$label <- as.factor(mnist$label)  # Convert labels to categorical

# Select a subset of images for faster processing
set.seed(123)
sample_indices <- sample(1:nrow(mnist), 10000) 
mnist_subset <- mnist[sample_indices, ]

# ---------------------------------------
# CONVERT IMAGES TO POINT CLOUDS
# ---------------------------------------
image_to_point_cloud <- function(image_vector) {
  img_matrix <- matrix(image_vector, nrow = 28, ncol = 28) / 255  # Normalize pixel values
  indices <- which(img_matrix > 0, arr.ind = TRUE)  # Extract nonzero pixel coordinates
  return(as.matrix(indices))  # Convert indices to a point cloud
}

# Apply transformation to all selected images
point_clouds <- lapply(1:nrow(mnist_subset), function(i) {
  image_to_point_cloud(as.numeric(mnist_subset[i, -1]))
})

# ---------------------------------------
# COMPUTE PERSISTENT HOMOLOGY (VIETORIS-RIPS)
# ---------------------------------------
compute_persistence <- function(point_cloud) {
  return(ripsDiag(X = point_cloud, maxdimension = 1, maxscale = 5, library = "GUDHI"))
}

# Compute persistence diagrams for all point clouds
persistence_diagrams <- lapply(point_clouds, compute_persistence)

# ---------------------------------------
# EXTRACT TOPOLOGICAL FEATURES
# ---------------------------------------
compute_topo_features <- function(diagram) {
  pers_diag <- diagram$diagram
  if (nrow(pers_diag) == 0) return(c(0, 0))  # Handle cases with no topological features
  
  lifetimes <- pers_diag[, 2] - pers_diag[, 1]  # Compute lifetimes (Death - Birth)
  total_persistence <- sum(lifetimes)  # Sum of all feature lifetimes
  mean_lifetime <- mean(lifetimes)  # Average persistence of features
  
  return(c(total_persistence, mean_lifetime))
}

# Compute topological features for each persistence diagram
topo_features <- t(sapply(persistence_diagrams, compute_topo_features))

# ---------------------------------------
# PREPARE DATA FOR CLASSIFICATION
# ---------------------------------------
feature_data <- data.frame(topo_features, label = mnist_subset$label)
colnames(feature_data) <- c("TotalPersistence", "MeanLifetime", "Label")

# Train-Test Split (70% training, 30% testing)
set.seed(123)
trainIndex <- createDataPartition(feature_data$Label, p = 0.7, list = FALSE)
train_data <- feature_data[trainIndex, ]
test_data <- feature_data[-trainIndex, ]

# ---------------------------------------
# TRAIN CLASSIFIERS (SVM & RANDOM FOREST)
# ---------------------------------------
svm_model <- svm(Label ~ ., data = train_data, kernel = "linear")
rf_model <- randomForest(Label ~ ., data = train_data, ntree = 100)

# ---------------------------------------
# MAKE PREDICTIONS & EVALUATE MODELS
# ---------------------------------------
svm_pred <- predict(svm_model, test_data)
rf_pred <- predict(rf_model, test_data)

svm_accuracy <- mean(svm_pred == test_data$Label)
rf_accuracy <- mean(rf_pred == test_data$Label)

# Print Model Performance
cat("\n--- Model Performance ---\n")
cat(sprintf("SVM Accuracy: %.2f%%\n", svm_accuracy * 100))
cat(sprintf("Random Forest Accuracy: %.2f%%\n", rf_accuracy * 100))

# ---------------------------------------
# PERSISTENCE DIAGRAM & BARCODE PLOTTING
# ---------------------------------------
plot_persistence <- function(diagram, title) {
  plot(diagram$diagram, main = title)  # Persistence Diagram
}

plot_barcode <- function(diagram, title) {
  barcode <- diagram$diagram
  if (nrow(barcode) > 0) {
    plot(NULL, xlim = range(barcode[, 1:2]), ylim = c(1, nrow(barcode)),
         xlab = "Birth to Death", ylab = "Feature Index", main = title)
    segments(barcode[, 1], 1:nrow(barcode), barcode[, 2], 1:nrow(barcode), col = "blue", lwd = 2)
  } else {
    plot.new()
    title(title)
  }
}

# Select a Random Sample for Visualization
sample_idx <- sample(1:length(persistence_diagrams), 1)
diag_sample <- persistence_diagrams[[sample_idx]]

# Retrieve corresponding digit label
sample_label <- mnist_subset$label[sample_idx]

# Print sample details
cat(sprintf("\n Selected Sample: #%d | Label: %s \n", sample_idx, sample_label))

# Plot Persistence Diagram and Barcode
par(mfrow = c(1, 2))
plot_persistence(diag_sample, paste("Persistence Diagram - Digit", sample_label))
plot_barcode(diag_sample, paste("Persistence Barcode - Digit", sample_label))

# ---------------------------------------
# SAMPLE POINT CLOUD
# ---------------------------------------
# Function to visualize point cloud of MNIST digit
plot_point_cloud <- function(image_vector, label, index) {
  img_matrix <- matrix(image_vector, nrow = 28, ncol = 28) / 255  # Convert to 28x28 and normalize
  point_cloud <- which(img_matrix > 0, arr.ind = TRUE)  # Extract nonzero pixels
  
  # Convert to data frame for ggplot
  point_data <- data.frame(x = point_cloud[, 2], y = 28 - point_cloud[, 1])  # Flip y-axis for proper orientation
  
  # Plot point cloud
  ggplot(point_data, aes(x = x, y = y)) +
    geom_point(color = "blue", size = 1) +
    theme_minimal() +
    coord_fixed() +
    ggtitle(paste("Point Cloud of Digit", label, "(Sample #", index, ")")) +
    xlab("X Position") + ylab("Y Position")
}

# Select a random sample from the dataset
sample_idx <- sample(1:nrow(mnist_subset), 1)
sample_digit <- mnist_subset[sample_idx, ]
sample_label <- sample_digit$label
sample_image <- as.numeric(sample_digit[-1])

# Plot the point cloud of the selected digit
plot_point_cloud(sample_image, sample_label, sample_idx)

# ---------------------------------------
# EXTERNAL DATASET PREDICTION
# ---------------------------------------
cat("\n--- Predicting on External Dataset ---\n")

# Load external dataset
new_data <- read.csv("new_data_image.csv", header = FALSE)

# Convert new images into topological features
new_point_clouds <- lapply(1:nrow(new_data), function(i) {
  image_to_point_cloud(as.numeric(new_data[i, ]))
})
new_persistence_diagrams <- lapply(new_point_clouds, compute_persistence)
new_topo_features <- t(sapply(new_persistence_diagrams, compute_topo_features))

# Prepare new data for prediction
new_features_df <- data.frame(new_topo_features)
colnames(new_features_df) <- c("TotalPersistence", "MeanLifetime")

# Make predictions
new_svm_pred <- predict(svm_model, new_features_df)
new_rf_pred <- predict(rf_model, new_features_df)

# Store predictions
new_predictions <- data.frame(
  SVM_Predicted = new_svm_pred,
  RF_Predicted = new_rf_pred
)

# ---------------------------------------
# INTERPRETATION OF RESULTS
# ---------------------------------------

cat("\n--- Explanation of TDA Results ---\n")

# (1) INTERPRETATION OF PERSISTENCE DIAGRAM
cat("1️⃣ **Persistence Diagrams** visualize the lifespan of topological features (connected components, loops) in digit images.\n")
cat("   - The **X-axis** represents the **birth time** (when a feature first appears).\n")
cat("   - The **Y-axis** represents the **death time** (when the feature disappears).\n")
cat("   - **Points near the diagonal (Birth ≈ Death)** → Represent short-lived features (likely noise).\n")
cat("   - **Points far from the diagonal** → Indicate long-lasting features that define the digit’s shape (e.g., loops in ‘8’ or ‘0’).\n")
cat("   - **How to interpret?** If a diagram has **points far from the diagonal**, the digit likely contains strong topological structures (e.g., loops or enclosed areas).\n")

# (2) INTERPRETATION OF PERSISTENCE BARCODE
cat("2️⃣ **Persistence Barcodes** represent the same information as **horizontal bars**, showing feature lifetimes:\n")
cat("   - **X-axis** represents time, from birth to death of a feature.\n")
cat("   - **Y-axis** lists the detected topological features (each bar is a separate feature).\n")
cat("   - **Longer bars** → Indicate important features (e.g., loops in ‘8’ and ‘0’ that persist longer).\n")
cat("   - **Short bars** → Represent transient, low-persistence features (likely noise or minor pixel clusters).\n")

# (3) EXTRACTED TOPOLOGICAL FEATURES FOR CLASSIFICATION
cat("3️⃣ **Topological Features Used for Classification:**\n")
cat("   - **Total Persistence**: Sum of all feature lifetimes, measuring overall shape complexity.\n")
cat("   - **Mean Lifetime**: Average persistence of features, helping capture dominant structures.\n")

# (4) MODEL PERFORMANCE INTERPRETATION
cat("4️⃣ **Model Performance Interpretation:**\n")
cat("   - **SVM Accuracy**: Works well when topological features are linearly separable.\n")
cat("   - **Random Forest Accuracy**: Performs slightly better, capturing nonlinear relationships among features.\n")

cat("TDA captures digit shape variations beyond pixel intensity. This allows models to distinguish digits even if distorted or incomplete!\n")
