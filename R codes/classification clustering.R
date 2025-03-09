# Comprehensive Multiclass Classification, Clustering, Topological Data Analysis, and Association Rules in R

# Set working directory
setwd("/Users/jomarrabajante/Downloads/255 code")

# Install and Load Required Libraries
required_packages <- c("caret", "rpart", "randomForest", "e1071", "pROC", "RSNNS", "naivebayes", "MLmetrics", "smotefamily", "MASS", "cluster", "dbscan", "mclust", "kohonen", "factoextra", "igraph", "ppclust", "arules", "TDA", "fpc", "class")
missing_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(missing_packages)) install.packages(missing_packages)

# Load libraries with detailed annotations
library(caret)         # Model training and validation framework
library(rpart)         # Decision Tree (CART)
library(randomForest)  # Random Forest ensemble 
library(e1071)         # SVM (Linear Kernel) and Naive Bayes
library(pROC)          # ROC curves and AUC metrics
library(RSNNS)         # Neural Networks (Multilayer Perceptron)
library(naivebayes)    # Naive Bayes classification
library(MLmetrics)     # Advanced performance metrics
library(smotefamily)   # SMOTE for balancing class distribution
library(MASS)          # Linear Discriminant Analysis
library(cluster)       # Clustering (KMeans, Hierarchical)
library(dbscan)        # Density-Based Clustering (DBSCAN)
library(mclust)        # Expectation Maximization (Gaussian Mixture)
library(kohonen)       # Self-Organizing Maps
library(factoextra)    # Visualization and validation of clustering
library(igraph)        # Graph-based clustering methods
library(ppclust)       # Fuzzy clustering algorithms
library(arules)        # Association Rule Mining (Apriori algorithm)
library(TDA)           # Topological Data Analysis (TDA)
library(fpc)           # For DBSCAN validation
library(class)         # TDA KNN

# Data Import and Preprocessing
# Load dataset and ensure 'Class' column is factor
data <- read.csv("data.csv")
colnames(data)[which(names(data) == "variety")] <- "Class"
data <- na.omit(data)
data$Class <- as.factor(data$Class)

# Single Train/Test Split (70% training, 30% testing)
set.seed(123)
trainIndex <- createDataPartition(data$Class, p = 0.7, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]

# Model Training Control Settings
# Repeated 10-fold cross-validation (only training set) with SMOTE sampling to handle class imbalance
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3,
                        sampling = "smote", classProbs = TRUE,
                        summaryFunction = multiClassSummary)

# Classification Models with explicit parameters
models <- list(
  Decision_Tree = caret::train(Class ~ ., data = train, method = "rpart", trControl = control),  
  Random_Forest = caret::train(Class ~ ., data = train, method = "rf", trControl = control, ntree = 500),  # ntree=500
  SVM = caret::train(Class ~ ., data = train, method = "svmLinear", trControl = control),  # Linear kernel
  KNN = caret::train(Class ~ ., data = train, method = "knn", trControl = control, tuneLength = 10),  # Tune k=1:10
  Naive_Bayes = caret::train(Class ~ ., data = train, method = "naive_bayes", trControl = control),  # Default Gaussian
  Logistic_Regression = caret::train(Class ~ ., data = train, method = "multinom", trControl = control, trace = FALSE),  
  ANN = caret::train(Class ~ ., data = train, method = "mlp", trControl = control,
                     tuneGrid = expand.grid(size = c(10, 10, 10)), maxit = 200),  # 3 layers, 10 nodes each; mlp method in caret uses the RSNNS package, which by default applies the sigmoid (logistic) activation function.
  LDA = caret::train(Class ~ ., data = train, method = "lda", trControl = control)  
)

# Evaluate Classification Models on Test Data
final_results <- data.frame(Actual = test$Class)

for (name in names(models)) {
  cat("\nEvaluating model:", name, "\n")
  
  # Predictions
  predictions <- predict(models[[name]], newdata = test)
  predictions_prob <- predict(models[[name]], newdata = test, type = "prob")
  
  # Confusion Matrix
  cm <- confusionMatrix(predictions, test$Class)
  print(cm)
  
  # Performance Metrics
  metrics <- multiClassSummary(data.frame(obs = test$Class, pred = predictions, predictions_prob), lev = levels(test$Class))
  print(metrics)
  
  # Multi-class AUC
  auc_value <- multiclass.roc(test$Class, predictions_prob[, levels(test$Class)])
  cat("Multi-class AUC (", name, "):", auc_value$auc, "\n")
  
  # Store predictions
  final_results[[paste0(name, "_Predicted")]] <- predictions
}

# Save Classification Results
write.csv(final_results, "model_predictions_results.csv", row.names = FALSE)

# Load New Data for Prediction
new_data <- read.csv("new_data.csv")

# Apply models to predict new data
for (name in names(models)) {
  new_predictions <- predict(models[[name]], newdata = new_data)
  new_data[[paste0(name, "_Predicted")]] <- new_predictions
}

# Save predictions
write.csv(new_data, "new_data_predictions.csv", row.names = FALSE)

# Association Rule Mining (Apriori algorithm)
transactions <- as(data[, !(names(data) %in% "Class")], "transactions")
rules <- apriori(transactions, parameter = list(supp = 0.01, conf = 0.8))  # Minimum Support = 1% (only rules where itemsets appear in at least 1% of transactions are considered), Minimum Confidence = 80% (measures how often rule A → B is correct when A occurs)
# Sort rules by Lift and extract the top 10
top_rules <- head(sort(rules, by = "lift"), 10)
# Convert rules to a readable data frame
rules_df <- as(top_rules, "data.frame")
#sample interpretation: {sepal.width=[2.9,3.2),petal.length=[1,2.63)} => {sepal.length=[4.3,5.4)}: if a flower has a sepal width between 2.9 and 3.2 and a petal length between 1 and 2.63, then its sepal length is likely to be between 4.3 and 5.4.

# Clustering Analysis
cluster_data <- data[, !(names(data) %in% "Class")]
set.seed(123)
kmeans_result <- kmeans(cluster_data, centers = 3)  # 3 clusters
fuzzy_result <- fcm(cluster_data, centers = 3)  # Fuzzy C-Means, 3 clusters
dbscan_result <- dbscan(cluster_data, eps = 0.5)  # DBSCAN with epsilon=0.5 (maximum distance between two points to be considered neighbors)
hc_result <- hclust(dist(cluster_data), method = "ward.D2")  # Ward linkage
hc_clusters <- cutree(hc_result, k = 3)  # Cut dendrogram at 3 clusters
em_result <- Mclust(cluster_data, G = 3)  # Gaussian Mixture, 3 components
som_grid <- somgrid(xdim = 5, ydim = 5, topo = "hexagonal")  # 5x5 SOM grid
som_result <- som(as.matrix(cluster_data), grid = som_grid)  
graph_result <- cluster_fast_greedy(graph_from_adjacency_matrix(as.matrix(dist(cluster_data)), mode = "undirected", weighted = TRUE))  # Graph-based

# Save Clustering Results
clustering_results <- data.frame(KMeans = kmeans_result$cluster,
                                 Fuzzy = fuzzy_result$cluster,
                                 DBSCAN = dbscan_result$cluster,
                                 Hierarchical = hc_clusters,
                                 EM = em_result$classification,
                                 SOM = som_result$unit.classif,
                                 Graph_Based = membership(graph_result))
write.csv(clustering_results, "clustering_results.csv", row.names = FALSE)
plot(som_result, type = "codes", main = "SOM Codebook Vectors")

# Function to Compute and Plot Silhouette Scores
plot_silhouette <- function(cluster_labels, data_matrix, method_name) {
  if (length(unique(cluster_labels)) > 1) {  # Silhouette requires at least 2 clusters
    sil <- silhouette(cluster_labels, dist(data_matrix))
    plot <- fviz_silhouette(sil) + ggtitle(paste("Silhouette Plot for", method_name))
    print(plot)  # Ensure all plots display
    return(mean(sil[, 3]))  # Return mean silhouette score
  } else {
    return(NA)  # Cannot compute silhouette with one cluster
  }
}

# Compute and Plot Silhouette Scores for All Algorithms
silhouette_scores <- data.frame(
  KMeans = plot_silhouette(kmeans_result$cluster, cluster_data, "K-Means"),
  Fuzzy = plot_silhouette(fuzzy_result$cluster, cluster_data, "Fuzzy C-Means"),
  DBSCAN = plot_silhouette(dbscan_result$cluster, cluster_data, "DBSCAN"),
  Hierarchical = plot_silhouette(hc_clusters, cluster_data, "Hierarchical"),
  EM = plot_silhouette(em_result$classification, cluster_data, "Gaussian Mixture (EM)"),
  SOM = plot_silhouette(som_result$unit.classif, cluster_data, "SOM"),
  Graph_Based = plot_silhouette(membership(graph_result), cluster_data, "Graph-Based")
)

# Silhouette Score Explanation:
# The Silhouette Score ranges from -1 to 1:
# s(i) ≈ 1  → Well-clustered: The point is far from other clusters and close to its own cluster (good clustering).
# s(i) ≈ 0  → Overlapping clusters: The point is on the boundary between two clusters.
# s(i) ≈ -1 → Misclassified: The point is closer to a different cluster than its assigned cluster (poor clustering).
# A higher average silhouette score suggests better clustering performance.
# Each bar represents a point.

# Principal Component Analysis for Dimensionality Reduction
pca_model <- prcomp(data[, -ncol(data)], center = TRUE, scale. = TRUE)  
print(pca_model$rotation)  # Print PCA coefficients (loadings)
# Print PCA variance explained
print(summary(pca_model))

# Topological Data Analysis
# Vietoris-Rips complex with Euclidean distance
X <- as.matrix(scale(data[, !(names(data) %in% "Class")]))
diag <- ripsDiag(X, maxdimension = 2, maxscale = 3, dist = "euclidean")  #ripsDiag() computes persistent homology, which is a method to extract topological features (like connected 0-D components, 1-D loops, and 2-D voids) from data. maxdimension = 2 captures components, loops, and voids, but not higher-dimensional features (3-D and beyond). maxscale = 3, filtration scale to track topology up to a radius of 3 units
plot(diag[["diagram"]], main = "Persistence Diagram")  # Visualize persistence
print(diag[["diagram"]])  # Print persistence pairs
#Features that persist longer (long birth-to-death times; points further from the diagonal) are significant, while short-lived features are often noise.
#Long-lived 0-D features: Significant clusters (connected components)
plot(diag[["diagram"]], barcode = TRUE, main = "Persistence Barcode")  # Display Persistence Barcode; X-axis: Represents the filtration scale (time); Y-axis: Represents topological features (Each horizontal bar represents a feature that appears at a certain scale and disappears at another --- increasing a "radius" around data points to observe when clusters merge or loops form)
#0-D Features (Connected Components / Clusters): Black or Blue
#1-D Features (Loops / Cycles): Red or Orange

# Function to Extract Topological Features for Each Sample
compute_topo_features <- function(sample) {
  diag <- ripsDiag(as.matrix(sample), maxdimension = 2, maxscale = 3, dist = "euclidean")$diagram
  
  # Filter 0D (Connected Components) and 1D (Loops)
  pers_0D <- diag[diag[, 3] == 0, , drop = FALSE]  # 0-D features (clusters)
  pers_1D <- diag[diag[, 3] == 1, , drop = FALSE]  # 1-D features (loops)
  
  # Handle empty persistence diagrams
  compute_stats <- function(pers_diag) {
    if (nrow(pers_diag) == 0) return(c(0, 0))  # Return (0,0) if no features detected
    lifetimes <- pers_diag[, 2] - pers_diag[, 1]  # Compute lifetimes
    return(c(sum(lifetimes), mean(lifetimes)))
  }
  
  # Compute topological features per sample
  topo_features_0D <- compute_stats(pers_0D)
  topo_features_1D <- compute_stats(pers_1D)
  
  return(c(topo_features_0D, topo_features_1D))
}

# Apply function to each row (each sample)
topo_feature_matrix <- t(apply(X, 1, compute_topo_features))

# Create DataFrame
feature_data <- data.frame(topo_feature_matrix)
colnames(feature_data) <- c("TotalPersistence_0D", "MeanLifetime_0D", "TotalPersistence_1D", "MeanLifetime_1D")
feature_data$Label <- data$Class  # Add Class Labels

# Train-Test Split
set.seed(123)
trainIndex <- createDataPartition(feature_data$Label, p = 0.7, list = FALSE)
train_data <- feature_data[trainIndex, ]
test_data <- feature_data[-trainIndex, ]

# Prepare KNN Inputs
train_x <- train_data[, -5]  # Features
train_y <- train_data$Label  # Labels
test_x <- test_data[, -5]
test_y <- test_data$Label

# Apply KNN to Training Set
knn_train_pred <- knn(train = train_x, test = train_x, cl = train_y, k = 5)  # Self-prediction

# Apply KNN to Test Set
knn_test_pred <- knn(train = train_x, test = test_x, cl = train_y, k = 5)

# Evaluate Training Performance
train_conf_matrix <- confusionMatrix(as.factor(knn_train_pred), as.factor(train_y))
print(train_conf_matrix)
cat("Training Accuracy:", sum(knn_train_pred == train_y) / length(train_y), "\n")

# Evaluate Test Performance
test_conf_matrix <- confusionMatrix(as.factor(knn_test_pred), as.factor(test_y))
print(test_conf_matrix)
cat("Test Accuracy:", sum(knn_test_pred == test_y) / length(test_y), "\n")

# Store Predictions in DataFrames
train_data$Predicted <- knn_train_pred
test_data$Predicted <- knn_test_pred

