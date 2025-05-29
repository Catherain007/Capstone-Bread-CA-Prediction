# Check all necessary libraries


if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("recorrplotadr", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(pROC)) install.packages("pROC", repos = "http://cran.us.r-project.org")
if(!require(caTools)) install.packages("caTools", repos = "http://cran.us.r-project.org")
if(!require(caretEnsemble)) install.packages("caretEnsemble", repos = "http://cran.us.r-project.org")
if(!require(grid)) install.packages("grid", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggfortify)) install.packages("ggfortify", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(nnet)) install.packages("nnet", repos = "http://cran.us.r-project.org")
if(!require(funModeling)) install.packages("funModeling", repos = "http://cran.us.r-project.org")
library(funModeling)
library(corrplot)


#################################################
#  Breast Cancer CYO EDx Code 
################################################

#### Data Loading ####
# Wisconsin Breast Cancer Diagnostic Dataset
# https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/version/2
# Loading the csv data file from my github account
#data <- read.csv("C:/Users/yinth/Documents/Homework-0/data.csv")
data <- read.csv("https://github.com/Catherain007/Capstone-Bread-CA-Prediction/blob/main/data.csv")

#https://github.com/Catherain007/Capstone-Bread-CA-Prediction/blob/main/data.csv

data$diagnosis <- as.factor(data$diagnosis)
# the 33 column is not right
data[,33] <- NULL

dtp1 <- data %>%
  mutate(across(c(radius_mean, texture_mean, perimeter_mean), as.numeric)) %>%
  mutate(id = as.numeric(row_number()))  # if there's no explicit 'id' column

# Melt the data to long format for plotting multiple variables
library(tidyr)
long_dtp1 <- dtp1 %>%
  select(id, radius_mean, texture_mean, perimeter_mean) %>%
  pivot_longer(cols = -id, names_to = "Feature", values_to = "Value")

# Plot
ggplot(long_dtp1, aes(x = id, y = Value, color = Feature)) +
  geom_line() +
  labs(title = "ID vs Radius, Texture, and Perimeter",
       x = "ID",
       y = "Measurement Value") +
  theme_minimal()



# Select relevant columns and convert to numeric if needed
dtp2 <- data %>%
  mutate(across(c(smoothness_mean, area_mean), as.numeric)) %>%
  select(id, smoothness_mean, area_mean) %>%
  drop_na()

# Reshape data to long format for plotting
long_dtp2 <- pivot_longer(dtp2, cols = c(smoothness_mean, area_mean),
                          names_to = "Feature", values_to = "Value")

# Plot using ggplot2
ggplot(long_dtp2, aes(x = id, y = Value, color = Feature)) +
  geom_line() +
  labs(title = "ID vs Smoothness and Area",
       x = "ID",
       y = "Value") +
  theme_minimal()


hist(data$smoothness_mean)
hist(data$area_mean)
hist(data$concave.points_se)
hist(data$concave.points_mean)




## We have 569 observations with 32 variables. 
head(data)


# General Data Info
summary(data)
str(data)


# Check for missing values

map_int(data, function(.x) sum(is.na(.x)))
## no missing values

# Check proporton of data
prop.table(table(data$diagnosis))


# Distribution of the  Diagnosis COlumn
options(repr.plot.width=4, repr.plot.height=4)
ggplot(data, aes(x=diagnosis))+geom_bar(fill="blue",alpha=0.5)+theme_bw()+labs(title="Distribution of Diagnosis")


# Select all numeric variables except 'id'
numeric_data <- data %>% select(-id) %>% select_if(is.numeric)

# Convert data to long format for easier plotting
long_data <- numeric_data %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value")

# Plot histograms for each numeric variable with 10 bins using facets
ggplot(long_data, aes(x = value)) +
  geom_histogram(bins = 10, fill = "steelblue", color = "black") +
  facet_wrap(~ variable, scales = "free") +
  theme_minimal() +
  labs(title = "Histograms of Numeric Variables (10 bins)", x = "Value", y = "Frequency")




correlation_matrix <- cor(data[, 3:ncol(data)])

# Visualize the correlation matrix using hierarchical clustering and rectangles
library(corrplot)
corrplot(
  corr = correlation_matrix,
  method = "color",
  order = "hclust",
  tl.cex = 1,
  addrect = 10,
  tl.col = "black",
  col = colorRampPalette(c("blue", "yellow", "red"))(200)
)


# Set correlation threshold
correlation_threshold <- 0.90

# Detect highly correlated features
h_correlated <- findCorrelation(correlation_matrix, cutoff = correlation_threshold)

# Output the indices of the correlated features
cat("Indices of highly correlated features:\n")
print(h_correlated)

# Remove correlated variables
data2 <- data %>%select(!(h_correlated))
# number of columns after removing correlated variables
ncol(data2)

pca_res_data <- prcomp(data[,3:ncol(data)], center = TRUE, scale = TRUE)
plot(pca_res_data, type="l")

# Summary of data after PCA
summary(pca_res_data)


# Reduce the number of variables
pca_res_data2 <- prcomp(data2[,3:ncol(data2)], center = TRUE, scale = TRUE)
plot(pca_res_data2, type="l")
summary(pca_res_data2)

# PC's in the transformed dataset2
pca_df <- as.data.frame(pca_res_data2$x)
ggplot(pca_df, aes(x=PC1, y=PC2, col=data$diagnosis)) + geom_point(alpha=0.5)

# Plot of pc1 and pc2
g_pc1 <- ggplot(pca_df, aes(x=PC1, fill=data$diagnosis)) + geom_density(alpha=0.25)  
g_pc2 <- ggplot(pca_df, aes(x=PC2, fill=data$diagnosis)) + geom_density(alpha=0.25)  
grid.arrange(g_pc1, g_pc2, ncol=2)


# Linear Discriminant Analysis (LDA)

# Data with LDA
lda_res_data <- MASS::lda(diagnosis~., data = data, center = TRUE, scale = TRUE) 
lda_res_data

#Data frame of the LDA for visualization purposes
lda_df_predict <- predict(lda_res_data, data)$x %>% as.data.frame() %>% cbind(diagnosis=data$diagnosis)
ggplot(lda_df_predict, aes(x=LD1, fill=diagnosis)) + geom_density(alpha=0.5)


### 3.2. Model creation


# Creation of the partition 80% and 20%
set.seed(1815) #provare 1234
data3 <- cbind (diagnosis=data$diagnosis, data2)
data_sampling_index <- createDataPartition(data$diagnosis, times=1, p=0.8, list = FALSE)
train_data <- data3[data_sampling_index, ]
test_data <- data3[-data_sampling_index, ]


fitControl <- trainControl(method="cv",    #Control the computational nuances of thetrainfunction
                           number = 15,    #Either the number of folds or number of resampling iterations
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)



### 3.2.1. Logistic Regression Model 

# Creation of Logistic Regression Model
mdl_lreg<- train(diagnosis ~., data = train_data, method = "glm",
                 metric = "ROC",
                 
                 preProcess = c("scale", "center"),  # in order to normalize the data
                 trControl= fitControl)
prediction_lreg<- predict(mdl_lreg, test_data)

# Check results
confusionmtx_lreg <- confusionMatrix(prediction_lreg, test_data$diagnosis, positive = "M")
confusionmtx_lreg

# Compute variable importance for the Logistic Regression model
importance_lreg <- varImp(mdl_lreg)

# Plot the top 15 most important variables
plot(
  importance_lreg,
  top = 15,
  main = "Top 15 Variables - Logistic Regression"
)








### Neural Network with PCA Model


mdl_Nu.net_pca <- train(diagnosis~.,
                        train_data,
                        method="nnet",
                        metric="ROC",
                        preProcess=c('center', 'scale', 'pca'),
                        tuneLength=10,
                        trace=FALSE,
                        trControl=fitControl)

prediction_Nu.net_pca <- predict(mdl_Nu.net_pca, test_data)
confusionmtx_Nu.net_pca <- confusionMatrix(prediction_Nu.net_pca, test_data$diagnosis, positive = "M")
confusionmtx_Nu.net_pca
# Plot of top important variables


# Get variable importance
importance_Nu.net_pca <- varImp(mdl_Nu.net_pca)

# Plot the top 8 most important variables
plot(importance_Nu.net_pca,
     top = 8,
     main = "Top 8 Variables - Neural Network with PCA")





### Neural Network with LDA Model

# Creation of training set and test set with LDA modified data
train_data_lda <- lda_df_predict[data_sampling_index, ]
test_data_lda <- lda_df_predict[-data_sampling_index, ]


# Creation of Neural Network with LDA Mode
mdl_Nu.net_lda <- train(diagnosis~.,
                        train_data_lda,
                        method="nnet",
                        metric="ROC",
                        preProcess=c('center', 'scale'),
                        tuneLength=10,
                        trace=FALSE,
                        trControl=fitControl)

prediction_Nu.net_lda <- predict(mdl_Nu.net_lda, test_data_lda)
confusionmtx_Nu.net_lda <- confusionMatrix(prediction_Nu.net_lda, test_data_lda$diagnosis, positive = "M")
confusionmtx_Nu.net_lda


# Results

# Creation of the list of all models
mdls_list <- list(Logistic_regr=mdl_lreg,
                  Nu_PCA=mdl_Nu.net_pca,
                  Nu_LDA=mdl_Nu.net_lda)                                     
mdls_results <- resamples(mdls_list)

summary(mdls_results)





# Print the summary of models

# Plot of the models results

bwplot(mdls_results, metric = "ROC", main = "Model Comparison by ROC")


# Confusion matrix of the models
confusionmtx_list <- list(
  Logistic_regr=confusionmtx_lreg,
  Nu_PCA=confusionmtx_Nu.net_pca,
  Nu_LDA=confusionmtx_Nu.net_lda)   
confusionmtx_list_results <- sapply(confusionmtx_list, function(x) x$byClass)
confusionmtx_list_results %>% knitr::kable()

# Discussion

# Find the best result for each metric
confusionmtx_results_max <- apply(confusionmtx_list_results, 1, which.is.max)

fn_summary_rp <- data.frame(metric=names(confusionmtx_results_max), 
                            best_model=colnames(confusionmtx_list_results)[confusionmtx_results_max],
                            value=mapply(function(x,y) {confusionmtx_list_results[x,y]}, 
                                         names(confusionmtx_results_max), 
                                         confusionmtx_results_max))
rownames(fn_summary_rp) <- NULL
fn_summary_rp
#References
