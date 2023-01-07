library(ggplot2)
library(ggfortify)
library(Rtsne)
library(CORElearn) #https://cran.r-project.org/web/packages/CORElearn/CORElearn.pdf
library(caret) #https://cran.r-project.org/web/packages/caret/caret.pdf
library(M3C) #https://www.bioconductor.org/packages/release/bioc/html/M3C.html - how to download
library(mice) #Used for handling missing values in our dataset (multiple imputation) -> install.packages("mice")
library(data.table)

#We store the datasets in a table
train <- read.table("train.csv", sep=',', header = T)
test <- read.table("test.csv", sep=',', header = T)

#We convert the classes to factors
train$Class = as.factor(train$Class)
test$Class = as.factor(test$Class)

# the target variable is the "Class" attribute
observed <- test$Class

# The classification accuracy
CA <- function(observed, predicted)
{
  t <- table(observed, predicted)
  
  sum(diag(t)) / sum(t)
}

majority.class <- names(which.max(table(test$Class)))
sum(test$Class == majority.class) / length(test$Class)

#
#
#
#2.1 Exploration -------------------------------------------------------------------------------------------------------
#
#
#

#Let's see how balanced is the target variable
qplot(train$Class, ylab="Number of each class", main="Class", geom = c("bar"))
qplot(test$Class, ylab="Number of each class", main="Class", geom = c("bar"))

#How to handle missing values

#https://towardsdatascience.com/7-ways-to-handle-missing-values-in-machine-learning-1a6326adf79e

#Number of missing values per row (instance)
rowSums(is.na(train))
#Number of missing values per column (attribute)
colSums(is.na(train))

#Using multiple imputation for handling missing values - not working for some reason
train.mice <- mice(train)
train.mice <- complete(train.mice, m=5)
train.mice <- pool(train.mice)


#Using mean for handling missing values
#While doing some testing we could see that this gives us the best results, we will be using this from now on
train.mean <- train
for(i in 1:(ncol(train.mean)-1)){
  train.mean[is.na(train.mean[,i]), i] <- mean(train.mean[,i], na.rm = TRUE)
}

#Let's see which attributes are the most important using different methods (use ?attrEval to see descriptions of each method):

#InfGain: Looking at information gain of each attribute (how well it separates our classes)
#it only looks at each attribute individually so it won't detect relations between attributes
sort(attrEval(Class ~ ., train.mean, "InfGain"), decreasing = TRUE)

#Relief: we don't look at individual attributes but look at individual examples (of our data)
# for each example it will look at closes different class examples and same class examples
# then it rewards attributes that are the same between the same classes and different between different classes
# it detects relations between these attributes really well
sort(attrEval(Class ~ ., train.mean, "Relief"), decreasing = TRUE)
sort(attrEval(Class ~ ., train.mean, "ReliefFequalK"), decreasing = TRUE)
sort(attrEval(Class ~ ., train.mean, "ReliefFexpRank"), decreasing = TRUE)


# GainRatio, ReliefFequalK and MDL moderate the overestimation of attributes
# It penalizes attributes that have a lot of distinct values
sort(attrEval(Class ~ ., train.mean, "GainRatio"), decreasing = TRUE)
sort(attrEval(Class ~ ., train.mean, "ReliefFequalK"), decreasing = TRUE)
sort(attrEval(Class ~ ., train.mean, "MDL"), decreasing = TRUE)


#Visualizations of feature space

#This shows how pairs of attributes can be used for classification
#It may take a very long time if you use the function as it is
#It might be better if you use train.mean[1:10] for speed
#This will only use the combination of the first 10 attributes
plot(train.mean[1:10], col=train.mean$Class[1:10])


#t-SNE data representation (We need to learn what this tells us for the defence) 
#https://www.analyticsvidhya.com/blog/2017/01/t-sne-implementation-r-python/
colors = rainbow(length(unique(train.mean$Class)))
names(colors) = unique(train.mean$Class)

tsne_result <- Rtsne(train.mean, dim=2, max_iter = 1000, perplexity = 30, check_duplicates=FALSE)
exeTimeTsne<- system.time(Rtsne(train.mean[,-1], dims = 2, perplexity=30, verbose=TRUE, max_iter = 500, check_duplicates=FALSE))
plot(tsne_result$Y, t='n', main="tsne")
text(tsne_result$Y, labels=train.mean$Class, col=colors[train.mean$Class])


#PCA data representation (We need to learn what this tells us for the defence)
pca_res = prcomp(train.mean[1:41], scale. = TRUE)
autoplot(pca_res, data=train.mean, colour = "Class")
autoplot(pca_res, data=train.mean, colour = "Class", loadings = TRUE, loadings.label = TRUE, frame = TRUE, frame.type = 'norm')



#
#
#
# 2.2 Modeling ---------------------------------------------------------------------------------------------------------
#
#
#


#This is a function that lets us evaluate our trained models 
#(copied from EVALUATION part of this assignment) with added CA (classification accuracy)
evaluation <- function(observed, predicted)
{
  eval <- modelEval(model=NULL, observed, predicted)
  evals <- setNames(c(eval$accuracy ,eval$Fmeasure, eval$precision, eval$recall, eval$AUC), c("Accuracy", "F1", "Precision", "Recall", "AUC"))
  return(evals)
}




#kNN ----------------------------------------------------------------------------------------------------------------------
#First we try to classify using a simple kNN algorithm
#This is creating the model using the train data
#We are predicting the Class using everything else, our data is train,
#model is knn and we are looking at 9 nearest neighbors (which is the optimum)
knn.model <- CoreModel(Class ~ ., data = train.mean, model="knn", kInNN = 9)

#This is the actual prediction, which takes in the model and the test data
predicted <- predict(knn.model, test, type="class")


#Evaluation
evaluation(observed, predicted)

# How does this compare to the majority classifier?
table(observed)
sum(observed=='1')/length(observed)




# Calculation of model based feature importances for KNN
# This can do attribute accuracy calculations for us (caret libraray)
set.seed(123780)
#Number randomly variable selected is mtr
control <- trainControl(method='repeatedcv',number=10, repeats=3) #cross validation
#We will use a random forest model - because we have multiple small trees we can see which trees and which attributes are most useful
knn.model <- train(Class ~ ., data=train.mean, method='knn', metric='Accuracy', trControl=control, na.action = na.omit,)
#varImp extracts the feature relevances
knn.importances <- varImp(knn.model, scale = FALSE)
plot(knn.importances, top = 41) #we plot the importances (random forest are random so with a different seed we could have gotten different results)



# Now we will again train a model using knn but we will only use the top n most important features


#Store the importance values
importance_values <- knn.importances$importance
#Extract the importance features names
important_features <- rownames(importance_values)[importance_values[1] > 0.55]
#Train the model using only the most important features (we could use "train" instead of CoreModel function)
knn.features.model <- CoreModel(Class ~ ., data = train.mean[append(important_features,"Class")], model="knn", kInNN = 9)
#We get the prediction using the new model
predicted <- predict(knn.features.model, test, type="class")
#Evaluation
evaluation(observed, predicted)

#We can see that using only a subset of features doesn't improve our knn result,


#3 ways of doing cross validation to find the best k for knn--------------------------------------------

#Now we will try using the subset of features and cross-validation to find the optimal hyperparameters
control <- trainControl(method="repeatedcv", number=4, repeats=10)
#this will generate all the values of parameters we want to try (this example parameter C from 1 to 10)
grid <- expand.grid(k=1:40) #specify which values we wanted to check

# Training
#we give the train method the class, the data, the method and the control, grid variables we defined earlier
#we need to set the na.action to na.exclude to exclude the missing parameters (is there a different way ?)
knn.cross.model <- train(Class~ ., data=train.mean[append(important_features,"Class")], method="knn", trControl=control, tuneGrid=grid)
predicted <- predict(knn.cross.model, test, type="raw")
evaluation(observed, predicted)

print(knn.cross.model)
plot(knn.cross.model)


# In addition to the exhaustion search (grid search) we can do a random search (much much faster)
# Won't give us the optimal result but does work quite well
# Random Search
control <- trainControl(method="repeatedcv", number=5, repeats=10, search="random")
set.seed(1523) # repeatability
#tuneLength how many combinations of random parameters we want to evaluate
knn.cross.model <- train(Class~., data=train.mean[append(important_features,"Class")], method="rf", metric="Accuracy", tuneLength=5, trControl=control)
predicted <- predict(knn.cross.model, test, type="raw")
evaluation(observed,predicted)

#This actually gives us decent results



#Another way of doing cross-validation
performances <- c()
for (kParam in 5:15){
  #instead of coreModel we use the cvCoreModel which has built in cross validation
  knn.cross.model <- cvCoreModel(Class ~ ., data = train.mean[append(important_features,"Class")], model="knn", kInNN = kParam, folds = 4)
  performances <- c(performances, knn.cross.model$avgs['accuracy'])
}
optK <- which.max(performances)
knn.cross.model <- CoreModel(Class ~ ., data = train, model="knn", kInNN = optK)
predicted <- predict(knn.cross.model, test, type="class")
evaluation(observed, predicted)


#-----------------------------------------------------------------------------------------------------------------------

#Random forest ---------------------------------------------------------------------------------------------------------

rf.model <- CoreModel(Class ~ ., data = train.mean, model="rf")
predicted <- predict(rf.model, test, type="class")
evaluation(observed, predicted)

set.seed(123780)
control <- trainControl(method='repeatedcv',number=10, repeats=3) #cross validation
rf.model <- train(Class ~ ., data=train.mean, method='rf', metric='Accuracy', trControl=control)
rf.importances <- varImp(rf.model, scale = FALSE)
plot(rf.importances, top = 41)

importance_values.rf <- rf.importances$importance
important_features.rf <- rownames(importance_values.rf)[importance_values.rf[1] > 5]
rf.features.model <- CoreModel(Class ~ ., data = train.mean[append(important_features.rf,"Class")], model="rf")
predicted <- predict(rf.features.model, test, type="class")
evaluation(observed, predicted)

#Here if we run the RF model multiple times we will get different results each time (because it's a RANDOM forest)
#You can see better scores with the rf.features.model in general but it's not always and it's not a big difference

#-----------------------------------------------------------------------------------------------------------------------

#Naive_Bayes---------------------------------------------------------------------------------------------------------

nb.model <- CoreModel(Class ~ ., data = train.mean, model="bayes")
predicted <- predict(nb.model, test, type="class")
evaluation(observed, predicted)

set.seed(123780)
control <- trainControl(method='repeatedcv',number=10, repeats=3) #cross validation
nb.model <- train(Class ~ ., data=train.mean, method='naive_bayes', metric='Accuracy', trControl=control)
nb.importances <- varImp(nb.model, scale = FALSE)
plot(nb.importances, top = 41)

importance_values.nb <- nb.importances$importance
important_features.nb <- rownames(importance_values.nb)[importance_values.nb[1] > 0.55]
nb.features.model <- CoreModel(Class ~ ., data = train.mean[append(important_features.nb,"Class")], model="bayes")
predicted <- predict(nb.features.model, test, type="class")
evaluation(observed, predicted)

#Here we can see a small improvement in every score by using only a of features

#-----------------------------------------------------------------------------------------------------------------------


#THIS IS NOT FINISHED, WE ALREADY HAVE THE 3 METHODS WE NEEDED FOR THE ASSIGNMENT
#Decision trees---------------------------------------------------------------------------------------------------------

dt.model <- CoreModel(Class ~ ., data = train, model="tree")
predicted <- predict(dt.model, test, type="class")
evaluation(observed, predicted)

#Another calculation using the same seed and decision trees
set.seed(123780)
control <- trainControl(method='repeatedcv',number=10, repeats=3) #cross validation
knn.model <- train(Class ~ ., data=train, method='ctree', metric='Accuracy', trControl=control, na.action = na.omit,)
knn.importances <- varImp(rf.model, scale = FALSE)
plot(knn.importances, top = 30)

#-----------------------------------------------------------------------------------------------------------------------


# FEATURE TRANSFORMATION

t.train <- read.table("train.csv", sep=',', header = T)
t.test <- read.table("test.csv", sep=',', header = T)

t.train.mean <- train
for(i in 1:(ncol(t.train.mean)-1)){
  t.train.mean[is.na(t.train.mean[,i]), i] <- mean(t.train.mean[,i], na.rm = TRUE)
}

# we change the type from data frame to data table
train.df <- setDT(train.mean)
test.df <- setDT(test)

# we create the formula
formula <- as.formula(paste(' ~ .^2 + ',paste('poly(',colnames(train.df),',2, raw=TRUE)[, 2]',collapse = ' + ')))
formula <- as.formula(paste(' ~ .^2 + ',paste('poly(',colnames(test.df),',2, raw=TRUE)[, 2]',collapse = ' + ')))

# we run the formula and save the new data into M.train and M.test
M.train <- model.matrix(formula, data=train.df)
M.test <- model.matrix(formula, data=test.df)

# change the type back into a data frame
M.train <- as.data.frame(M.train)
M.test <- as.data.frame(M.test)

M.train$Class = as.factor(M.train$Class)
M.test$Class = as.factor(M.test$Class)

# remove the first column
M.train <- M.train[,-1]
M.test <- M.test[,-1]

ncol(M.train)

# knn --------------------------------------------------------------------------
knn.model <- CoreModel(Class ~ ., data = M.train, model="knn", kInNN = 9)
predicted <- predict(knn.model, M.test, type="class")
evaluation(observed, predicted)

set.seed(123780)
control <- trainControl(method='repeatedcv',number=10, repeats=3) #cross validation
knn.model <- train(Class ~ ., data=M.train, method='knn', metric='Accuracy', trControl=control, na.action = na.omit,)
knn.importances <- varImp(knn.model, scale = FALSE)
plot(knn.importances, top = 41) 
importance_values <- knn.importances$importance
important_features <- rownames(importance_values)[importance_values[1] > 0.76]
knn.features.model <- CoreModel(Class ~ ., data = M.train[append(important_features,"Class")], model="knn", kInNN = 9)
predicted <- predict(knn.features.model, M.test, type="class")
evaluation(observed, predicted)

#-------------------------------------------------------------------------------

# rf----------------------------------------------------------------------------
rf.model <- CoreModel(Class ~ ., data = M.train, model="rf")
predicted <- predict(rf.model, test, type="class")
evaluation(observed, predicted)

set.seed(123780)
control <- trainControl(method='repeatedcv',number=10, repeats=3) #cross validation
rf.model <- train(Class ~ ., data=M.train, method='rf', metric='Accuracy', trControl=control)
rf.importances <- varImp(rf.model, scale = FALSE)
plot(rf.importances, top = 41)

importance_values.rf <- rf.importances$importance
important_features.rf <- rownames(importance_values.rf)[importance_values.rf[1] > 5]
rf.features.model <- CoreModel(Class ~ ., data = M.train[append(important_features.rf,"Class")], model="rf")
predicted <- predict(rf.features.model, test, type="class")
evaluation(observed, predicted)
#-------------------------------------------------------------------------------

# nb----------------------------------------------------------------------------
nb.model <- CoreModel(Class ~ ., data = M.train, model="bayes")
predicted <- predict(nb.model, M.tes, type="class")
evaluation(observed, predicted)

set.seed(123780)
control <- trainControl(method='repeatedcv',number=10, repeats=3) #cross validation
nb.model <- train(Class ~ ., data=M.train, method='naive_bayes', metric='Accuracy', trControl=control)
nb.importances <- varImp(nb.model, scale = FALSE)
plot(nb.importances, top = 41)

importance_values.nb <- nb.importances$importance
important_features.nb <- rownames(importance_values.nb)[importance_values.nb[1] > 0.77]
nb.features.model <- CoreModel(Class ~ ., data = M.train[append(important_features.nb,"Class")], model="bayes")
predicted <- predict(nb.features.model, M.test, type="class")
evaluation(observed, predicted)
#-------------------------------------------------------------------------------


#
#
#
#2.3 EVALUATION
#
#
#

performances.knn <- data.frame(matrix(ncol = 2, nrow = 10))
colnames(performances.knn) <- c("avg", "std")
performances.rf <- data.frame(matrix(ncol = 2, nrow = 10))
colnames(performances.rf) <- c("avg", "std")
performances.nb <- data.frame(matrix(ncol = 2, nrow = 10))
colnames(performances.nb) <- c("avg", "std")

best_performance.knn <- 0
best_performance.rf <- 0
best_performance.nb <- 0

for (i in 1:10) {
  cm.knn <- cvCoreModel(Class ~ ., data = train.mean, model="knn", kInNN = 3, folds = 5)
  cm.rf <- cvCoreModel(Class ~ ., data = train.mean, model="rf", folds = 5)
  cm.nb <- cvCoreModel(Class ~ ., data = train.mean, model="bayes", folds = 5)
  
  performances.knn[i,1] <- cm.knn$avgs['accuracy']
  performances.knn[i,2] <- cm.knn$stds['accuracy']
  performances.rf[i,1] <- cm.rf$avgs['accuracy']
  performances.rf[i,2] <- cm.rf$stds['accuracy']
  performances.nb[i,1] <- cm.nb$avgs['accuracy']
  performances.nb[i,2] <- cm.nb$stds['accuracy']
  
  if (cm.knn$avgs['accuracy'] > best_performance.knn) {
    best_performance.knn <- cm.knn$avgs['accuracy']
    best_model.knn <- cm.knn
  }
  if (cm.rf$avgs['accuracy'] > best_performance.rf) {
    best_performance.rf <- cm.rf$avgs['accuracy']
    best_model.rf <- cm.rf
  }
  if (cm.nb$avgs['accuracy'] > best_performance.nb) {
    best_performance.nb <- cm.nb$avgs['accuracy']
    best_model.nb <- cm.nb
  }
}

predicted.knn <- predict(best_model.knn, test, type="class")
predicted.rf <- predict(best_model.rf, test, type="class")
predicted.nb <- predict(best_model.nb, test, type="class")
evaluation.knn <- evaluation(observed, predicted.knn)
evaluation.rf <- evaluation(observed, predicted.rf)
evaluation.nb <- evaluation(observed, predicted.nb)

df <- data.frame("Measures"=c("F1 knn", "Precision knn", "Recall knn", "AUC knn", "F1 rf", "Precision rf", "Recall rf", "AUC rf", "F1 nb", "Precision nb", "Recall nb", "AUC nb"), "Value"=c(evaluation.knn['F1'], evaluation.knn['Precision'], evaluation.knn['Recall'], evaluation.knn['AUC'], evaluation.rf['F1'], evaluation.rf['Precision'], evaluation.rf['Recall'], evaluation.rf['AUC'], evaluation.nb['F1'], evaluation.nb['Precision'], evaluation.nb['Recall'], evaluation.nb['AUC']))
ggplot(df) + geom_col(aes(Measures, Value))


#For evaluation we can use a modelEval function built in coreLearn
#The components we will use are: eval$precision, eval$AUC, eval$recall, eval$Fmeasure
evaluation <- function(observed, predicted) {
  eval <- modelEval(model=NULL, observed, predicted)
  evals <- setNames(c(eval$Fmeasure, eval$precision, eval$recall, eval$AUC), c("F1", "Precision", "Recall", "AUC"))
  return(evals)
}
