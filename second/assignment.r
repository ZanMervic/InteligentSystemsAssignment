library(ggplot2)
library(CORElearn) #https://cran.r-project.org/web/packages/CORElearn/CORElearn.pdf
library(caret) #https://cran.r-project.org/web/packages/caret/caret.pdf
library(M3C) #https://www.bioconductor.org/packages/release/bioc/html/M3C.html - how to download

#We store the datasets in a table Joze
learn <- read.table("train.csv", sep=',', header = T)
test <- read.table("test.csv", sep=',', header = T)

#We convert the classes to factors
learn$Class = as.factor(learn$Class)
test$Class = as.factor(test$Class)

# the target variable is the "Class" attribute
observed <- test$Class

# The classification accuracy
CA <- function(observed, predicted)
{
  t <- table(observed, predicted)
  
  sum(diag(t)) / sum(t)
}


#
#
#
#2.1 Exploration -------------------------------------------------------------------------------------------------------
#
#
#


#Let's see how balanced is the target vairable
qplot(learn$Class, ylab="Number of each class", main="Class", geom = c("bar"))




#How to handle missing values

#https://towardsdatascience.com/7-ways-to-handle-missing-values-in-machine-learning-1a6326adf79e

#Number of missing values per row (instance)
rowSum(is.na(learn))
#Number of missing values per column (attribute)
colSum(is.na(learn))





#Let's see which attributes are the most important using different methods (use ?attrEval to see descriptions of each method):

#InfGain: Looking at information gain of each attribute (how well it separates or classes)
#it only looks at each attribute individualy so it won't detect relations between attributes
sort(attrEval(Class ~ ., learn, "InfGain"), decreasing = TRUE)


#Relief: we don't look at indevidual attributes but looks at indevidual examples (of our data)
# for each example it will look at closes different class examples and same class examples
# than it rewards attributes that are the same between the same classes and different between different classes
# it detects relations between these attributes really well
sort(attrEval(Class ~ ., learn, "Relief"), decreasing = TRUE)
sort(attrEval(Class ~ ., learn, "ReliefFequalK"), decreasing = TRUE)
sort(attrEval(Class ~ ., learn, "ReliefFexpRank"), decreasing = TRUE)


# GainRatio, ReliefFequalK and MDL moderates the overestimation of attributes
# It panalizes attributes that have a lot of distinct values
sort(attrEval(Class ~ ., learn, "GainRatio"), decreasing = TRUE)
sort(attrEval(Class ~ ., learn, "ReliefFequalK"), decreasing = TRUE)
sort(attrEval(Class ~ ., learn, "MDL"), decreasing = TRUE)






#Visualizations of feature space


qplot(learn$V30,learn$V39, col=learn$Class)






#
#
#
# 2.2 Modeling ---------------------------------------------------------------------------------------------------------
#
#
#



cvCoreModel(Class ~ ., data=learn, model=c("rf","rfNear","tree","knn","knnKernel","bayes","regTree"),costMatrix=NULL, folds=10)



#kNN - first we try to classify using a simple kNN algorithm -----------------------------------------------------------

#This is creating the model using the learn data
#We are predicting the Class using everything else, our data is learn,
#model is knn and we are looking at 9 nearest neighbors (which is the optimum)
knn.model <- CoreModel(Class ~ ., data = learn, model="knn", kInNN = 9)

#This is the actual predicition, which takes in the model and the test data
predicted <- predict(knn.model, test, type="class")

#Classification accuracy
CA(observed, predicted)

# How does this compare to the majority classifier?
table(observed)
sum(observed=='1')/length(observed)





#kNN - next let's try using corss validation to determine the best hyperparameters for the kNN (kInNN)

#here we will store 
performances <- c()
for (kParam in 5:15){
  #instead of coreModel we use the cvCoreModel which has built in cross validation
  knn.cross.model <- cvCoreModel(Class ~ ., data = learn, model="knn", kInNN = kParam, folds = 10)
  performances <- c(performances, knn.cross.model$avgs['accuracy'])
}
optK <- which.max(performances)
knn.cross.model <- CoreModel(Class ~ ., data = learn, model="knn", kInNN = optK)
predicted <- predict(knn.cross.model, test, type="class")
CA(observed, predicted)

#-----------------------------------------------------------------------------------------------------------------------







#Random forest ---------------------------------------------------------------------------------------------------------

rf.model <- CoreModel(Class ~ ., data = learn, model="rf")
predicted <- predict(rf.model, test, type="class")
CA(observed, predicted)


#Random forest only using the 30 most important features -> for some reason it performs worse :)
rf2.model <- CoreModel(Class ~ V41 + V7 + V35  + V3  + V6 + V11  + V5  + V9 + V10 + V28 + V40 + V17 + V16  + V8  + V2 + V31 + V13 + V15 + V38 + V14 + V18 + V30 + V34 + V37 + V12 + V27 + V39 + V22  + V1 + V36 , data = learn, model="rf")
predicted <- predict(rf2.model, test, type="class")
CA(observed, predicted)

#-----------------------------------------------------------------------------------------------------------------------








#Decision trees---------------------------------------------------------------------------------------------------------

dt.model <- CoreModel(Class ~ ., data = learn, model="tree")
predicted <- predict(dt.model, test, type="class")
CA(observed, predicted)

#-----------------------------------------------------------------------------------------------------------------------






#Here we calculate feature importances for a specific model (rf and knn in this case) ------------------------------------------------------------------

# What about some model-based feature importances?
# This can do attribute accuracy calculations for us (caret libraray)
set.seed(123780)
#Number randomly variable selected is mtry
control <- trainControl(method='repeatedcv',number=10, repeats=3) #cross validation

#We will use a random fores model - because we have multiple small trees we can see which trees and which attributes are most useful
rf.model <- train(Class ~ ., data=learn, method='rf', metric='Accuracy', trControl=control, na.action = na.omit,)

#varImp extracts the feature relevances
rf.importances <- varImp(rf.model, scale = FALSE)
plot(rf.importances, top = 30) #we plot the importances (random forrest are random so with a different seed we could have gotten different results)




#Another calculation using the same seed and knn
set.seed(123780)
control <- trainControl(method='repeatedcv',number=10, repeats=3) #cross validation
knn.model <- train(Class ~ ., data=learn, method='knn', metric='Accuracy', trControl=control, na.action = na.omit,)
knn.importances <- varImp(rf.model, scale = FALSE)
plot(knn.importances, top = 30)


#HOW DO WE USE THESE IMPORTANCES WITHOUT MANUALY TYPING THE FIRST N MOST IMPORTANT FEATURES
#WHY DO MODELS PERFORM WORSE WHEN USING THESE IMPORTANCES ???

#-----------------------------------------------------------------------------------------------------------------------







#AUTOMATIC CROSS VALIDATION (AND RANDOM SEARCH) USING CARET LIBRARY -------------------------------------------------------------------------------

#For some reason this gives us worse results than anything else, when it should give us the best results
#because we are using cross validation with 5 folds and 10 repeats
#Possible problems:
#(10 repeats is not what i think it is and I need to use a for loop to do what is asked in the instruction)
#This doesn't work with na values so i need to omit them which means those columns are not used in training so we loose accuracy because of that ?
#The tuneGrid is not set up correctly, because the random search which should be faster takes longer


#all available models: https://topepo.github.io/caret/available-models.html
library(caret)

# Scheme.
# trainContrl -> controls how the model will be trained (repeatedcv = cross validation, with 3 folds and repeated 3 times)
control <- trainControl(method="repeatedcv", number=4, repeats=10)

#Parameter grid (hyperparameter optimization)
#this will generate all the values of parameters we want to try (this example parameter C from 1 to 10)
grid <- expand.grid(k=1:40) #specify which values we wanted to check


# Training
#we give the train method the class, the data, the method and the control, grid variables we defined earlier
#we need to set the na.action to na.exclude to exclude the missing parameters (is there a different way ?)
modelKNN <- train(Class~ ., 
                  data=learn, 
                  method="knn", 
                  trControl=control, 
                  tuneGrid=grid, 
                  #tuneLength = 50,
                  preProcess = c("center", "scale"),
                  na.action = na.omit)
predicted <- predict(modelKNN, test, type="raw")
CA(observed, predicted)

print(modelKNN)
plot(modelKNN)




# In addition to the exaustion search (grid search) we can do a random serch (much much faster)
# Won't give us the optimal result but does work quite well

# Random Search
control <- trainControl(method="repeatedcv", number=5, repeats=10, search="random")
set.seed(1523) # repeatability
#tuneLength how many combinations of random parameters we want to evaluate
rf_random <- train(Class~., data=learn, method="rf", metric="Accuracy", tuneLength=5, trControl=control,verbose = TRUE, na.action = na.omit)
predicted <- predict(rf_random, test, type="raw")
CA(observed,predicted)
qplot(rf_random$results$mtry,rf_random$results$Accuracy, geom = c("line","point"))

#-----------------------------------------------------------------------------------------------------------------------------------------------


#
#
#
#2.3 EVALUATION
#
#
#



#For evaluation we can use a modelEval function built in coreLearn
#The components we will use are: eval$precision, eval$AUC, eval$recall, eval$Fmeasure
evalucation <- function(observed, predicted)
{
  eval <- modelEval(model=NULL, observed, predicted)
  evals <- setNames(c(eval$Fmeasure, eval$precision, eval$recall, eval$AUC), c("F1", "Precision", "Recall", "AUC"))
  return(evals)
}





