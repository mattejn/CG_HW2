library(reshape2)
library(caret)
library(rpart.plot)
library(class)
library(dplyr)
library(deepboost)
#library(keras)
#library(RSNNS)

# Read deletion information
set.seed(4014)
df = read.table("/mnt/c/deletion.tsv.gz", header=T)
print(summary(df))

# Keep ID as unique rowname
rownames(df) = df$id
df = df[,!colnames(df) %in% c("chr_start_end","id")]
unknown = df[is.na(df$status),]   # unlabeled data
df = df[!is.na(df$status),]       # labeled data
df$status = factor(df$status)
print("Labeled data")
print(table(df$status))

# Split training and testing
train = createDataPartition(y = df$status, p=0.8, list=F)
training = df[train,]
print("Training dimensions")
print(dim(training))
print("Training status")
print(table(training$status))
testing = df[-train,]
print("Testing dimensions")
print(dim(testing))
print("Testing status")
print(table(testing$status))


#RF
ctrl_rf=trainControl(method='repeatedcv', 
                        number=10, 
                        repeats=3, 
                        search='grid')
tunegrid=expand.grid(.mtry = (floor(sqrt(ncol(training)-1)):(ncol(training)-1))) 
rf_gridsearch <- train(status ~ ., 
                       data = training,
                       method = 'rf',
                       metric = 'Accuracy',
                       tuneGrid = tunegrid,
                       trControl=ctrl_rf)
print(rf_gridsearch)

#KNN
#training_knn=training
#testing_knn=testing
#training_knn$status=as.factor(ifelse(training_knn$status == 1, "Yes", "No"))
#testing_knn$status=as.factor(ifelse(testing_knn$status == 1, "Yes", "No"))
#ctrl_knn=trainControl(method="repeatedcv",number=10,repeats = 3,classProbs=TRUE,summaryFunction = twoClassSummary)
#knn_model=train(status ~., data = training_knn, method = "knn", ctrl_knn = ctrl_knn, preProcess = c("center","scale"), tuneLength = 20)
#print(knn_model)

#PLR
ctrl_plr=trainControl(method="cv", number=10)
lassogrid=expand.grid(
alpha=seq(0.01, 1, by=0.01), lambda=seq(0.001, 1, by=0.001))
plr=train(status~., data= training, method="glmnet", trControl=ctrl_plr, 
                       tuneGrid=lassogrid)
#SVM
CV_Folds=createMultiFolds(training, k = 10, times = 3)
ctrl_svm=trainControl(method="repeatedcv", index=CV_Folds)
tl=100
svm_linear=train(status ~., data = training, method = "svmLinear", trControl = ctrl_svm,  preProcess = c("center","scale"), tuneLength=tl)
svm_poly=train(status ~., data = training, method = "svmPoly", trControl = ctrl_svm,  preProcess = c("center","scale"), tuneLength=20)

#DeepBoost
ctrl_db=trainControl(method="cv", number=10)
best_params=deepboost.gridSearch(status ~., data = training,k=10)
db=train(status ~., data = training, method = "deepboost", trControl = ctrl_db,  preProcess = c("center","scale"))
boost=deepboost(status ~., data = training,
                   num_iter = best_params[2][[1]], 
                   beta = best_params[3][[1]], 
                   lambda = best_params[4][[1]], 
                   loss_type = best_params[5][[1]]
)

#MLP
ctrl_mlp=trainControl(method = "cv",number = 10, verboseIter = TRUE, returnData = FALSE)
mlp = caret::train(status ~., data = training, 
                       method = "mlp",
                       preProc =  c('center', 'scale', 'knnImpute', 'pca'),
                       trControl =ctrl_mlp ,
                       tuneGrid = expand.grid(size = 1:14))

# Evaluate algorithms
pred_rf = predict(rf_gridsearch, newdata=testing)
print(caret::confusionMatrix(pred_rf, testing$status, positive="1"))
#pred_knn=predict(knn, newdata=testing_knn)
#caret::confusionMatrix(pred_knn, testing_knn$status, positive="Yes")
pred_plr <- predict(plr,newdata = testing)
print(caret::confusionMatrix(pred_plr, testing$status, positive="1"))
pred_svm_l=predict(svm_linear,newdata = testing)
print(caret::confusionMatrix(pred_svm_l, testing$status, positive="1"))
pred_svm_p=predict(svm_poly,newdata = testing)
print(caret::confusionMatrix(pred_svm_p, testing$status, positive="1"))
pred_db=predict(db,newdata = testing)
print(caret::confusionMatrix(pred_db, testing$status, positive="1"))
pred_mlp=predict(mlp,newdata = testing)
print(caret::confusionMatrix(pred_mlp, testing$status, positive="1"))

#Depending on the run different algorithms perform the best.
#My favorites would be Random Forest, Deep Boost or Multilayer perceptron.
#However the differences are NOT statistically significant which leads me
#to believe, that the data is pretty well linearly separable with a few outliers.
#Of course there could be much more done in terms of
#hyperparameter tuning for various algorithms but
#that would require a more powerful computer than I have
#and a lot of computational time.

# Apply to unknown data
pred = predict(mlp, newdata=unknown)
print("Outcome prediction of unlabeled data")
print(table(pred))
unknown$status = pred

# Write the predicted data to a file
df = rbind(df, unknown)
df$id = rownames(df)
write.table(df, "predictions.tsv", quote=F, row.names=F, col.names=T, sep="\t")
