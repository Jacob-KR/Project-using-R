library(glmnet)
library(rpart)
library(randomForest)

# Read the cleaned csv file
House <- read.csv('A:/Fall 1/Data Science/dataset/Airbnb Data/NY_final.csv',sep=',',header=TRUE)
House$log_price <- log(House$price)
House <- House[House$price>0 & House$price <= 2000,]

# Split data set: 70% train data 30% test data
set.seed(100)
train <- sample(nrow(House), nrow(House)*0.7)
house_train <- House[train, -c(1,3,4,5,6,9)]
house_test <- House[-train, -c(1,3,4,5,6,9)]

# Change Data Type to train the dataset
x <- as.matrix(house_train[, -63])
y <- house_train$log_price

########################################################################################
########################################################################################
### Generalized Linear Model
### Compare 3 models
### Models to compare:
### 1. linear: Multiple Linear regression
### 2. lasso: Lasso Regression to select min choice of lambda
### 3. ridge : Ridge Regression to select min choice of lambda

##### Lasso #####
lasso <- glmnet(x,y,alpha=1)
lasso_cv <- cv.glmnet(x,y,alpha = 1)
par(mar=c(1.5,1.5,2,1.5))
par(mai=c(1.5,1.5,2,1.5))
plot(lasso_cv, main="Fitting Graph for CV Lasso \n \n # of non-zero coefficients  ", xlab = expression(paste("log(",lambda,")")))
lambda_lasso <- lasso_cv$lambda.min  

##### Ridge #####
ridge <- glmnet(x,y,alpha=0)
ridge_cv <- cv.glmnet(x,y,alpha = 0)
par(mar=c(1.5,1.5,2,1.5))
par(mai=c(1.5,1.5,2,1.5))
plot(ridge_cv, main="Fitting Graph for CV Ridge \n \n # of non-zero coefficients  ", xlab = expression(paste("log(",lambda,")")))
lambda_ridge <- ridge_cv$lambda.min  

##### K fold #####
set.seed(18)
n <- nrow(house_train)
nfold <- 10
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]
OOS <- data.frame(linear=rep(NA,nfold),lasso=rep(NA,nfold), ridge=rep(NA,nfold))


for(k in 1:nfold){ 
  train <- which(foldid!=k) # train on all but fold `k'
  
  ### This is the CV for the Linear estimates  
  linear <- glm(log_price~., data=house_train[train,])
  predlinear <- predict(linear, newdata=house_train[-train,], type="response")
  OOS$linear[k] <- 1-sum((y[-train]-predlinear)^2)/sum((y[-train]-mean(y[-train]))^2) 
  
  ### This is the CV for the Lasso Estimates
  lassomin  <- glmnet(x[train,],y[train],alpha=1,lambda = lambda_lasso)
  predlassomin <- predict(lassomin, newx=x[-train,], type="response")
  OOS$lasso[k] <- 1-sum((y[-train]-predlassomin)^2)/sum((y[-train]-mean(y[-train]))^2)
  
  ### This is the CV for the Ridge estimates  
  ridgemin  <- glmnet(x[train,],y[train],alpha=0,lambda = lambda_ridge)
  predridgemin <- predict(ridgemin, newx=x[-train,], type="response")
  OOS$ridge[k] <- 1-sum((y[-train]-predridgemin)^2)/sum((y[-train]-mean(y[-train]))^2)
  
  
  ###
  print(paste("Iteration",k,"of",nfold,"completed"))
}

colMeans(OOS)
barplot(colMeans(OOS), las=2,xpd=FALSE , xlab="", ylim=c(0,1), ylab = bquote( "Average Out of Sample "~R^2))

m.OOS <- as.matrix(OOS)
rownames(m.OOS) <- c(1:nfold)
barplot(t(as.matrix(OOS)), beside=TRUE, legend=TRUE, args.legend=c(xjust=1, yjust=0.5),
        ylab= bquote( "Out of Sample" ~ R^2), xlab="Fold", names.arg = c(1:10),ylim=c(0,0.7))

# Predict the test data
test.x<- as.matrix(house_test[, -63])
test.lasso <- predict(lassomin, newx=test.x, type="response")
test.ridge <- predict(ridgemin, newx=test.x, type="response")
test.linear <- predict(linear,house_test, type="response")
R2.lasso <- 1-sum((house_test$log_price-test.lasso)^2)/sum((house_test$log_price-mean(house_test$log_price))^2)
plot(house_test$log_price,test.lasso, main = 'test data',xlab = 'Actual Price', ylab = 'Predicted Price',xlim=c(2,8),ylim=c(2,8))
R2.ridge <- 1-sum((house_test$log_price-test.ridge)^2)/sum((house_test$log_price-mean(house_test$log_price))^2)
plot(house_test$log_price,test.ridge, main = 'test data',xlab = 'Actual Price', ylab = 'Predicted Price',xlim=c(2,8),ylim=c(2,8))
R2.linear <- 1-sum((house_test$log_price-test.linear)^2)/sum((house_test$log_price-mean(house_test$log_price))^2)
plot(house_test$log_price,test.linear, main = 'test data',xlab = 'Actual Price', ylab = 'Predicted Price',xlim=c(2,8),ylim=c(2,8))


######################################################################################
######################################################################################
### Regression Tree
### Compare 2 models
### Models to compare:
### 1. train.tree: decision tree
### 2. train.RF: Random Forest

##### Decision Tree #####
tree <- rpart(log_price~.,method = "anova",data=house_train,control = list(cp = 0, xval = 10))
printcp(tree)
plotcp(tree)
# ‘CP’ stands for Complexity Parameter of the tree
ptree<- prune(tree,cp= tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"])
printcp(ptree)
plotcp(ptree)
abline(v = 20, lty = "dashed")
train.tree<- prune(tree,cp= tree$cptable[20,"CP"])
printcp(train.tree)

# Output to be present as PNG file
setwd("A:/Fall 1/Data Science/dataset/Airbnb Data")
png(file = "DecisionTree.png", width = 600,height = 600)

# Plot the tree to see how it splits
plot(train.tree, uniform = TRUE,main = "Price Prediction by Decision Tree")
text(train.tree, use.n = TRUE, cex = .65)

# Save the file
dev.off()

# Test the data
test.tree <- predict(train.tree, house_test, method = "anova")
R2.tree <- 1-sum((house_test$log_price-test.tree)^2)/sum((house_test$log_price-mean(house_test$log_price))^2)
plot(house_test$log_price, test.tree, main = 'test data',xlab = 'Actual Price', ylab = 'Predicted Price')

##### Random Forest #####
train.RF <- randomForest(log_price~.,data=house_train,importance=TRUE,ntree=500)
train.RF 
# %explained variance is a measure of how well out-of-bag predictions explain the target variance of the training set. 

# Variables order by Importance based on %IncMSE (or per cent increase in mean squared error)
importance_train <- train.RF$importance
importance_train <- data.frame(importance_train)
importance_train <- importance_train[order(importance_train$X.IncMSE, decreasing = TRUE), ]
head(importance_train)

# Feature Selection
set.seed(100)
house_train.cv <- rfcv(house_train[,-c(63)], house_train$log_price, cv.fold = 5)
with(house_train.cv, plot(n.var, error.cv, log="x", type="o", lwd=2))
# All the variables need to be added


# Use test data to estimate the accuracy
test.RF <- predict(train.RF, house_test)
R2.randomforest <- 1-sum((house_test$log_price-test.RF)^2)/sum((house_test$log_price-mean(house_test$log_price))^2)
plot(house_test$log_price, test.RF, main = 'test data',xlab = 'Actual Price', ylab = 'Predicted Price',xlim=c(2,8),ylim=c(2,8))



########################################################################################
########################################################################################
### Generalized Linear Model With Ineraction
### Compare 3 models
### Models to compare:
### 1. linearInter: Linear regression with interactions
### 2. lassoInter: Lasso Regression with interactions to select min choice of lambda
### 3. ridgeInter : Ridge Regression with interactions to select min choice of lambda

x<- model.matrix(~.^2, data=house_train[-63])
y<- house_train$log_price
##### Lasso #####
lassoInter <- glmnet(x,y,alpha=1)
lassoInter_cv <- cv.glmnet(x,y,alpha = 1)
par(mar=c(1.5,1.5,2,1.5))
par(mai=c(1.5,1.5,2,1.5))
plot(lassoInter_cv, main="Fitting Graph for CV Lasso \n \n # of non-zero coefficients  ", xlab = expression(paste("log(",lambda,")")))
lambda_lassoInter <- lassoInter_cv$lambda.min  

##### Ridge #####
ridgeInter <- glmnet(x,y,alpha=0)
ridgeInter_cv <- cv.glmnet(x,y,alpha = 0)
par(mar=c(1.5,1.5,2,1.5))
par(mai=c(1.5,1.5,2,1.5))
plot(ridgeInter_cv, main="Fitting Graph for CV Ridge \n \n # of non-zero coefficients  ", xlab = expression(paste("log(",lambda,")")))
lambda_ridgeInter <- ridgeInter_cv$lambda.min  

##### K fold #####
set.seed(18)
n <- nrow(house_train)
nfold <- 10
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]
OOS.Inter <- data.frame(linearInter=rep(NA,nfold),lassoInter=rep(NA,nfold), ridgeInter=rep(NA,nfold))


for(k in 1:nfold){ 
  train <- which(foldid!=k) # train on all but fold `k'
  
  ### This is the CV for the Linear Interactions estimates  
  linearInter <- glm(log_price~.^2, data=house_train[train,])
  predlinearInter <- predict(linearInter, newdata=house_train[-train,], type="response")
  OOS.Inter$linearInter[k] <- 1-sum((y[-train]-predlinearInter)^2)/sum((y[-train]-mean(y[-train]))^2) 
  
  
  ### This is the CV for the Lasso Interactions Estimates
  lassominInter  <- glmnet(x[train,],y[train],alpha=1,lambda = lambda_lassoInter)
  predlassominInter <- predict(lassominInter, newx=x[-train,], type="response")
  OOS.Inter$lassoInter[k] <- 1-sum((y[-train]-predlassominInter)^2)/sum((y[-train]-mean(y[-train]))^2)
  
  ### This is the CV for the Ridge Interactions estimates  
  ridgeminInter  <- glmnet(x[train,],y[train],alpha=0,lambda = lambda_ridgeInter)
  predridgeminInter <- predict(ridgeminInter, newx=x[-train,], type="response")
  OOS.Inter$ridgeInter[k] <- 1-sum((y[-train]-predridgeminInter)^2)/sum((y[-train]-mean(y[-train]))^2)
  
  ###
  print(paste("Iteration",k,"of",nfold,"completed"))
}

colMeans(OOS.Inter)
barplot(colMeans(OOS.Inter), las=2,xpd=FALSE , xlab="", ylim=c(0,1), ylab = bquote( "Average Out of Sample "~R^2))

m.OOSInter <- as.matrix(OOS.Inter)
rownames(m.OOSInter) <- c(1:nfold)
barplot(t(as.matrix(OOS.Inter)), beside=TRUE, legend=TRUE, args.legend=c(xjust=1, yjust=0.5),
        ylab= bquote( "Out of Sample" ~ R^2), xlab="Fold", names.arg = c(1:10))

# Use test data to estimate the accuracy
test.x<- model.matrix(~.^2, data=house_test[-63])
test.lassoInter <- predict(lassominInter, newx=test.x, type="response")
test.ridgeInter <- predict(ridgeminInter, newx=test.x, type="response")
test.linearInter <- predict(linearInter,house_test, type="response")
R2.lassoInter <- 1-sum((house_test$log_price-test.lassoInter)^2)/sum((house_test$log_price-mean(house_test$log_price))^2)
plot(house_test$log_price,test.lassoInter, main = 'test data',xlab = 'Actual Price', ylab = 'Predicted Price',xlim=c(2,8),ylim=c(2,8))
R2.ridgeInter <- 1-sum((house_test$log_price-test.ridgeInter)^2)/sum((house_test$log_price-mean(house_test$log_price))^2)
plot(house_test$log_price,test.ridgeInter, main = 'test data',xlab = 'Actual Price', ylab = 'Predicted Price',xlim=c(2,8),ylim=c(2,8))
R2.linearInter <- 1-sum((house_test$log_price-test.linearInter)^2)/sum((house_test$log_price-mean(house_test$log_price))^2)
plot(house_test$log_price,test.linearInter, main = 'test data',xlab = 'Actual Price', ylab = 'Predicted Price',xlim=c(2,8),ylim=c(2,8))



par(mar=c(1.5,1.5,1,1))
par(mai=c(1.5,1.5,1,1))
barplot(c(R2.linear, R2.lasso,R2.ridge,  R2.tree,R2.randomforest,R2.linearInter,R2.lassoInter,R2.ridgeInter), las=2, xlab="", 
        names = c("linear\n regression","lasso\n regression","ridge\n regression","tree\n model","random\n forest","linear with\n Interaction","lasso with\n  Interaction","ridge with\n  Interaction"), ylab = bquote(R^2))






