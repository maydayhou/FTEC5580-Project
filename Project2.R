library(lubridate)
library(quantmod)
library(dplyr)
library(pROC)              
library(MASS)
library(MLeval)

setwd('D:/NOW/FTEC5580/project/project2/')
data = read.csv("NVDA.csv",header = TRUE,stringsAsFactors = FALSE)

#set date type
data$Date = ymd(data$Date)

#calculate net return
price =ts(data$Adj.Close)
data$return = periodReturn(price,period='daily') 
data = data[-1,]

## data clean
for (i in 1:5){
  data[paste("return_lag",i,sep='')]=lag(data$return,i,0)
}

for (i in 1:5){
  data[paste("Volume_lag",i,sep='')]=lag(data$Volume,i,0)
}

data1 = data[-(1:5),]
data1$return_d = ifelse(data1$return>0,1,0)

data2 = data1[,-c(2,3,4)]
data2[,c(2:11)] = scale(data2[,c(2:11)],center=T,scale=T) 
summary(data2)

train_set = data2[data2$Date<='2019-01-01',-1]
test_set = data2[data2$Date>'2019-01-01',-1]
test_set$return_d = as.vector(test_set$return_d)
train_set$return_d = as.vector(train_set$return_d)


##model 1 Logistic
glm_train_set = train_set
glm_test_set = test_set

glm0.a = glm(return_d~1,family = binomial(link=logit),data=glm_train_set)
glm1.a = glm(return_d~.,
             family = binomial(link=logit),data=glm_train_set)
anova(glm0.a,glm1.a)
1-pchisq(7.8243,10)

glm_test_set$glm_pre = predict(glm1.a,glm_test_set[,-c(12)],type = "response")

glm.aic <- roc(glm_test_set$return_d,glm_test_set$glm_pre)    
plot(glm.aic,print.auc=T,print.auc.x=0.4,print.auc.y=0.4,print.thres=T,
     print.auc.cex=1.5,print.thres.cex=1.5)

table(glm_test_set$return_d,1*(glm_test_set$glm_pre>0.570))

##train predict
glm_train_set$glm_pre = predict(glm1.a,glm_train_set[,-c(12)],type = "response")

glm.aic1 <- roc(glm_train_set$return_d,glm_train_set$glm_pre)    
plot(glm.aic1,print.auc=T,print.auc.x=0.4,print.auc.y=0.4,print.thres=T,
     print.auc.cex=1.5,print.thres.cex=1.5)

table(1*(glm_train_set$glm_pre>0.538),glm_train_set$return_d)
#model2 lda
lda_train_set = train_set
lda_test_set = test_set
lda_test_set$return_d1 = ifelse(lda_test_set$return_d==1,'YES','NO')

lda.fit=lda(return_d~., data=lda_train_set)
lda.fit

lda.pred = predict(lda.fit, lda_test_set[,-12])
lda_test_set$lda_pre = lda.pred$class
table(lda_test_set$lda_pre, lda_test_set$return_d)

1-mean (lda_test_set$lda_pre == lda_test_set$return_d)



lda_test_set$return_d1 = ifelse(lda_test_set$return_d==1,'YES','NO')
lda_1 = data.frame(lda.pred$posterior,lda_test_set$return_d1)
names(lda_1) = c('NO','YES','return_d')
lda.eval=evalm(lda_1)
names(lda.eval)
#####train predict
lda.pred1 = predict(lda.fit, lda_train_set[,-12])
lda_train_set$lda_pre = lda.pred1$class
table(lda_train_set$lda_pre, lda_train_set$return_d)

1-mean (lda_train_set$lda_pre == lda_train_set$return_d)



lda_train_set$return_d1 = ifelse(lda_train_set$return_d==1,'YES','NO')
lda_2 = data.frame(lda.pred1$posterior,lda_train_set$return_d1)
names(lda_2) = c('NO','YES','return_d')
lda.eval1=evalm(lda_2)


#########qda
qda_train_set = train_set
qda_test_set = test_set
qda.fit=qda(return_d~., data=qda_train_set)
qda.fit

qda.pred = predict(qda.fit, qda_test_set[,-12])
qda_test_set$qda_pre = qda.pred$class
table(qda_test_set$qda_pre, qda_test_set$return_d)

qda_test_set$return_d1 = ifelse(qda_test_set$return_d==1,'YES','NO')
qda_1 = data.frame(qda.pred$posterior,qda_test_set$return_d1)
names(qda_1) = c('NO','YES','return_d')
qda.eval=evalm(qda_1)

###train predict
qda.pred1 = predict(qda.fit, qda_train_set[,-12])
qda_train_set$qda_pre = qda.pred1$class
table(qda_train_set$qda_pre, qda_train_set$return_d)

qda_train_set$return_d1 = ifelse(qda_train_set$return_d==1,'YES','NO')
qda_1 = data.frame(qda.pred1$posterior,qda_train_set$return_d1)
names(qda_1) = c('NO','YES','return_d')
qda.eval=evalm(qda_1)

###############decision tree
library(tree)
dt_train_set = train_set
dt_test_set = test_set
dt_train_set$return_d1 = as.factor(ifelse(dt_train_set$return_d==1,'YES','NO'))
dt_test_set$return_d1 = as.factor(ifelse(dt_test_set$return_d==1,'YES','NO'))

dt1 = dt_train_set[,-11]
tree.fit = tree(return_d1~.-return_d, data=dt_train_set)
summary(tree.fit)

###########bagging random forest
library(randomForest)
set.seed(1)
bag.fit = randomForest(return_d1~.-return_d, data=dt_train_set, mtry=3, ntree=500, importance=TRUE)
bag.fit
# Random Forest
RF = randomForest(return_d1~.-return_d, data=dt_train_set, mtry=1, ntree= 500, importance=TRUE)
RF

importance(bag.fit)
varImpPlot(bag.fit)
importance(RF)
varImpPlot(RF)

bag.pred = predict(bag.fit,newdata=dt_test_set)
table(bag.pred,dt_test_set$return_d1)
RF.pred = predict(RF,newdata=dt_test_set)
table(RF.pred,dt_test_set$return_d1)

bag.pred1 = predict(bag.fit)
table(bag.pred1,dt_train_set$return_d1)
RF.pred1 = predict(RF)
table(RF.pred1,dt_train_set$return_d1)

#############boosting
library(gbm)
set.seed(1)
bst = gbm(train_set$return_d~., data=train_set, distribution="bernoulli", n.trees =500, 
          interaction.depth=1, shrinkage=0.1, cv.folds=10)
bst
summary(bst)
which.min(bst$cv.error)

bst.pred=predict(bst, newdata=test_set,type="response", n.trees=8)
bst.pred.class=ifelse(bst.pred>0.5,1,0)
table(bst.pred.class,test_set$return_d)

bst.pred1=predict(bst,type="response", n.trees=8)
bst.pred1.class=ifelse(bst.pred1>0.5,1,0)
table(bst.pred1.class,train_set$return_d)
####SVM
library(e1071 )

set.seed(1)
tune.fit =tune(svm, return_d1~.-return_d, data=dt_train_set, kernel="linear",
               ranges=list(cost=c(0.001 , 0.01, 0.1, 1,5,10,100)))
summary(tune.fit)


# Show the best model 
bm=tune.fit$best.model
summary(bm)

# Prediction 
svm.fit.pred=predict(bm, dt_test_set)
table(svm.fit.pred, dt_test_set$return_d1)

svm.fit1.pred=predict(bm)
table(svm.fit1.pred, dt_train_set$return_d1)

#To use nonlinear SVMs, you can use the same functions as before but specify kernel="polynomial" or kernel="radial".

psvm.fit =svm(return_d1~.-return_d, data=dt_train_set, 
              kernel="polynomial", cost=100, scale=FALSE)
psvm.fit.pred=predict(psvm.fit, dt_test_set)
table(psvm.fit.pred, dt_test_set$return_d1)

psvm.fit.pred1=predict(psvm.fit, dt_train_set)
table(psvm.fit.pred1, dt_train_set$return_d1)

rsvm.fit=svm(return_d1~.-return_d, data=dt_train_set,
             kernel="radial", cost=100, scale=FALSE)
rsvm.fit.pred=predict(rsvm.fit, dt_test_set)
table(rsvm.fit.pred, dt_test_set$return_d1)

rsvm.fit1.pred=predict(rsvm.fit, dt_train_set)
table(rsvm.fit1.pred, dt_train_set$return_d1)