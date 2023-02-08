library(dplyr)
library(caret)
library(mice)
library(neuralnet)
library(xgboost)
library(tidyverse)
library(ggplot2)
library(randomForest)
library(pROC)
#importing in dataset 
library(readr)
aug_train <- read_csv("aug_train.csv")
aug_train <- aug_train %>%
  mutate(
    city = as.factor(city),
    gender = as.factor(gender),
    relevent_experience = as.factor(relevent_experience),
    enrolled_university = as.factor(enrolled_university),
    education_level = as.factor(education_level),
    major_discipline = as.factor(major_discipline),
    experience = as.factor(experience),
    company_size = as.factor(company_size),
    company_type = as.factor(company_type),
    last_new_job = as.factor(last_new_job),
    target = as.factor(target)
  )
#Data preprocessing step 
summary(aug_train)
colSums(is.na(aug_train))
#Adjusting experience variable 
aggregate(data.frame(count = aug_train$experience), list(value = aug_train$experience), length)
sum(is.na(aug_train$experience))
aug_train=aug_train %>% 
  mutate(experience = ifelse(as.character(experience) == "<1", "0", as.character(experience))) %>%
  mutate(experience = ifelse(as.character(experience) == ">20", "21", as.character(experience))) 
  
#Adjusting last new job variable 
aggregate(data.frame(count = aug_train$last_new_job), list(value = aug_train$last_new_job), length)
sum(is.na(aug_train$last_new_job))
aug_train=aug_train %>% 
  mutate(last_new_job = ifelse(as.character(last_new_job) == "never", "0", as.character(last_new_job))) %>%
  mutate(last_new_job = ifelse(as.character(last_new_job) == ">4", "5", as.character(last_new_job))) 

#Adjusting company size
aggregate(data.frame(count = aug_train$company_size), list(value = aug_train$company_size), length)
aug_train=aug_train %>% 
  mutate(company_size = ifelse(as.character(company_size) == "10/49", "10-49", as.character(company_size))) %>%
  mutate(company_size = ifelse(as.character(company_size) == "10000+", ">10000", as.character(company_size))) 

#Adjusting education level and major discipline
#Subsetting by highschool education level
aggregate(data.frame(count = aug_train$education_level), list(value = aug_train$education_level), length)
aug_train_ed_lvl=subset(aug_train, education_level=='High School')
aggregate(data.frame(count = aug_train_ed_lvl$major_discipline), list(value = aug_train_ed_lvl$major_discipline), length)
hs_edlvl=sum(is.na(aug_train_ed_lvl$major_discipline))

#Subsetting by primary school education level
aug_train_ed_lvl=subset(aug_train, education_level=='Primary School')
aggregate(data.frame(count = aug_train_ed_lvl$major_discipline), list(value = aug_train_ed_lvl$major_discipline), length)
ps_edlvl=sum(is.na(aug_train_ed_lvl$major_discipline))

#Subsetting by NA education level
aug_train_ed_lvl=subset(aug_train, is.na(education_level))
aggregate(data.frame(count = aug_train_ed_lvl$major_discipline), list(value = aug_train_ed_lvl$major_discipline), length)
na_edlvl=sum(is.na(aug_train_ed_lvl$major_discipline))

#Subsetting by Graduate education level
aug_train_ed_lvl=subset(aug_train, education_level=='Graduate')
aggregate(data.frame(count = aug_train_ed_lvl$major_discipline), list(value = aug_train_ed_lvl$major_discipline), length)
grad_edlvl=sum(is.na(aug_train_ed_lvl$major_discipline))

#Subsetting by Masters level
aug_train_ed_lvl=subset(aug_train, education_level=='Masters')
aggregate(data.frame(count = aug_train_ed_lvl$major_discipline), list(value = aug_train_ed_lvl$major_discipline), length)
ms_edlvl=sum(is.na(aug_train_ed_lvl$major_discipline))

#Subsetting by Phd level
aug_train_ed_lvl=subset(aug_train, education_level=='Phd')
aggregate(data.frame(count = aug_train_ed_lvl$major_discipline), list(value = aug_train_ed_lvl$major_discipline), length)
phd_edlvl=sum(is.na(aug_train_ed_lvl$major_discipline))

#Removing records with NA education level and NA for major discipline
aug_train_mod = aug_train[!(is.na(aug_train$education_level) & is.na(aug_train$major_discipline)), ]
colSums(is.na(aug_train))
colSums(is.na(aug_train_mod))

#Adjusting Gender
aggregate(data.frame(count = aug_train_mod$gender), list(value = aug_train_mod$gender), length)
sum(is.na(aug_train_mod$gender))

# readjusting variable types
aug_train <- aug_train %>%
  mutate(
    experience = as.numeric(experience),
    company_size = as.factor(company_size),
    last_new_job = as.numeric(last_new_job),
  )

#Imputations using mice
#Training Data
init = mice(aug_train, maxit=0) 
meth = init$method
predM = init$predictorMatrix
predM[, c("enrollee_id","target")]=0
meth[c("city","city_development_index","relevent_experience","training_hours")]=""
meth[c("gender")]="polyreg"
meth[c("enrolled_university")]="polyreg"
meth[c("education_level")]="polyreg"
meth[c("major_discipline")]="polyreg"
meth[c("experience")]="norm" 
meth[c("company_size")]="polyreg"
meth[c("company_type")]="polyreg"
meth[c("last_new_job")]="norm" 
set.seed(103)
imputed = mice(aug_train, method=meth, predictorMatrix=predM, m=5)
imputed = complete(imputed)
sapply(imputed, function(x) sum(is.na(x)))

write.csv(imputed, "imputed_aug_train.csv")

#Start here 
library(dplyr)
library(caret)
library(mice)
#importing in dataset 
library(readr)
#reading in csv of imputed data
imputed=read.csv("imputed_aug_train.csv")

#Readjusting variable types 
imputed <- imputed %>%
  mutate(
    city = as.character(city),
    gender = as.factor(gender),
    relevent_experience = as.factor(relevent_experience),
    enrolled_university = as.factor(enrolled_university),
    education_level = as.factor(education_level),
    major_discipline = as.factor(major_discipline),
    experience = as.numeric(experience),
    company_size = as.factor(company_size),
    company_type = as.factor(company_type),
    last_new_job = as.numeric(last_new_job),
    target = as.factor(target)
  )
 
train_imputed=imputed
#Dealing unbalanced dataset by oversampling
df_pos_job_change_ind <- which(train_imputed$target == 1)
df_neg_job_change_ind <- which(train_imputed$target == 0)
### setting negative counts to be same as positive counts - so that the data is balanced
oversample_df1 <- train_imputed[c(df_pos_job_change_ind, df_neg_job_change_ind , df_pos_job_change_ind), ]
dim(oversample_df1)
table(oversample_df1$target)

#Oversampling is better technique removing none of our data like undersampling technique, which removed over half the dataset.

#Taking top 35 city ID's

sort(table(oversample_df1$city), decreasing = TRUE)[1:39]

top_39=tail(names(sort(table(oversample_df1$city))), 39)

for(i in 1:nrow(oversample_df1)){
  if(oversample_df1$city[i] %in% top_39){
    oversample_df1$city[i] <- oversample_df1$city[i]
  }
  else(oversample_df1$city[i]<-"Other")
}

oversample_df1 <- oversample_df1 %>%
  mutate(
    city = as.factor(city),
    gender = as.factor(gender),
    relevent_experience = as.factor(relevent_experience),
    enrolled_university = as.factor(enrolled_university),
    education_level = as.factor(education_level),
    major_discipline = as.factor(major_discipline),
    experience = as.numeric(experience),
    company_size = as.factor(company_size),
    company_type = as.factor(company_type),
    last_new_job = as.numeric(last_new_job),
    target = as.factor(target)
  )

#Splitting train into train and Test
#Drop enrollee_id
oversample_df1_noID=subset(oversample_df1, select = -c(X,enrollee_id))
library(caret)
set.seed(346)
trainindex=createDataPartition(oversample_df1_noID$target,p=.7,
                               list=FALSE,
                               times=1)

Train=oversample_df1_noID[trainindex,]
Valid=oversample_df1_noID[-trainindex,]



#Train100=sample_n(Train,100)

#Building Bagging method
library(randomForest)
set.seed(2)
#Determining how many trees needed
# bag.job=randomForest(target~., data=Train, ntree=2000,mtry=12,importance=T, proximity=T)
# plot(bag.job)
#Final bagging model
bag.job=randomForest(target~., data=Train, ntree=250,mtry=12,importance=T, proximity=T,do.trace=T)
#verifying train error
bag.predT = predict(bag.job, Train)
mean(bag.predT == Train$target)
1-mean(bag.predT == Train$target)
#Test error
bag.predV = predict(bag.job, Valid)
mean(bag.predV == Valid$target)
1-mean(bag.predV == Valid$target)
varImp(bag.job)
plot(bag.job)

#Confusion Matrix
#Train
probsTrain <- predict(bag.job,Train, type = "prob")
threshold <- 0.5
pred<- factor( ifelse(probsTrain[, 1] > threshold, 0, 1) )
pred<- relevel(pred, 1)  
confusionMatrix(pred, Train$target)
library(ggplot2)
Target <- factor(c(0, 0, 1, 1))
Prediction <- factor(c(0, 1, 0, 1))
Y<- c(10058,9,7,6681)
df <- data.frame(Target, Prediction, Y)
ggplot(data = df, mapping = aes(x = Target, y = Prediction)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "#009194") +
  theme_bw() + theme(legend.position = "none")
#Test
probsTest <- predict(bag.job,Valid, type = "prob")
threshold <- 0.5
pred<- factor( ifelse(probsTest[, 1] > threshold, 0, 1) )
pred<- relevel(pred, 1)  
confusionMatrix(pred, Valid$target)
Target <- factor(c(0, 0, 1, 1))
Prediction <- factor(c(0, 1, 0, 1))
Y<- c(3550,764,455,2411)
df <- data.frame(Target, Prediction, Y)
ggplot(data = df, mapping = aes(x = Target, y = Prediction)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "#009194") +
  theme_bw() + theme(legend.position = "none")
#ROC
library(pROC)
#Train
probsTrain <- predict(bag.job, Train, type = "prob")
rocCurve   <- roc(response = Train$target,
                  predictor = probsTrain[, 1],
                  levels = rev(levels(Train$target)),
                  plot = TRUE, 
                  print.auc = TRUE,
                  print.thres = "best")
#Test
probsTest <- predict(bag.job, Valid, type = "prob")
rocCurve   <- roc(response = Valid$target,
                  predictor = probsTrain[, 1],
                  levels = rev(levels(Valid$target)),
                  plot = TRUE, 
                  print.auc = TRUE,
                  print.thres = "best")
#random forest
set.seed(23)
#Finding best mtry
tuning<-vector(length =12)
for (i in 1:12) {
  
  rf.job<-randomForest(target~.,mtry = i,data=Train, ntree = 250, importance=T, proximity=T);
  
  tuning[i]<-rf.job$err.rate[250,1];
  
}
#getting test errors for mtry 
plot(tuning)
tuning
#Rerunning rf with best mtry
rf.job = randomForest(target~., data=Train, mtry=5,ntree=250, importance=T,do.trace=T)
plot(rf.job)
varImp(rf.job)
#verifying train error
rf.predT = predict(rf.job, Train)
mean(rf.predT == Train$target)
1-mean(rf.predT == Train$target)
#Test error
rf.predV = predict(rf.job, Valid)
mean(rf.predV == Valid$target)
1-mean(rf.predV == Valid$target)
#Confusion Matrix
#Train
probsTrain <- predict(rf.job,Train, type = "prob")
threshold <- 0.5
pred<- factor( ifelse(probsTrain[, 1] > threshold, 0, 1) )
pred<- relevel(pred, 1)  
confusionMatrix(pred, Train$target)
Target <- factor(c(0, 0, 1, 1))
Prediction <- factor(c(0, 1, 0, 1))
Y<- c(10050,17,34,6654)
df <- data.frame(Target, Prediction, Y)
ggplot(data = df, mapping = aes(x = Target, y = Prediction)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "#009194") +
  theme_bw() + theme(legend.position = "none")
#Test
probsTest <- predict(rf.job,Valid, type = "prob")
threshold <- 0.5
pred<- factor( ifelse(probsTest[, 1] > threshold, 0, 1) )
pred<- relevel(pred, 1)  
confusionMatrix(pred, Valid$target)
Target <- factor(c(0, 0, 1, 1))
Prediction <- factor(c(0, 1, 0, 1))
Y<- c(3608,706,458,2408)
df <- data.frame(Target, Prediction, Y)
ggplot(data = df, mapping = aes(x = Target, y = Prediction)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "#009194") +
  theme_bw() + theme(legend.position = "none")
#ROC
#Train
probsTrain <- predict(rf.job, Train, type = "prob")
rocCurve   <- roc(response = Train$target,
                  predictor = probsTrain[, 1],
                  levels = rev(levels(Train$target)),
                  plot = TRUE, 
                  print.auc = TRUE,
                  print.thres = "best")
#Test
probsTest <- predict(rf.job, Valid, type = "prob")
rocCurve   <- roc(response = Valid$target,
                  predictor = probsTest[, 1],
                  levels = rev(levels(Valid$target)),
                  plot = TRUE, 
                  print.auc = TRUE,
                  print.thres = "best")

#dummy variables for categorical data
dmy <- dummyVars(" ~.-target", data = oversample_df1_noID)
trsf <- data.frame(predict(dmy, newdata = oversample_df1_noID))
trsf$target=oversample_df1_noID$target
trsf
#splitting into train and Test datasets
set.seed(7)
trainindex=createDataPartition(trsf$target,p=.7,
                               list=FALSE,
                               times=1)

Train_trsf=trsf[trainindex,]
Valid_trsf=trsf[-trainindex,]

# Fit the xgboost model on the training set
library(xgboost)
set.seed(50)
xgb.model <- train(
  target ~., data = Train_trsf, method = "xgbTree",
  trControl = trainControl("cv", number = 10)
)
plot(xgb.model)
# Best tuning parameter
xgb.model$bestTune
#Variable importance
library(tidyverse)
varImp(xgb.model)
plot(xgbImpVar)
xgbImpVar=varImp(xgb.model)$importance %>% 
  as.data.frame()
xgbImpVar20=head(xgbImpVar,20)
xgbImpVar20%>%
  rownames_to_column() %>%
  arrange(Overall) %>%
  mutate(rowname = forcats::fct_inorder(rowname )) %>%
  ggplot()+
  geom_col(aes(x = rowname, y = Overall))+
  coord_flip()+
  theme_bw()
# Make predictions on the Training data
xgb.predT <- xgb.model %>% predict(Train_trsf)
head(xgb.predT)

# Compute model prediction accuracy rate
mean(xgb.predT == Train_trsf$target)

# Make predictions on the Test data
xgb.predV <- xgb.model %>% predict(Valid_trsf)
head(xgb.predV)

# Compute model prediction Test accuracy rate
mean(xgb.predV == Valid_trsf$target)

#Confusion Matrix
#Train
probsTrain <- predict(xgb.model,Train_trsf, type = "prob")
threshold <- 0.5
pred<- factor( ifelse(probsTrain[, 1] > threshold, 0, 1) )
pred<- relevel(pred, 1)  
confusionMatrix(pred, Train_trsf$target)
Target <- factor(c(0, 0, 1, 1))
Prediction <- factor(c(0, 1, 0, 1))
Y<- c(8609,1458,2711,3977)
df <- data.frame(Target, Prediction, Y)
ggplot(data = df, mapping = aes(x = Target, y = Prediction)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "#009194") +
  theme_bw() + theme(legend.position = "none")
#Test
probsTest <- predict(xgb.model,Valid_trsf, type = "prob")
threshold <- 0.5
pred<- factor( ifelse(probsTest[, 1] > threshold, 0, 1) )
pred<- relevel(pred, 1)  
confusionMatrix(pred, Valid_trsf$target)
Target <- factor(c(0, 0, 1, 1))
Prediction <- factor(c(0, 1, 0, 1))
Y<- c(3623,691,1221,1645)
df <- data.frame(Target, Prediction, Y)
ggplot(data = df, mapping = aes(x = Target, y = Prediction)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "#009194") +
  theme_bw() + theme(legend.position = "none")
#ROC
#Train
probsTrain <- predict(xgb.model, Train_trsf, type = "prob")
rocCurve   <- roc(response = Train_trsf$target,
                  predictor = probsTrain[, 1],
                  levels = rev(levels(Train_trsf$target)),
                  plot = TRUE, 
                  print.auc = TRUE,
                  print.thres = "best")
#Test
probsTest <- predict(xgb.model, Valid_trsf, type = "prob")
rocCurve   <- roc(response = Valid_trsf$target,
                  predictor = probsTest[, 1],
                  levels = rev(levels(Valid_trsf$target)),
                  plot = TRUE, 
                  print.auc = TRUE,
                  print.thres = "best")

#Neural net in R
library(neuralnet)
set.seed(2)
inf.nn = neuralnet(target ~ ., data = Train_trsf, hidden = 1, act.fct = "logistic",err.fct = "ce", linear.output=F,threshold=0.1, rep = 1)

load("nerualnet.RData")
#Train
set.seed(2)
probsTrain <- predict(inf.nn,Train_trsf, type = "prob")
threshold <- 0.5
pred<- factor( ifelse(probsTrain[, 1] > threshold, 0, 1) )
pred<- relevel(pred, 1)  
confusionMatrix(pred, Train_trsf$target)
Target <- factor(c(0, 0, 1, 1))
Prediction <- factor(c(0, 1, 0, 1))
Y<- c(8354,1713,2858,3830)
df <- data.frame(Target, Prediction, Y)
ggplot(data = df, mapping = aes(x = Target, y = Prediction)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "#009194") +
  theme_bw() + theme(legend.position = "none")
#Test
set.seed(2)
probsTest <- predict(inf.nn,Valid_trsf, type = "prob")
threshold <- 0.5
pred<- factor( ifelse(probsTest[, 1] > threshold, 0, 1) )
pred<- relevel(pred, 1)  
confusionMatrix(pred, Valid_trsf$target)
Target <- factor(c(0, 0, 1, 1))
Prediction <- factor(c(0, 1, 0, 1))
Y<- c(3553,761,1214,1652)
df <- data.frame(Target, Prediction, Y)
ggplot(data = df, mapping = aes(x = Target, y = Prediction)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "#009194") +
  theme_bw() + theme(legend.position = "none")
#ROC
#Train
probsTrain <- predict(inf.nn, Train_trsf, type = "prob")
rocCurve   <- roc(response = Train_trsf$target,
                  predictor = probsTrain[, 1],
                  levels = rev(levels(Train_trsf$target)),
                  plot = TRUE, 
                  print.auc = TRUE,
                  print.thres = "best")
#Test
probsTest <- predict(inf.nn, Valid_trsf, type = "prob")
rocCurve   <- roc(response = Valid_trsf$target,
                  predictor = probsTest[, 1],
                  levels = rev(levels(Valid_trsf$target)),
                  plot = TRUE, 
                  print.auc = TRUE,
                  print.thres = "best")
