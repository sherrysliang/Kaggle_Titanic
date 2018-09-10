library(dplyr)
library(caret)
library(xgboost)
library(fastDummies)

setwd("/Users/mac/Documents/Kaggle/Titanic/")
train_org <- read.csv("train.csv",stringsAsFactors = F,na.strings = c("NA", "")) # ,stringsAsFactors = F
test_org <- read.csv("test.csv",stringsAsFactors = F,na.strings = c("NA", "")) # ,stringsAsFactors = F
str(train_org) # 891, 12
head(train_org,2)
dim(test_org) # 418 11
head(test_org,2)


all <-bind_rows(train_org,test_org) # test$Survived = NA
head(all)

# Grab title from passenger names
all$Title <- gsub('(.*, )|(\\..*)', '', all$Name)
nrow(all[is.na(all$Title),])

# Show title counts by sex
table(all$Sex, all$Title)

#         Capt Col Don Dona  Dr Jonkheer Lady Major Master Miss Mlle Mme  Mr Mrs  Ms Rev Sir the Countess
# female    0   0   0    1   1        0    1     0      0  260    2   1   0 197   2   0   0            1
# male      1   4   1    0   7        1    0     2     61    0    0   0 757   0   0   8   1            0
#          70  54  40   40  43       38   48    48    5.5   22   24  24  32  37  28  41  49           33
#                           x1
# mean(all[all$Title == "Sir",]$Age, na.rm=TRUE)

unique(all$Title)

## Remove Capitain
# all <- all[-(all$Title == "Capt"),]
# all[all$Title == "Capt",]
# dim(all)

## keep 7 Categories: Col,Dr,Miss,Mrs,Master,Mr,Rev
title_mr <- c("Col","Don","Jonkheer","Major","Mr","Sir")
title_miss <- c("Miss","Mlle","Mme","Ms")
title_mrs <-c("Mrs","the Countess","Dona","Lady")
## reassign mlle, ms, lady -- > Miss; mme --> Mrs; Sir --> Mr
all$Title[all$Title %in% title_miss] <- 'Miss' 
all$Title[all$Title %in% title_mr]   <- 'Mr'
all$Title[all$Title %in% title_mrs] <- 'Mrs' 

unique(all$Title)
all$Title <- as.factor(all$Title)

# reassign age by title group mean
age_table <- aggregate(all$Age, list(all$Title), mean,na.rm=TRUE)
colnames(age_table) <- c("Title","Age_new")
age_table$Age_new <- round(age_table$Age_new,1)
all <- merge(all,age_table,by="Title")
all[is.na(all$Age),]$Age <- all[is.na(all$Age),]$Age_new

all <- all[order(all$PassengerId),]
rownames(all) <- all$PassengerId

##################################################################################
####   After merging, the dataset is in order of Title!!! not PassengerId!!   ####
##################################################################################


all$Sex <- as.factor(all$Sex)
all$Survived <- as.factor(all$Survived)
all$Pclass <- as.ordered(all$Pclass) #because Pclass is ordinal

all$PclassSex[all$Pclass=='1' & all$Sex=='male'] <- 'P1Male'
all$PclassSex[all$Pclass=='2' & all$Sex=='male'] <- 'P2Male'
all$PclassSex[all$Pclass=='3' & all$Sex=='male'] <- 'P3Male'
all$PclassSex[all$Pclass=='1' & all$Sex=='female'] <- 'P1Female'
all$PclassSex[all$Pclass=='2' & all$Sex=='female'] <- 'P2Female'
all$PclassSex[all$Pclass=='3' & all$Sex=='female'] <- 'P3Female'
all$PclassSex <- as.factor(all$PclassSex)

all$Fsize <- all$SibSp+all$Parch +1
all$Alone <- ifelse(all$Fsize>1,0,1)

all$Group <- all$Fsize
all$GroupSize[all$Group==1] <- 'solo'
all$GroupSize[all$Group==2] <- 'duo'
all$GroupSize[all$Group>=3 & all$Group<=4] <- 'group'
all$GroupSize[all$Group>=5] <- 'large group'
all$GroupSize <- as.factor(all$GroupSize)


###### Check NA ########

sapply(all, function(x) {sum(is.na(x))})

all[is.na(all$Embarked),]$Embarked <- "C"
all$Embarked <- as.factor(all$Embarked)

# fill NA Fare -- only 1 case
sapply(all, function(x) {sum(is.na(x))})
all[is.na(all$Fare),]$Fare <- median(all[!is.na(all$Fare) & all$Pclass == 3,]$Fare)

####################################################################################
####################################################################################

colnames(all)
var_to_include <- c("Survived","Pclass","Sex","Age","SibSp","Parch","Fare",
                    "Embarked","PclassSex","GroupSize","Title")  
all_new <- all[,var_to_include]
train_new <- all_new[!is.na(all_new$Survived),]
train_new$Survived <- as.factor(train_new$Survived)
test_new <- all_new[is.na(all_new$Survived),]

head(test_new)

# -------------- GBM --------------------

set.seed(2017)
gbm_new <- caret::train(Survived ~ ., data=train_new, method='gbm', preProcess= c('center', 'scale'), trControl=trainControl(method="cv", number=7), verbose=FALSE)
print(gbm_new)
pred_gbm <-as.numeric(predict(gbm_new, newdata = test_new))-1

result_gbm <- cbind(test_org$PassengerId,pred_gbm)
colnames(result_gbm) <- c("PassengerId","Survived" )
head(result_gbm)
write.csv(result_gbm,"submission_gbm.csv",row.names = FALSE)

# --------------- RF ----------------

set.seed(2017)
rf <- caret::train(x=train_new[,-which(names(train_new) == "Survived")], y=train_new$Survived, data=train_new, method='rf', trControl=trainControl(method="cv", number=5))
rf$results
pred_rf <- as.numeric(predict(rf, newdata = test_new))-1
result_com <- cbind(pred_gbm, pred_rf)

result_rf <- cbind(test_org$PassengerId,pred_rf)
colnames(result_rf) <- c("PassengerId","Survived" )
head(result_rf)
write.csv(result_rf,"submission_rf.csv",row.names = FALSE)

####################################################################################
################ rf performance is always 3%+ higher  than gbm  ####################
####################################################################################

var_factor <- names(train_new)[sapply(train_new,is.factor)]
train_dummy <- dummy_cols(train_new, select_columns = var_factor, remove_first_dummy = TRUE)
head(train_dummy)
X_train <- data.matrix(train_dummy[, !colnames(train_new) %in% c("Survived_1",var_factor)])
colnames(X_train)

y_train <- train_new$Survived

xgb_trcontrol = trainControl(
        method = "cv",
        number = 5,  
        allowParallel = TRUE,
        verboseIter = FALSE,
        returnData = FALSE
)

xgbGrid <- expand.grid(nrounds = c(100,200),  # this is n_estimators in the python code above
                       max_depth = c(10, 15, 20, 25),
                       colsample_bytree = seq(0.5, 0.9, length.out = 5),
                       ## The values below are default values in the sklearn-api. 
                       eta = 0.1,
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1
)

set.seed(0) 
xgb_model = train(
        X_train, y_train,  
        trControl = xgb_trcontrol,
        tuneGrid = xgbGrid,
        method = "xgbTree"
)

xgb_model$bestTune

pred <- as.numeric(predict(xgb_model, X_test))-1
result_xgb <- cbind(test_org$PassengerId,pred)
colnames(result_xgb) <- c("PassengerId","Survived" )
head(result_xgb)
write.csv(result_gbm,"submission_xgb.csv",row.names = FALSE)





