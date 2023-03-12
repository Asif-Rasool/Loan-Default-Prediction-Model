# R script for Super Cool Bank Credit Risk Modeling
# Author: Asif Rasool
# email: asif.rasool@outlook.com

# Loading the required packages

# install.packages("caret")

suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(cowplot))
suppressPackageStartupMessages(library(randomForest))
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(naniar))
suppressPackageStartupMessages(library(lessR))

set.seed(6666)

# Loading and cleaning  the train data

train.data <-  read.csv("Training_R-197119_Candidate Attach.csv", 
                        header = TRUE, na.strings = c("", "NA"))
str(train.data)


train.data[train.data$Def_ind == 0,]$Def_ind <- "not defaulted"
train.data[train.data$Def_ind == 1,]$Def_ind <- "defaulted"
train.data$Def_ind <- as.factor(train.data$Def_ind)

train.data[train.data$ind_XYZ == 0,]$ind_XYZ <- "No Account"
train.data[train.data$ind_XYZ == 1,]$ind_XYZ <- "Has Account"
train.data$ind_XYZ <- as.factor(train.data$ind_XYZ)

train.data$rep_education <- as.factor(train.data$rep_education)

train.data$credit_age <- as.numeric(train.data$credit_age)
train.data$credit_age_good_account <- as.numeric(train.data$credit_age_good_account)
train.data$credit_card_age <- as.numeric(train.data$credit_card_age)
train.data$num_acc_30d_past_due_12_months <- as.numeric(train.data$num_acc_30d_past_due_12_months) 
train.data$num_acc_30d_past_due_6_months <- as.numeric(train.data$num_acc_30d_past_due_6_month)
train.data$num_mortgage_currently_past_due <-  as.numeric(train.data$num_mortgage_currently_past_due)
train.data$num_inq_12_month <- as.numeric(train.data$num_inq_12_month)
train.data$num_card_inq_24_month <- as.numeric(train.data$num_card_inq_24_month)
train.data$num_card_12_month <- as.numeric(train.data$num_card_12_month)
train.data$num_auto_.36_month <- as.numeric(train.data$num_auto_.36_month)
train.data$rep_income <- as.numeric(train.data$rep_income)

str(train.data)



# write.csv(train.data,"assdf.csv", row.names = FALSE) 


# Loading the test data

test.data <-  read.csv("Test_R-197119_Candidate Attach.csv", 
                       header = TRUE, na.strings = c("", "NA"))

str(test.data)

test.data[test.data$Def_ind == 0,]$Def_ind <- "not defaulted"
test.data[test.data$Def_ind == 1,]$Def_ind <- "defaulted"
test.data$Def_ind <- as.factor(test.data$Def_ind)

test.data[test.data$ind_XYZ == 0,]$ind_XYZ <- "No Account"
test.data[test.data$ind_XYZ == 1,]$ind_XYZ <- "Has Account"
test.data$ind_XYZ <- as.factor(test.data$ind_XYZ)

test.data$rep_education <- as.factor(test.data$rep_education)

test.data$credit_age <- as.numeric(test.data$credit_age)
test.data$credit_age_good_account <- as.numeric(test.data$credit_age_good_account)
test.data$credit_card_age <- as.numeric(test.data$credit_card_age)
test.data$num_acc_30d_past_due_12_months <- as.numeric(test.data$num_acc_30d_past_due_12_months) 
test.data$num_acc_30d_past_due_6_months <- as.numeric(test.data$num_acc_30d_past_due_6_month)
test.data$num_mortgage_currently_past_due <-  as.numeric(test.data$num_mortgage_currently_past_due)
test.data$num_inq_12_month <- as.numeric(test.data$num_inq_12_month)
test.data$num_card_inq_24_month <- as.numeric(test.data$num_card_inq_24_month)
test.data$num_card_12_month <- as.numeric(test.data$num_card_12_month)
test.data$num_auto_.36_month <- as.numeric(test.data$num_auto_.36_month)
test.data$rep_income <- as.numeric(test.data$rep_income)

str(test.data)

train.data.imputed <- rfImpute(Def_ind ~., data = train.data, iter = 6)

table(train.data.imputed$Def_ind)
prop.table(table(train.data.imputed$Def_ind)) 

train.data.imputed <- SMOTE(Def_ind ~ ., data = train.data.imputed, 
                            perc.over = 300, perc.under = 300)

table(train.data.imputed$Def_ind)
prop.table(table(train.data.imputed$Def_ind)) 


random.forest <- randomForest(Def_ind ~., 
                          data = train.data.imputed, proximity =TRUE)
random.forest


# Error plot to determine optimal number of trees


oob.error.data <- data.frame(
  Trees=rep(1:nrow(random.forest$err.rate), times=3),
  Type=rep(c("OOB", "defaulted", "not defaulted"), each=nrow(random.forest$err.rate)),
  Error=c(random.forest$err.rate[,"OOB"], 
          random.forest$err.rate[,"defaulted"], 
          random.forest$err.rate[,"not defaulted"]))

ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type))

random.forest.1000 <- randomForest(Def_ind ~., 
                              data = train.data.imputed, ntree= 1000, 
                              proximity =TRUE)
random.forest.1000



oob.error.data.1000 <- data.frame(
  Trees=rep(1:nrow(random.forest.1000$err.rate), times=3),
  Type=rep(c("OOB", "defaulted", "not defaulted"), each=nrow(random.forest.1000$err.rate)),
  Error=c(random.forest.1000$err.rate[,"OOB"], 
          random.forest.1000$err.rate[,"defaulted"], 
          random.forest.1000$err.rate[,"not defaulted"]))

ggplot(data=oob.error.data.1000, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type))


# m-try

oob.values <- vector(length=10)

for(i in 1:10) {
  temp.model <- randomForest(Def_ind ~ ., data=train.data.imputed, mtry=i, ntree=1000)
  oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate),1]
}
oob.values

## find the minimum error
min(oob.values)


random.forest.predict <- predict(random.forest, newdata = test.data)

confusionMatrix(data = random.forest.predict, test.data$Def_ind, mode = "everything")




