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

# Determining if there are blank or missing data

any_na(train.data)
n_miss(train.data)
prop_miss(train.data)
train.data %>% is.na() %>% colSums()

nrow(train.data[is.na(train.data$pct_card_over_50_uti) | is.na(train.data$rep_income),])
nrow(train.data)

train.data <- train.data[!(is.na(train.data$pct_card_over_50_uti)|is.na(train.data$rep_income) | 
                                     is.na(train.data$rep_income)),]

nrow(train.data)
any_na(train.data)
n_miss(train.data)


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

# Determining if there are blank or missing data in the testing data

any_na(test.data)
n_miss(test.data)
prop_miss(test.data)
test.data %>% is.na() %>% colSums()

nrow(test.data[is.na(test.data$pct_card_over_50_uti) |is.na(test.data$rep_education)| 
                 is.na(test.data$rep_income),])
nrow(test.data)
test.data <- test.data[!(is.na(test.data$pct_card_over_50_uti) 
                                 | is.na(test.data$rep_income) |
                                   is.na(test.data$rep_education)),]
nrow(test.data)
any_na(test.data)
n_miss(test.data)

str(test.data)

xtabs(~Def_ind + rep_education, data = train.data)
xtabs(~Def_ind + ind_XYZ, data = train.data)

# Dealing with the imbalanced Classification problems

table(train.data$Def_ind)
prop.table(table(train.data$Def_ind)) 

library(DMwR)
train.data <- SMOTE(Def_ind ~ ., data = train.data, perc.over = 300, perc.under = 200)

table(train.data$Def_ind)
prop.table(table(train.data$Def_ind))

# Running the Logit models

logit_simple <- glm(Def_ind ~ ind_XYZ, 
                    data = train.data, family = "binomial")
summary(logit_simple)

logit_fancy <- glm(Def_ind ~ ., data = train.data, family = "binomial")
summary(logit_fancy)


# McFadden's Pseudo R square

ll.null <- logit_fancy$null.deviance/-2

ll.proposed <- logit_fancy$deviance/-2

(ll.null - ll.proposed) / ll.null

1 - pchisq(2*(ll.proposed - ll.null), df=(length(logit_fancy$coefficients)-1))


ll.null.simple <- logit_simple$null.deviance/-2
ll.proposed.simple <- logit_simple$deviance/-2
(ll.null.simple - ll.proposed.simple) / ll.null.simple

1 - pchisq(2*(ll.proposed.simple - ll.null.simple), 
           df=(length(logit_simple$coefficients)-1))


# Fitting the test data into the logit model

res <- predict(logit_fancy, test.data, type= "response")
res

confmat_logit_fancy <- table(Actual_Value = test.data$Def_ind, 
                             Predicted_value = res > 0.5)
confmat_logit_fancy


# Accuracy of the logit model

(confmat_logit_fancy[[1,1]] + confmat_logit_fancy[[2,2]]) / sum(confmat_logit_fancy) 


logit_fancy.predict <- predict(logit_fancy, newdata = test.data, type = "response")

# Create confusion matrix
confusionMatrix(data = logit_fancy.predict, test.data$Def_ind, mode = "everything")

logit_fancy <- train(Def_ind ~ tot_balance + avg_bal_cards + credit_age + 
                  credit_age_good_account + credit_card_age + num_acc_30d_past_due_12_months
                + num_acc_30d_past_due_6_months + num_mortgage_currently_past_due +
                  tot_amount_currently_past_due + num_inq_12_month + num_card_inq_24_month
                + num_card_12_month + num_auto_.36_month + uti_open_card + pct_over_50_uti
                + uti_max_credit_line + pct_card_over_50_uti + ind_XYZ +
                  rep_income+ rep_education, data = train.data, method = "glm",
                family = binomial)
prediction <- predict(logit_fancy, newdata = test.data)
confusionMatrix(data = prediction, test.data$Def_ind)


