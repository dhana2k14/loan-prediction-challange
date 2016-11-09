
setwd("D:\\Kaggle\\Yodlee")

library(tree)
library(dplyr)
loan_data <- read.csv("Training.csv", sep = ",", header  = T, stringsAsFactors = FALSE)

loan_data <- loan_data %>%
  mutate(Loan_Duration = as.integer(sub('months', '', Loan_Duration)),
         Interest_Rate = as.integer(sub('%', '', Interest_Rate)),
         Bank_Rating = as.factor(Bank_Rating), 
         Employment = as.factor(Employment),
         Home_Ownership = as.factor(Home_Ownership),
         Verification_Status = as.factor(Verification_Status),
         Loan_Purpose = as.factor(Loan_Purpose),
         State = as.factor(State),
         Loan_Status = as.factor(Loan_Status)) 

l1 <- loan_data %>%
  group_by(State) %>% 
  summarise(loans_by_state = n())

l2 <- loan_data %>%
  group_by(Bank_Rating) %>%
  summarise(loans_by_Rating = n())
  
loan_data <- loan_data %>%
  left_join(l1, by = 'State') %>%
  left_join(l2, by = 'Bank_Rating')

loan_data <- loan_data %>%
  mutate(loan_to_income = round(Loan_Amount/Annual_Income, 2),
         No_of_Loans = ifelse(is.na(No_of_Loans), mean(No_of_Loans, na.rm = TRUE), No_of_Loans),
         Credit_Balance = ifelse(is.na(Credit_Balance), mean(Credit_Balance, na.rm = TRUE), Credit_Balance),
         Percentage_Used_Credit = ifelse(is.na(Percentage_Used_Credit), mean(Percentage_Used_Credit, na.rm = TRUE), Percentage_Used_Credit),
         No_of_Loans_Other_banks = ifelse(is.na(No_of_Loans_Other_banks), mean(No_of_Loans_Other_banks, na.rm = TRUE), No_of_Loans_Other_banks))


plot(loan_data$Annual_Income ~ loan_data$Loan_Amount, xlim = c(50000, 400000), ylim = c(50000, 1300000))

loan_data <- loan_data[,-c(1, 2, 6, 9, 11, 13, 16, 17, 18, 25)]

## Train & Test sampling
set.seed(2)
sam.size <- floor(.7*nrow(loan_data))
train_ind <- sample(seq_len(nrow(loan_data)), size = sam.size)
loan_train <- loan_data[train_ind,]
loan_test <- loan_data[-train_ind,]
loan_Status_test <- loan_test[,17] ## extract loan_status from test observations to cross validate
loan_test <- loan_test[,-17]


### Build logistic regression model using GLM
require(glm2)

# glm.fit1 <- glm(Loan_Status ~ EMI + Verification_Status + Negative_Comments + No_of_Loans_Other_banks + Late_Fees + Last_Amount +
#                loans_by_state + loan_to_income, data = loan_train, family = binomial)
# summary(glm.fit1)
# 
# glm.fit2 <- glm(Loan_Status ~ EMI +  Late_Fees + Last_Amount +
#                   loans_by_state + loan_to_income, data = loan_train, family = binomial)
# summary(glm.fit2)

glm.fit3 <- glm(Loan_Status ~ EMI +  Late_Fees + Last_Amount +
                  loans_by_state, data = loan_train, family = binomial)
summary(glm.fit3)

glm.probs <- predict(glm.fit3, newdata = loan_test, type = 'response')
glm.pred <- ifelse(glm.probs>0.5, 1, 0)
table(glm.pred, loan_Status_test)
mean(glm.pred == loan_Status_test)

## Build a Decision Tree model

train.tree <- tree(Loan_Status ~., data = loan_train)
summary(train.tree)
plot(train.tree);text(train.tree, pretty = 0)
  
tree.pred <- predict(train.tree, newdata = loan_test, type = 'class')
table(tree.pred,loan_Status_test)
mean(tree.pred == loan_Status_test)
(335+6041)/7200       

### Prune the tree to select number of trees to end up with.
cv.trees = cv.tree(train.tree, FUN = prune.misclass)
plot(cv.trees)

## apply the prune tree
prune.tree <- prune.misclass(train.tree, best = 4)
plot(prune.tree);text(prune.tree, pretty =0)

tree.pred <- predict(prune.tree, newdata = loan_test, type = 'class')
table(tree.pred,loan_test$Loan_Status)
(335+6041)/7200          
  

### Adaboost algorithm
library(ada)
boost.model <- ada(Loan_Status~., data = loan_train)
summary(boost.model)
plot(boost.model)
varplot(boost.model) 

boost.pred <- predict(boost.model, newdata = loan_test)

table(loan_test$Loan_Status, boost.pred)
mean(boost.pred == loan_Status_test)
(376+5990)/7200

### Boosting tress using adabag algorithm

library(adabag)
adaboost.model <- boosting(Loan_Status ~., data = loan_train, boos = TRUE, coeflearn = 'Breiman')

errorevol(adaboost.model, loan_test)

adaboost.pred <- predict.boosting(adaboost.model, newdata = loan_test)

adaboost.pred$confusion
adaboost.pred$error

table(adaboost.pred, loan_Status_test)


