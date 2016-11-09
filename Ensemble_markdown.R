
# Loan prediction challange:

The challange is to model loan applications and predict if they will go bad or not.

In the datasets, the first row contains the labels of the attributes. In the training data set, there is an additional final column (named Loan Status) which has the information on whether the loans will turn bad (0) or not (1).

Our task here is to split the given training dataset into train and test (i.e., hold-out sample) and build a model on the train dataset and test the model predicted results against the Loan Status column in the test data set. 

Labels of attributes is shown below. 

![](C:\\Users\\33093\\Desktop\\Presentation - Ensemble Methods\\Dictionary.png)


## Setting up the working directory in R 
## Load required R packages for this exercise 

```{r}

setwd("D:\\Kaggle\\Yodlee")

library(tree)
library(dplyr)
library(adabag)
library(glm2)

```

### Read and store the given training data set in R 

```{r, echo= FALSE}

loan_data <- read.csv("Training.csv", sep = ",", header  = T, stringsAsFactors = FALSE)

```

Since we have 'percentage symbols' and 'months' attached with the values in EMI column and Loan Duration column it is desirable to remove them and convert the character type varibles into factor variables. '%>%' is an operator used in dplyr package.

```{r, echo = FALSE}

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
```

# Feature Engineering 

We cannot use State as one the predictors in our model hence it is decided to compute and use the frequency of loan applications per State. Likewise the frequency of loans are computed by bank rating and used.   

```{r, echo = FALSE}

l1 <- loan_data %>%
  group_by(State) %>% 
  summarise(loans_by_state = n())

l2 <- loan_data %>%
  group_by(Bank_Rating) %>%
  summarise(loans_by_Rating = n())
  
loan_data <- loan_data %>%
  left_join(l1, by = 'State') %>%
  left_join(l2, by = 'Bank_Rating')

```

# Missing values

Missing values are noted in the following columns. 

  + No_of_Loans (Number of time the borrower has availed the loan from the bank)
  + Credit_Balance (Total Credit Revolving balance)
  + Percentage_Used_Credit (Percentage of Credit the Borrower is Using Relative to All 
                            Available Revolving Credit)
  + No_of_Loans_Other_Banks (Number of Times the Borrower has Availed Loan from All Banks)

```{r, echo = FALSE}

loan_data <- loan_data %>%
  mutate(loan_to_income = round(Loan_Amount/Annual_Income, 2),
         No_of_Loans = ifelse(is.na(No_of_Loans), mean(No_of_Loans, na.rm = TRUE),
         No_of_Loans),
         Credit_Balance = ifelse(is.na(Credit_Balance), mean(Credit_Balance, na.rm = TRUE),
         Credit_Balance),
         Percentage_Used_Credit = ifelse(is.na(Percentage_Used_Credit),
         mean(Percentage_Used_Credit, na.rm = TRUE), Percentage_Used_Credit),
         No_of_Loans_Other_banks = ifelse(is.na(No_of_Loans_Other_banks),
         mean(No_of_Loans_Other_banks, na.rm = TRUE), No_of_Loans_Other_banks))


```

### Exclude the variables like Loan_No (Loan Application Number) and other columns with date values

```{r, echo = FALSE}

loan_data <- loan_data[,-c(1, 2, 6, 9, 11, 13, 16, 17, 18, 25)]

```

# Cross validation samples - Split the data set into two portions (70% - Train; 30% - Test)

```{r, echo = FALSE}

set.seed(2)
sam.size <- floor(.7*nrow(loan_data))
train_ind <- sample(seq_len(nrow(loan_data)), size = sam.size)
loan_train <- loan_data[train_ind,]
loan_test <- loan_data[-train_ind,]

loan_Status_test <- loan_test[,17] 
loan_test <- loan_test[,-17]

```

# Logistic Regression Model 

First two GLM models were studied and based on the results GLM3 (glm.fit3) is finalised.

```{r, echo = FALSE}

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

```

## GLM model was fit for test observations

```{r, echo = FALSE}

glm.probs <- predict(glm.fit3, newdata = loan_test, type = 'response')
glm.pred <- ifelse(glm.probs>0.5, 1, 0)

```

# Confusion matrix and Model Accuracy

```{r, echo = FALSE}

table(glm.pred, loan_Status_test)
mean(glm.pred == loan_Status_test)

```

# Decision Tree Model 
## Decision tree is built using tree package in R

```{r, echo = FALSE}

train.tree <- tree(Loan_Status ~., data = loan_train)
summary(train.tree)
```

Decision Tree:

```{r, echo = FALSE}
plot(train.tree);text(train.tree, pretty = 0)

```

## Decision tree model is fit fot test observations and computed Confusion matrix.

```{r, echo = FALSE}

tree.pred <- predict(train.tree, newdata = loan_test, type = 'class')
table(tree.pred,loan_Status_test)
mean(tree.pred == loan_Status_test)

```

# Tree Pruning using Cross Validation method
### Prune the tree to select number of trees to end up with.

```{r,echo = FALSE}

cv.trees = cv.tree(train.tree, FUN = prune.misclass)
plot(cv.trees)

```

## Application of Repruned tree

```{r, echo = FALSE}

prune.tree <- prune.misclass(train.tree, best = 4)
plot(prune.tree);text(prune.tree, pretty =0)

tree.pred <- predict(prune.tree, newdata = loan_test, type = 'class')
table(tree.pred,loan_test$Loan_Status)
mean(tree.pred == loan_Status_test)

```       
  
# ADABOOST.M1 Algorithm

```{r, echo = FALSE}

adaboost.model <- boosting(Loan_Status ~., data = loan_train, boos = TRUE, coeflearn = 'Breiman')

adaboost.model$terms
adaboost.model$weights

```

## Predict the boosting model on test observations and test errors are evaluated.

```{r, echo = FALSE}

boost.pred <- predict.boosting(adaboost.model, newdata = loan_test)

predClass <- boost.pred$class

````

## Confusion matrix - **Adapative** **Boosting** **Technique**

table(predClass, loan_Status_test)

````


