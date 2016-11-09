
library(dplyr)
library(e1071)
setwd("D:\\Kaggle\\Yodlee")
train <- read.csv("Yodlee_Training.csv", sep = ",", header = T, stringsAsFactors = FALSE)

train <- train %>%
  rename(loanAppNo = Loan.Application.Number, loanAmt = Loan.Amount, 
         loanDur = Loan.Duration, intRate = Interest.Rate, emi = EMI,
         borrRate = Borrower.Rating.by.Bank, borrEmp = Borrower.s.Duration.of.Employment,
         borrHomeOwn = Home.Ownership.of.Borrower, borrInc = Annual.Income.of.Borrower,
         borrStatus = Borrower.s.Verification.Status, loanDate = Loan.Issue.Date,
         loanPurpose = Purpose.of.Loan, state = State, debtToInc = Debt.to.Income.Ratio..A.ratio.calculated.based.on.Borrower.s.monthly.debt.repayments.to.self.reported.monthly.income.,
         borrDelLast2 = Borrower.Delinquency.in.Last.Two.Years, 
         borrDateFirstLoan = Date.of.Borrower.s.First..Loan, 
         borrDelLast = Number.of.Months.Since.the.Borrower.s.Last.Delinquency, 
         borrCreditCardUpdate = The.Number.of.Months.Since.the.Borrower.s.Credit.Record..was.Updated,
         borrNumOfLoans = Number.of.Times.the.Borrower.has.Availed.Loan.from.the.Bank,
         borrNegCmt = Number.of.Negative.Comments.About.the.Borrower.in.Credit.History,
         creditRevBal =  Total.Credit.Revolving.Balance,
         percUseCredit =  Percentage.of.Credit.the.Borrower.is.Using.Relative.to.All.Available.Revolving.Credit,
         borrNumOfLoansAllBank = Number.of.Times.the.Borrower.has.Availed.Loan.from.All.Banks,
         borrLateFees = Late.Fees.Received.To.Date, 
         borrLastMthPayDate = Last.Month.Payment.was.Received,
         borrLastAmt = Last.Amount.Received.as.Payment, 
         loanStatus = Loan.Status)

train$intRate <- as.numeric(sub("%", "", as.factor(train$intRate))) /100
train$loanStatus <- as.factor(train$loanStatus)

## Naive bayes 
## Convert data types of variables

train$loanDur <- as.factor(train$loanDur)
train$borrRate <- as.factor(train$borrRate)
train$borrEmp <- as.factor(train$borrEmp)
train$borrHomeOwn <- as.factor(train$borrHomeOwn)
train$borrStatus <- as.factor(train$borrStatus)
train$loanPurpose <- as.factor(train$loanPurpose)
train$state <- as.factor(train$state)

nb.fit <- naiveBayes(loanStatus ~., data = train[,c(2:10, 12:15, 17:24, 26,27)])
pred <- predict(nb.fit, train)
table(train$loanStatus, pred)

