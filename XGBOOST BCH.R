{
  library(dplyr)
  library(forecast)
  library(prophet)
  library(xgboost)
  library(caret)
  library(car)
  library(tidymodels)
  library(parsnip)
  library(Rcpp)
  library(lubridate)
  library(ggplot2)
}

######Getting and setting Directory#####
getwd()
setwd("D:/RProject/Datasets")

#####Read CSV File Bitcoin Cash#######
BCH<-read.csv("bch19.csv")

########TimeStamp Conversion#####
BCH$Date<-as.POSIXct((BCH$Date)/1000,tz="UTC",origin = "1970-01-01")

##### Statstical view#########
summary(BCH)
str(BCH)

###Relevant Selection of Attributes for Bitcoin Cash ####
bch<- BCH$Close
BCH[,"Close"]->bch
BCH[,"Close"]->bch_y
BCH[,"Date"]->bch_ds
bch_pr<-data.frame(bch_ds,bch_y)
head(bch_pr)
str(bch_pr)
names(bch_pr)[1]<-"ds"
names(bch_pr)[2]<-"y"

#########Adding multiple lags in the train data & Partitioning Using first 80% as Train Data and last 20% as test #########
data2 <-data.frame(ds = bch_pr$ds[-1]
                     ,week.day = as.factor(weekdays(bch_pr$ds[-1]))
                     ,Hour = as.factor(hour(bch_pr$ds[-1]))
                     ,hour.sin = sin(2 * pi * hour(bch_pr$ds[-1]) / 24)
                     ,hour.cos = cos(2 * pi * hour(bch_pr$ds[-1]) / 24)
                     ,log.change = diff(log(bch_pr$y)))

total_rows <- nrow(data2)

indexes <- round(total_rows*80/100)

train_data<-data2[1:indexes,]

len_train <- nrow(train_data)

mean.train <- mean(train_data$log.change)
std.train <- sd(train_data$log.change)
scaled.data <- data.frame(data2[,-6], y = scale(data2$log.change, mean.train, std.train))

scaled.data <- scaled.data %>% 
  mutate(lag1 = lag(scaled.data$y, 1)
        ,lag2 = lag(scaled.data$y, 2)
        ,lag3 = lag(scaled.data$y, 3)
        ,lag4 = lag(scaled.data$y, 4)
        ,lag5 = lag(scaled.data$y, 5)
        ,lag6 = lag(scaled.data$y, 6)
        ,lag7 = lag(scaled.data$y, 7)
        ,lag19 = lag(scaled.data$y, 19)
        ,lag26 = lag(scaled.data$y, 26)
        ,lag36 = lag(scaled.data$y, 36)
  ) %>% group_by(Hour) %>%
          mutate(lagh1 = lag(scaled.data$y, 1, order_by = Hour)
                 ,lagh2 = lag(scaled.data$y, 2, order_by = Hour)
          )

scaled.data$Hour.plus <- scaled.data$hour.sin +scaled.data$hour.cos
scaled.data <- na.omit(scaled.data)

scaled.data <- data.frame(ungroup(scaled.data))

str(scaled.data)
scaled.train.data <- scaled.data[1:indexes,]

scaled.test.data <-scaled.data[(1+indexes):nrow(scaled.data), ]

len_test <- nrow(scaled.test.data)


######################################Hyper tuning XGBoost Model for best fit using Grid Function###############

Mtry<- (10:20)
Mtrees<-seq(500,800,50)
Mtree_depth<-(8:10)
Mygrid <- expand.grid(list(Mtry,Mtrees,Mtree_depth))


for( i in 1:nrow(Mygrid)){
xgb_bch1 <-boost_tree(mode = "regression",mtry = Mygrid$Var1[i],trees = Mygrid$Var2[i],min_n = 3,
                     tree_depth = Mygrid$Var3[i],learn_rate = 0.01,
                     loss_reduction = 0.01) %>% set_engine(engine = "xgboost") %>%  fit.model_spec(y~., data = scaled.train.data[ , -1])



predxx <- predict(xgb_bch1, new_data=scaled.test.data[,-1])

predxx.converted <- bch_pr$y[(which(bch_pr$ds == scaled.test.data$ds[1]) -1) : (nrow(bch_pr) - 1)] * exp((predxx * std.train + mean.train)) 

Actual <- bch_pr$y[which(bch_pr$ds == scaled.test.data$ds[1]) : nrow(bch_pr)]

print(paste0(i, " = ", sqrt(mean((predxx.converted$.pred - Actual)^2))))

}

#Summary of the prediction and model#
summary(predxx)
summary(xgb_bch1)

####Creating Feature importance##
imp <- xgb.importance(model=xgb_bch1$fit)
############################################Replacing the grid values from the least RMSE Generated##############################################
xgb_bchfit <-boost_tree(mode = "regression",mtry = 18,trees = 500,min_n = 3,
                     tree_depth = 8,learn_rate = 0.01,
                     loss_reduction = 0.01) %>% set_engine(engine = "xgboost") %>%  fit.model_spec(y~., data = scaled.train.data[ ,which(names(scaled.train.data) %in% c(imp$Feature[1:13], "y"))])

######Prediction##
predxx <- predict(xgb_bchfit, new_data=scaled.test.data[,-1])

predxx.converted <- bch_pr$y[(which(bch_pr$ds == scaled.test.data$ds[1]) -1) : (nrow(bch_pr) - 1)] * exp((predxx * std.train + mean.train)) 

Actual <- bch_pr$y[which(bch_pr$ds == scaled.test.data$ds[1]) : nrow(bch_pr)]

####Creating Data Frame###
Actual <- data.frame(scaled.test.data$ds,Actual)

length(Actual)

sqrt(mean((predxx.converted$.pred - Actual)^2))

pred_xgboost <- data.frame(scaled.test.data$ds,predxx.converted$.pred)
Actual_xgboost <- Actual

################Defining Feature Importance selecting top 13 variables with the grid values#######################
imp <- xgb.importance(model=xgb_bchfit$fit)
xgb.ggplot.importance(imp)
##################################################################

#####Plot Actual vs Predicted#############
par(mfrow=c(1,1))

plot(Actual_xgboost,type = "l",col = "red", xlab = "Date", ylab = 'Close', 
     main = "Bitcoin Cash")

lines(pred_xgboost, type = "l", col = "blue")

legend(
  "topleft", 
  lty=c(1,1), 
  col=c("red", "blue"), 
  legend = c("Real", "Predicted")
)

