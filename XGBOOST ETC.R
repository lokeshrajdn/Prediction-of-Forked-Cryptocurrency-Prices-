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
setwd("C:/Users/ASUS/Desktop/Project/Datasets")

#####Read CSV File Bitcoin Cash#######
ETC<-read.csv("etc19.csv")

########TimeStamp Conversion#####
ETC$Date<-as.POSIXct((ETC$Date)/1000,tz="UTC",origin = "1970-01-01")

##### Statstical view#########
summary(ETC)
str(ETC)

###Converting the data into time series format####
etc<- ETC$Close
ETC[,"Close"]->etc
ETC[,"Close"]->etc_y
ETC[,"Date"]->etc_ds
etc_pr<-data.frame(etc_ds,etc_y)
head(etc_pr)
str(etc_pr)
names(etc_pr)[1]<-"ds"
names(etc_pr)[2]<-"y"

########### Partitioning## Using first 80% as Train Data and last 20% as train data for prediction#########
data2 <-data.frame(ds = etc_pr$ds[-1]
                   ,week.day = as.factor(weekdays(etc_pr$ds[-1]))
                   ,Hour = as.factor(hour(etc_pr$ds[-1]))
                   ,hour.sin = sin(2 * pi * hour(etc_pr$ds[-1]) / 24)
                   ,hour.cos = cos(2 * pi * hour(etc_pr$ds[-1]) / 24)
                   ,log.change = diff(log(etc_pr$y)))

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


##########################################XGBoost Model Implementation###############

#####################HYPERTUNING#######################################################
Mtry<- (10:20)
Mtrees<-seq(500,800,50)
Mtree_depth<-(8:10)
Mygrid <- expand.grid(list(Mtry,Mtrees,Mtree_depth))
#View(gboost)

for( i in 1:nrow(Mygrid)){
  xgb_etc1 <-boost_tree(mode = "regression",mtry = Mygrid$Var1[i],trees = Mygrid$Var2[i],min_n = 3,
                        tree_depth = Mygrid$Var3[i],learn_rate = 0.01,
                        loss_reduction = 0.01) %>% set_engine(engine = "xgboost") %>%  fit.model_spec(y~., data = scaled.train.data[ , -1])
  
  
  predxx <- predict(xgb_etc1, new_data=scaled.test.data[,-1])
  
  predxx.converted <- etc_pr$y[(which(etc_pr$ds == scaled.test.data$ds[1]) -1) : (nrow(etc_pr) - 1)] * exp((predxx * std.train + mean.train)) 
  
  Actual <- etc_pr$y[which(etc_pr$ds == scaled.test.data$ds[1]) : nrow(etc_pr)]
  
  print(paste0(i, " = ", sqrt(mean((predxx.converted$.pred - Actual)^2))))
  
}
###############Selecting the Feature Importance for the model#######
imp <- xgb.importance(model=xgb_etc1$fit)

###########################################Replacing the grid values from the least RMSE Generated############################
xgb_etc <-boost_tree(mode = "regression",mtry = 20,trees = 500,min_n = 3,
                     tree_depth = 8,learn_rate = 0.01,
                     loss_reduction = 0.01) %>% set_engine(engine = "xgboost") %>%  fit.model_spec(y~., data = scaled.train.data[ ,which(names(scaled.train.data) %in% c(imp$Feature[1:13], "y"))])
                                                                                                   
predxx <- predict(xgb_etc, new_data=scaled.test.data[,-1])

predxx.converted <- etc_pr$y[(which(etc_pr$ds == scaled.test.data$ds[1]) -1) : (nrow(etc_pr) - 1)] * exp((predxx * std.train + mean.train)) 

Actual <- etc_pr$y[which(etc_pr$ds == scaled.test.data$ds[1]) : nrow(etc_pr)]

sqrt(mean((predxx.converted$.pred - Actual)^2))

##Defining Feature Importance selecting top 13 variables with the new grid values#########
imp <- xgb.importance(model=xgb_etc$fit)
xgb.ggplot.importance(imp)

########Summary########
summary(predxx)
summary(xgb_etc)


#####################################Plotting##############################
par(mfrow=c(1,1))

plot(Actual_xgboost,type = "l",col = "red", xlab = "Date", ylab = 'Close', 
     main = "Ethereum Classic")

lines(pred_xgboost, type = "l", col = "blue")




legend(
  "topright", 
  lty=c(1,1), 
  col=c("red", "blue"), 
  legend = c("Real", "Predicted")
)


