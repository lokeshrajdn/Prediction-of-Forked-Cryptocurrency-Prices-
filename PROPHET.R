{
library(readxl)
library(tseries)
library(dplyr)
library(forecast)
library(prophet)
library(car)
library(tidymodels)
library(Rcpp)
}
######Getting and setting Directory#####
getwd()
setwd("D:/RProject/Datasets")

#####Read CSV File Bitcoin Cash#######
BCH<-read.csv("bch19.csv")
ETC<-read.csv("etc19.csv")


########TimeStamp Conversion#####
BCH$Date<-as.POSIXct((BCH$Date)/1000,tz="UTC",origin = "1970-01-01")
ETC$Date<-as.POSIXct((ETC$Date)/1000,tz="UTC",origin = "1970-01-01")


##### Statstical view#########
summary(BCH)
summary(ETC)
str(BCH)
str(ETC)


###Relevant Selection of Attributes for Bitcoin Cash & Ethererum Classic ####
bch<- BCH$Close
BCH[,"Close"]->bch
BCH[,"Close"]->bch_y
BCH[,"Date"]->bch_ds
bch_pr<-data.frame(bch_ds,bch_y)
head(bch_pr)
str(bch_pr)
names(bch_pr)[1]<-"ds"
names(bch_pr)[2]<-"y"


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
total_rows <- nrow(bch_pr)
total_rows1 <- nrow(etc_pr)

indexes <- round(total_rows*80/100) 
indexes1 <- round(total_rows1*80/100)

train_data<-bch_pr[1:indexes,]
train_data1<-etc_pr[1:indexes1,]

len_train <- nrow(train_data)
len_train1 <- nrow(train_data1)

test_data<-bch_pr[(1+indexes):(total_rows),]
test_data1<-etc_pr[(1+indexes1):(total_rows1),]

len_test <- nrow(test_data)
len_test1 <- nrow(test_data1)

head(train_data)
head(train_data1)

########################PROPHET MODEL BUILDING################
pro_bch <- prophet(train_data)
pro_etc <- prophet(train_data1)

####Prediction with the created model###############
pred_pro_bch<-predict(pro_bch,test_data)
pred_pro_etc<-predict(pro_etc,test_data1)

###RMSE Metrics##
RMSE_pro_bch<-sqrt(mean(test_data$y-pred_pro_bch$yhat)^2)
RMSE_pro_etc<-sqrt(mean(test_data1$y-pred_pro_etc$yhat)^2)

RMSE_pro_bch
RMSE_pro_etc

#### Forecast of Bitcoin Cash Using the PROPHET Model#################
future <- make_future_dataframe(pro_bch,periods = 300,freq = 3600)
tail(future)
forecast_pro_bch<-predict(pro_bch,future)
forecast_pro_bch$yhat

##Plot Bitcoin Cash##### 
plot(pro_bch,forecast_pro_bch)
####Interactive Plot for Bitcoin Cash###
dyplot.prophet(pro_bch,forecast_pro_bch)
##Component wise plot for Bitcoin Cash##
prophet_plot_components(pro_bch,forecast_pro_bch)

#### Forecast of Ethereum Classic Using the PROPHET Model#################
future_etc <- make_future_dataframe(pro_etc,periods = 300,freq = 3600)
tail(future_etc)
forecast_pro_etc<-predict(pro_etc,future_etc)
forecast_pro_etc$yhat

#Plot Ethereum Classic#
plot(pro_bch,forecast_pro_bch)
####Interactive Plot for Ethereum Classic###

dyplot.prophet(pro_etc,forecast_pro_etc)

##Component wise plot for Ethereum Classic##
prophet_plot_components(pro_etc,forecast_pro_etc)

