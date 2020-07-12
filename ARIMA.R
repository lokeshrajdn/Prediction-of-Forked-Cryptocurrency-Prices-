{
  library(tseries)
library(dplyr)
library(forecast)
library(caret)
library(car)
library(tidymodels)
library(Rcpp)
library(timeSeries)
library(tseries)
library(timeSeries)
library(timeDate)
}

######setting Directory#####
getwd()
setwd("D:/RProject/Datasets")

######Read CSV File Bitcoin Cash & Ethereum Classic#######
BCH<-read.csv("bch19.csv")
ETC<-read.csv("etc19.csv")

###View Dataset###
view(BCH)
View(ETC)

########TimeStamp Conversion#####
BCH$Date<-as.POSIXct((BCH$Date)/1000,tz="UTC",origin = "1970-01-01")
ETC$Date<-as.POSIXct((ETC$Date)/1000,tz="UTC",origin = "1970-01-01")

##### summary and Structure of Datasets#########
summary(BCH)
summary(ETC)
str(BCH)
str(ETC)


###Converting the data into time series format for Bitcoin Cash & Ethereum Classic####

bch<- BCH$Close
BCH[,"Close"]->bch
BCH[,"Close"]->bch_y
BCH[,"Date"]->bch_ds
bch_pr<-data.frame(bch_ds,bch_y)
head(bch_pr)
str(bch_pr)
names(bch_pr)[1]<-"ds"
names(bch_pr)[2]<-"y"

#############Ethereum Classic###
etc<- ETC$Close
ETC[,"Close"]->etc
ETC[,"Close"]->etc_y
ETC[,"Date"]->etc_ds
etc_pr<-data.frame(etc_ds,etc_y)
head(etc_pr)
str(etc_pr)
names(etc_pr)[1]<-"ds"
names(etc_pr)[2]<-"y"

##conversion of Time Series
bch_ts <- ts(bch_pr$y)
etc_ts <- ts(etc_pr$y)

######To Check Data Frame is converted to time series##
class(bch_ts)
class(etc_ts)

##Performing Stationarity Test##
adf.test(bch_ts)
adf.test(etc_ts)

###Auto-Correlation Check##
acf(bch_ts)
acf(etc_ts)
pacf(bch)
pacf(etc)

####Taking differences#
diff(bch,lag = 1,differences = 1)->bchb1
diff(etc,lag =1,differences = 1)->etcb2

###Testing for autocorrelation for diff value###
acf(bchb1)
acf(etcb2)

##Performing Stationarity Test##
adf.test(bchb1)
adf.test(etcb2)

as.ts(bchb1)->kk
as.ts(etcb2)->kk1

adf.test(kk)
adf.test(kk1)

# To Check whether Timeseries Converted#
head(bch_ts)
head(etc_ts)

########### Partitioning## Using first 80% as Train Data and last 20% as train data for prediction#########
total_rows <- length(bch_ts)
total_rows1 <- length(etc_ts)

indexes <- round(total_rows*80/100) 
indexes1 <- round(total_rows1*80/100)

train_data<-bch_ts[1:indexes]
train_data1<-etc_ts[1:indexes1]

len_train <- length(train_data)
len_train1 <- length(train_data1)

test_data<-bch_ts[(1+indexes):(total_rows)]
test_data1<-etc_ts[(1+indexes1):(total_rows1)]

len_test <- length(test_data)
len_test1 <- length(test_data1)

head(train_data)
head(train_data1)
##############


###### MODELS####

#### ARIMA Model (Bitcoin Cash)

fit_arima <- auto.arima(bch_pr$y)

Predictbch <- Arima(model = fit_arima, y = bch_pr$y )

######RMSE Metrics###
RMSE_bch<-sqrt(mean(Predictbch$residuals[(1+indexes):(total_rows)]^2)) 
RMSE_bch
plot(Predictbch$residuals,main = "Bitcoin Cash")

#### ARIMA Model (Ethrereum Classic)########
fit_arima1 <- auto.arima(etc_pr$y)
Predictetc <- Arima(model = fit_arima1, y = etc_pr$y )

######RMSE Metrics###
RMSE_etc<-sqrt(mean(Predictetc$residuals[(1+indexes1):(total_rows1)]^2))
RMSE_etc
plot(Predictetc$residuals,main="Ethereum Classic")
