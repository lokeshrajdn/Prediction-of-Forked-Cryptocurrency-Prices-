{
library(keras)
library(tensorflow)
library(ggplot2)
library(MLmetrics)
library(xlsx)
library(lubridate)
}

######Getting and setting Directory#####
getwd()
setwd("D:/RProject/Datasets")

###Read CSV File###
ETC<-read.csv("etc19.csv")

##Date Conversion######
ETC$Date<-as.POSIXct(ETC$Date/1000,origin = "1970-01-01")

###Sorting Date####
ETC <- ETC[order(ETC$Date, decreasing = F),]
etc<-ETC
###################Selecting Relevant Variables########
Series_etc<-etc[,5]
################Data visualization######################
ggplot(etc, aes(Date, Close)) + geom_line(color = "#00AFBB")+ ylab("Closing price")+ xlab("Years")+ggtitle("Ethereum Classic")

###To get rid of non-stationaryness, the first differentities were taken for each of the time series.#####################

diffed_etc = diff(Series_etc, differences = 1)

lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}

supervised_etc = lag_transform(diffed_etc, 1)

####################Sampling and scaling, Partioning 80% train and 20 % test#######################

N_etc = nrow(supervised_etc)
n_etc = round(N_etc *0.8, digits = 0)
train_etc = supervised_etc[1:n_etc, ]
test_etc= supervised_etc[(n_etc+1):N_etc,  ]

scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((test - min(x) ) / (max(x) - min(x)  ))
  
  scaled_train = std_train *(fr_max -fr_min) + fr_min
  scaled_test = std_test *(fr_max -fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler= c(min =min(x), max = max(x))) )
}

###The LSTM sigmoid function, the range of which is .-1, 1.##

Scaled_etc = scale_data(train_etc, test_etc, c(-1, 1))

y_train_etc = Scaled_etc$scaled_train[, 2]
x_train_etc= Scaled_etc$scaled_train[, 1]
y_test_etc = Scaled_etc$scaled_test[, 2]
x_test_etc = Scaled_etc$scaled_test[, 1]

###Return the projected values to the original scale.###
invert_scaling = function(Scaled, scaler, feature_range = c(0, 1)){
  min = scaler[1]
  max = scaler[2]
  t = length(Scaled)
  mins = feature_range[1]
  maxs = feature_range[2]
  inverted_dfs = numeric(t)
  
  for( i in 1:t){
    X = (Scaled[i]- mins)/(maxs - mins)
    rawValues = X *(max - min) + min
    inverted_dfs[i] <- rawValues
  }
  return(inverted_dfs)
}

########Define the model###############################

dim(x_train_etc) <- c(length(x_train_etc), 1, 1)
X_shape2_etc= dim(x_train_etc)[2]
X_shape3_etc = dim(x_train_etc)[3]
batch_size = 1                
units = 1   

model_etc <- keras_model_sequential() 

model_etc%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2_etc, X_shape3_etc), stateful= TRUE)%>%
  layer_dense(units = 1)
######################################
##########Compile the model########################

model_etc %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)
summary(model_etc)
######################USing 50 Epoch for optimal value############
Epochs = 50   

for(i in 1:Epochs ){
  model_etc %>% fit(x_train_etc, y_train_etc, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model_etc %>% reset_states()
}

################################Prediction#############
L_etc = length(x_test_etc)
scaler_etc = Scaled_etc$scaler
predictions_etc = numeric(L_etc)
for(i in 1:L_etc){
  X_etc= x_test_etc[i]
  dim(X_etc) = c(1,1,1)
  yhat = model_etc %>% predict(X_etc, batch_size=batch_size)
  # invert scaling
  yhat_etc = invert_scaling(yhat, scaler_etc,  c(-1, 1))
  # invert differencing
  yhat_etc  = yhat_etc + Series_etc[(n_etc+i)]
  # store
  predictions_etc[i] <- yhat_etc
}
###RMSE#########
Series_etc[n_etc:N_etc]
length(Series_etc[(n_etc+1):N_etc])
length(predictions_etc)
RMSE(predictions_etc,Series_etc[(n_etc+1):N_etc])

#### Forecast values vs Actual Prices###############
data_etc<-etc[(n_etc+1):N_etc,]
data_etc$Date <- as.Date(data_etc$Date)
merge_etc<-cbind(data_etc,predictions_etc)
merge_etc$Close
predictions_etc
as.POSIXct((merge_etc$Date)/1000,tz="UTC",origin = "1970-01-01")
actual_pred <- data.frame(merge_etc$Date,merge_etc$Close,predictions_etc)
colnames(actual_pred) <- c("Date","Price","Predicted")
length(merge_etc$Close)


ggplot()+
  geom_line(data=actual_pred,aes(y=Price,x= Date,colour="darkblue"),size=1.1)+
  geom_line(data=actual_pred,aes(y=Predicted,x= Date,colour="red"),size=1.1) +
  scale_color_discrete(name = "Ethereum Classic", labels = c("Predicted", "Actual"))
