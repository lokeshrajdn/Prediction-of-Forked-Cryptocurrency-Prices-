{
library(keras)
library(tensorflow)
library(ggplot2)
library(MLmetrics)
library(lubridate)
}

######Getting and setting Directory#####
getwd()
setwd("D:/RProject/Datasets")

###Reading  CSV File###
BCH<-read.csv("bch19.csv")

####Timestamp converstion
BCH$Date<-as.POSIXct((BCH$Date)/1000,tz="UTC",origin = "1970-01-01")

#sorting the Date in Order###
BCH <- BCH[order(BCH$Date, decreasing = F),]
bch<-BCH
###########################Selecting relevant columns#####
Series_bch<- bch[,5]
################Data visualization######################
ggplot(bch, aes(Date, Close)) + geom_line(color = "#00AFBB")+ ylab("Closing price")+ xlab("Years")+ggtitle("Bitcoin Cash")

###################Dealing with detected data deviations####To get rid of non-stationaryness diff was implemented#################

diffed_bch = diff(Series_bch, differences = 1)

lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}

supervised_bch = lag_transform(diffed_bch, 1)

head(supervised_bch)

#########################Sampling and scaling, Partioning 80% train and 20 % test############################

N_bch = nrow(supervised_bch)
n_bch = round(N_bch *0.8, digits = 0)
train_bch = supervised_bch[1:n_bch, ]
test_bch= supervised_bch[(n_bch+1):N_bch,  ]

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

Scaled_bch = scale_data(train_bch, test_bch, c(-1, 1))

y_train_bch = Scaled_bch$scaled_train[, 2]
x_train_bch= Scaled_bch$scaled_train[, 1]
y_test_bch = Scaled_bch$scaled_test[, 2]
x_test_bch = Scaled_bch$scaled_test[, 1]

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

#########Define the model########
dim(x_train_bch) <- c(length(x_train_bch), 1, 1)
X_shape2_bch= dim(x_train_bch)[2]
X_shape3_bch = dim(x_train_bch)[3]
batch_size = 1                
units = 1   

model_bch <- keras_model_sequential() 

model_bch%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2_bch, X_shape3_bch), stateful= TRUE)%>%
  layer_dense(units = 1)

############Compile the model################
model_bch %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

###Model Summary###
summary(model_bch)
###################

#################Fit the model ############
Epochs = 50   

for(i in 1:Epochs ){
  model_bch %>% fit(x_train_bch, y_train_bch, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model_bch %>% reset_states()
}

##################Make Predictions#############
L_bch = length(x_test_bch)
scaler_bch = Scaled_bch$scaler
predictions_bch = numeric(L_bch)
for(i in 1:L_bch){
  X_bch= x_test_bch[i]
  dim(X_bch) = c(1,1,1)
  yhat = model_bch %>% predict(X_bch, batch_size=batch_size)
  # invert scaling
  yhat_bch = invert_scaling(yhat, scaler_bch,  c(-1, 1))
  # invert differencing
  yhat_bch  = yhat_bch + Series_bch[(n_bch+i)]
  # store
  predictions_bch[i] <- yhat_bch
}
###RMSE#########
Series_bch[n_bch:N_bch]
length(Series_bch[(n_bch+1):N_bch])
length(predictions_bch)
RMSE(predictions_bch,Series_bch[(n_bch+1):N_bch])

####Plot  Forecast values vs Actual Prices###############
data_bch<-bch[(n_bch+1):N_bch,]
data_bch$Date <- as.Date(data_bch$Date)
merge_bch<-cbind(data_bch,predictions_bch)
merge_bch$Close
predictions_bch
as.POSIXct((merge_bch$Date)/1000,tz="UTC",origin = "1970-01-01")
actual_pred <- data.frame(merge_bch$Date,merge_bch$Close,predictions_bch)
colnames(actual_pred) <- c("Date","Price","Predicted")
length(merge_bch$Close)

ggplot()+
  geom_line(data=actual_pred,aes(y=Price,x= Date,colour="darkblue"),size=1.1)+
  geom_line(data=actual_pred,aes(y=Predicted,x= Date,colour="red"),size=1.1) +
  scale_color_discrete(name = "Bitcoin Cash", labels = c("Predicted", "Actual"))




