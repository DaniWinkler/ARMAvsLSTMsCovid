source("ARIMA_RNN_Main.R")

#Train Validation Test Split

#Germany
train_germany_cases <- germany_cases[1:(nrow(germany_cases) - 14),]
validation_germany_cases <- germany_cases[(nrow(train_germany_cases) + 1): (nrow(germany_cases) - 7),]
test_germany_cases <- germany_cases[(nrow(validation_germany_cases) + nrow(train_germany_cases) + 1) : nrow(germany_cases), ]


#Combine all tibbles to one tibble
germany_complete <- bind_rows(
  train_germany_cases %>% add_column(key = "train"),
  validation_germany_cases %>% add_column(key = "validation"),
  test_germany_cases%>% add_column(key = "test")) %>% 
  as_tbl_time(index = time)

#Scaling
germany_scaled <- scale(germany_complete$lag_2)
germany_complete_scaled <- na.omit(tibble(time = germany_complete$time,
                                          key = germany_complete$key,
                                          lag_2 = germany_scaled[,1]))


#center history for inverting after modelling
center_history_germany <- attributes(germany_scaled)[[2]]
scale_history_germany  <- attributes(germany_scaled)[[3]]

#modelling lstm: time series at lag 1 = input for LSTM

#Training Data Set: scaled
lag_train_germany <- germany_complete_scaled %>%
  mutate(value_lag = lag(lag_2, n = 1)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "train")

x_train_vec_germany <- lag_train_germany$value_lag
x_train_arr_germany <- array(data = x_train_vec_germany, dim = c(length(x_train_vec_germany), 1, 1))

y_train_vec_germany <- lag_train_germany$lag_2
y_train_arr_germany  <- array(data = y_train_vec_germany, dim = c(length(y_train_vec_germany), 1))

#Training Data Set: unscaled
lag_train_germany_unscaled <- germany_complete %>%
  mutate(value_lag = lag(lag_2, n = 1)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "train")

x_train_vec_germany_unscaled <- lag_train_germany_unscaled$value_lag
x_train_arr_germany_unscaled <- array(data = x_train_vec_germany_unscaled, dim = c(length(x_train_vec_germany_unscaled), 1, 1))

y_train_vec_germany_unscaled <- lag_train_germany_unscaled$lag_2
y_train_arr_germany_unscaled  <- array(data = y_train_vec_germany_unscaled, dim = c(length(y_train_vec_germany_unscaled), 1))

# Validation Set:scaled
lag_validation_germany <- germany_complete_scaled %>%
  mutate(value_lag = lag(lag_2, n = 1)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "validation")

x_validation_vec_germany <- lag_validation_germany$value_lag
x_validation_arr_germany <- array(data = x_validation_vec_germany, dim = c(length(x_validation_vec_germany), 1, 1))

y_validation_vec_germany <- lag_validation_germany$lag_2
y_validation_arr_germany  <- array(data = y_validation_vec_germany, dim = c(length(y_validation_vec_germany), 1))

# Validation Set:unscaled
lag_validation_germany_unscaled <- germany_complete%>%
  mutate(value_lag = lag(lag_2, n = 1)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "validation")

x_validation_vec_germany_unscaled <- lag_validation_germany_unscaled$value_lag
x_validation_arr_germany_unscaled <- array(data = x_validation_vec_germany_unscaled, dim = c(length(x_validation_vec_germany_unscaled), 1, 1))

y_validation_vec_germany_unscaled <- lag_validation_germany_unscaled$lag_2
y_validation_arr_germany_unscaled  <- array(data = x_validation_vec_germany_unscaled, dim = c(length(x_validation_vec_germany_unscaled), 1))


# Testing Set: scaled
lag_test_germany <- germany_complete_scaled %>%
  mutate(value_lag = lag(lag_2, n = 1)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "test")

x_test_vec_germany <- lag_test_germany$value_lag
x_test_arr_germany <- array(data = x_test_vec_germany, dim = c(length(x_test_vec_germany), 1, 1))

y_test_vec_germany <- lag_test_germany$lag_2
y_test_arr_germany  <- array(data = y_test_vec_germany, dim = c(length(y_test_vec_germany), 1))

# Testing Set: unscaled
lag_test_germany_unscaled <- germany_complete %>%
  mutate(value_lag = lag(lag_2, n = 1)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "test")

x_test_vec_germany_unscaled  <- lag_test_germany_unscaled$value_lag
x_test_arr_germany_unscaled  <- array(data = x_test_vec_germany_unscaled , dim = c(length(x_test_vec_germany_unscaled ), 1, 1))

y_test_vec_germany_unscaled  <- lag_test_germany_unscaled$lag_2
y_test_arr_germany_unscaled   <- array(data = y_test_vec_germany_unscaled , dim = c(length(y_test_vec_germany_unscaled ), 1))

#Train and Validation Combined:scaled
x_train_validation_arr_germany <- array(c(x_train_arr_germany, x_validation_arr_germany), dim = c((length(x_train_arr_germany)+
                                                                                                     length(x_validation_arr_germany)),1,1))

y_train_validation_arr_germany <- array(c(y_train_arr_germany,y_validation_arr_germany), dim = c((length(x_train_arr_germany)+length(x_validation_arr_germany)),1))

#Hyperparameter tuner 
#10 epochs
tuner_10_germany = RandomSearch(
  build_model,
  objective = 'val_mae',
  max_trials = 50,
  seed = 345,
  executions_per_trial = 1,
  directory = normalizePath("C:/"),
  project_name = "germany_models_10_ep")


tuner_10_germany %>% fit_tuner(x_train_arr_germany,y_train_arr_germany,
                               epochs = 10, 
                               validation_data = list(x_validation_arr_germany,y_validation_arr_germany),
                               batch_size = 32)


#Plot tuner_10
plot_tuner(tuner_10_germany)


#100 epochs
tuner_100_germany = RandomSearch(
  build_model,
  objective = 'val_mae',
  max_trials = 50,
  seed = 345,
  executions_per_trial = 1,
  directory = normalizePath("C:/"),
  project_name = "germany_models_100_ep")



tuner_100_germany %>% fit_tuner(x_train_arr_germany,y_train_arr_germany,
                                epochs = 100, 
                                validation_data = list(x_validation_arr_germany,y_validation_arr_germany),
                                batch_size = 32)


#Plot tuner_100
plot_tuner(tuner_100_germany)

#300 epochs
tuner_300_germany = RandomSearch(
  build_model,
  objective = 'val_mae',
  max_trials = 50,
  seed = 345,
  executions_per_trial = 1,
  directory = normalizePath("C:/"),
  project_name = "germany_models_300_ep")



tuner_300_germany %>% fit_tuner(x_train_arr_germany,y_train_arr_germany,
                                epochs = 300, 
                                validation_data = list(x_validation_arr_germany,y_validation_arr_germany),
                                batch_size = 32)


#Plot tuner_300
plot_tuner(tuner_300_germany)

#Increasing the epoch size to 100 has a high impact on mae
#Increasing the epoch size to 300 has a low impact on mae
#-> epochs: 100

#get best model
tuner_100_germany$results_summary()
#70 Neurons
#Learning Rate 0.01
#Dropout 0.1

#ARMA Models
models_df <- parameterdf %>%
  mutate(models = future_map(.x = order, .y = NULL,
                             .f = ~possibly(arima, otherwise = NULL)(x = y_train_vec_germany_unscaled,
                                                                     order = .x)))
#add forecasts
models_df <- models_df %>%
  mutate(forecast = map(models, ~possibly(forecast::forecast, otherwise = NULL)(., h = 7))) %>%
  mutate(point_forecast = map(forecast, ~.$`mean`))%>%
  mutate(true_value = rerun(nrow(parameterdf), y_validation_vec_germany_unscaled)) %>%
  mutate(mae = map2_dbl(point_forecast, true_value,
                        ~(mean(abs((.x - .y))))))

best_arima <- models_df %>% filter(mae == min(mae))
best_arima$order
#ARIMA(0,0,3)

#Test both models
#LSTM
lstm_germany <- keras_model_sequential()

lstm_germany %>%
  layer_lstm(units            = 70, 
             batch_size = 32,
             return_sequences = TRUE, 
             stateful         = FALSE,
             kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 345)) %>% 
  layer_dropout(0.1, seed = 345) %>% 
  layer_lstm(units            = 70, 
             return_sequences = FALSE, 
             stateful         = FALSE,
             kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 345)) %>% 
  layer_dropout(0.1, seed = 345) %>% 
  layer_dense(units = 1,
              kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 345))

lstm_germany %>% 
  compile(loss = 'mae', optimizer= tf$keras$optimizers$Adam(learning_rate = 0.01))

for (epoch in 1:100) {
  lstm_germany %>% fit(x    = x_train_validation_arr_germany,
                       y          = y_train_validation_arr_germany, 
                       verbose    = 1, 
                       epochs = 100,
                       batch_size = 32,
                       shuffle    = FALSE)
  
  lstm_germany %>% reset_states()
  cat("Epoch: ", epoch)
  
}

# Make Predictions LSTM
prediction_lstm_germany <- lstm_germany %>% 
  predict(x_test_arr_germany, batch_size = 32) %>%
  .[,1] 
prediction_lstm_germany_rescaled <- (prediction_lstm_germany * scale_history_germany + center_history_germany)

#MAE LSTM
mean(abs(prediction_lstm_germany_rescaled- y_test_arr_germany_unscaled))

#ARIMA
arima_germany <- arima(c(y_train_arr_germany_unscaled,y_validation_arr_germany_unscaled), order = c(0,0,3))

#Predictions
predictions_arima_germany <- forecast::forecast(arima_germany, h = 7)

#MAE ARIMA
mean(abs(predictions_arima_germany$mean- y_test_arr_germany_unscaled))

                                                                                                    