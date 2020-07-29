source("ARIMA_RNN_Main.R")

#Train Validation Test Split

#sweden
train_sweden_cases <- sweden_cases[1:(nrow(sweden_cases) - 14),]
validation_sweden_cases <- sweden_cases[(nrow(train_sweden_cases) + 1): (nrow(sweden_cases) - 7),]
test_sweden_cases <- sweden_cases[(nrow(validation_sweden_cases) + nrow(train_sweden_cases) + 1) : nrow(sweden_cases), ]


#Combine all tibbles to one tibble
sweden_complete <- bind_rows(
  train_sweden_cases %>% add_column(key = "train"),
  validation_sweden_cases %>% add_column(key = "validation"),
  test_sweden_cases%>% add_column(key = "test")) %>% 
  as_tbl_time(index = time)

#Scaling
sweden_scaled <- scale(sweden_complete$lag_2)
sweden_complete_scaled <- na.omit(tibble(time = sweden_complete$time,
                                        key = sweden_complete$key,
                                        lag_2 = sweden_scaled[,1]))


#center history for inverting after modelling
center_history_sweden <- attributes(sweden_scaled)[[2]]
scale_history_sweden  <- attributes(sweden_scaled)[[3]]

#modelling lstm: time series at lag 1 = input for LSTM

#Training Data Set: scaled
lag_train_sweden <- sweden_complete_scaled %>%
  mutate(value_lag = lag(lag_2, n = 1)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "train")

x_train_vec_sweden <- lag_train_sweden$value_lag
x_train_arr_sweden <- array(data = x_train_vec_sweden, dim = c(length(x_train_vec_sweden), 1, 1))

y_train_vec_sweden <- lag_train_sweden$lag_2
y_train_arr_sweden  <- array(data = y_train_vec_sweden, dim = c(length(y_train_vec_sweden), 1))

#Training Data Set: unscaled
lag_train_sweden_unscaled <- sweden_complete %>%
  mutate(value_lag = lag(lag_2, n = 1)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "train")

x_train_vec_sweden_unscaled <- lag_train_sweden_unscaled$value_lag
x_train_arr_sweden_unscaled <- array(data = x_train_vec_sweden_unscaled, dim = c(length(x_train_vec_sweden_unscaled), 1, 1))

y_train_vec_sweden_unscaled <- lag_train_sweden_unscaled$lag_2
y_train_arr_sweden_unscaled  <- array(data = y_train_vec_sweden_unscaled, dim = c(length(y_train_vec_sweden_unscaled), 1))

# Validation Set:scaled
lag_validation_sweden <- sweden_complete_scaled %>%
  mutate(value_lag = lag(lag_2, n = 1)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "validation")

x_validation_vec_sweden <- lag_validation_sweden$value_lag
x_validation_arr_sweden <- array(data = x_validation_vec_sweden, dim = c(length(x_validation_vec_sweden), 1, 1))

y_validation_vec_sweden <- lag_validation_sweden$lag_2
y_validation_arr_sweden  <- array(data = y_validation_vec_sweden, dim = c(length(y_validation_vec_sweden), 1))

# Validation Set:unscaled
lag_validation_sweden_unscaled <- sweden_complete%>%
  mutate(value_lag = lag(lag_2, n = 1)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "validation")

x_validation_vec_sweden_unscaled <- lag_validation_sweden_unscaled$value_lag
x_validation_arr_sweden_unscaled <- array(data = x_validation_vec_sweden_unscaled, dim = c(length(x_validation_vec_sweden_unscaled), 1, 1))

y_validation_vec_sweden_unscaled <- lag_validation_sweden_unscaled$lag_2
y_validation_arr_sweden_unscaled  <- array(data = x_validation_vec_sweden_unscaled, dim = c(length(x_validation_vec_sweden_unscaled), 1))


# Testing Set: scaled
lag_test_sweden <- sweden_complete_scaled %>%
  mutate(value_lag = lag(lag_2, n = 1)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "test")

x_test_vec_sweden <- lag_test_sweden$value_lag
x_test_arr_sweden <- array(data = x_test_vec_sweden, dim = c(length(x_test_vec_sweden), 1, 1))

y_test_vec_sweden <- lag_test_sweden$lag_2
y_test_arr_sweden  <- array(data = y_test_vec_sweden, dim = c(length(y_test_vec_sweden), 1))

# Testing Set: unscaled
lag_test_sweden_unscaled <- sweden_complete %>%
  mutate(value_lag = lag(lag_2, n = 1)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "test")

x_test_vec_sweden_unscaled  <- lag_test_sweden_unscaled $value_lag
x_test_arr_sweden_unscaled  <- array(data = x_test_vec_sweden_unscaled , dim = c(length(x_test_vec_sweden_unscaled ), 1, 1))

y_test_vec_sweden_unscaled  <- lag_test_sweden_unscaled $lag_2
y_test_arr_sweden_unscaled   <- array(data = y_test_vec_sweden_unscaled , dim = c(length(y_test_vec_sweden_unscaled ), 1))

#Train and Validation Combined:scaled
x_train_validation_arr_sweden <- array(c(x_train_arr_sweden, x_validation_arr_sweden), dim = c((length(x_train_arr_sweden)+
                                                                                               length(x_validation_arr_sweden)),1,1))

y_train_validation_arr_sweden <- array(c(y_train_arr_sweden,y_validation_arr_sweden), dim = c((length(x_train_arr_sweden)+
                                                                                              length(x_validation_arr_sweden)),1))

#Hyperparameter Tuner
#10 epochs
tuner_10_sweden = RandomSearch(
  build_model,
  objective = 'val_mae',
  max_trials = 50,
  seed = 345,
  executions_per_trial = 1,
  directory = normalizePath("C:/"),
  project_name = "sweden_models_10_ep_sweden")


tuner_10_sweden %>% fit_tuner(x_train_arr_sweden,y_train_arr_sweden,
                             epochs = 10, 
                             validation_data = list(x_validation_arr_sweden,y_validation_arr_sweden),
                             batch_size = 32)


#Plot tuner_10
plot_tuner(tuner_10_sweden)


#100 epochs
tuner_100_sweden = RandomSearch(
  build_model,
  objective = 'val_mae',
  max_trials = 50,
  seed = 345,
  executions_per_trial = 1,
  directory = normalizePath("C:/"),
  project_name = "sweden_models_100_ep_sweden")



tuner_100_sweden %>% fit_tuner(x_train_arr_sweden,y_train_arr_sweden,
                              epochs = 100, 
                              validation_data = list(x_validation_arr_sweden,y_validation_arr_sweden),
                              batch_size = 32)


#Plot tuner_100
plot_tuner(tuner_100_sweden)


#300 epochs
tuner_300_sweden = RandomSearch(
  build_model,
  objective = 'val_mae',
  max_trials = 50,
  seed = 345,
  executions_per_trial = 1,
  directory = normalizePath("C:/"),
  project_name = "sweden_models_300_ep_sweden")



tuner_300_sweden %>% fit_tuner(x_train_validation_arr_sweden,y_train_validation_arr_sweden,
                              epochs = 300, 
                              validation_data = list(x_validation_arr_sweden,y_validation_arr_sweden),
                              batch_size = 32)


#Plot tuner_300
plot_tuner(tuner_300_sweden)

#Increasing the epoch size to 100 has low impact on mae: epochs: 10

#get best model
tuner_10_sweden$results_summary()
#90 Neurons
#Learning Rate 0.01
#Dropout 0.4


#ARIMA Grid Search
models_df_sweden <- parameterdf %>%
  mutate(models = future_map(.x = order, .y = NULL,
                             .f = ~possibly(arima, otherwise = NULL)(x = y_train_vec_sweden_unscaled,
                                                                     order = .x,
                                                                     method = "ML")))

#add forecasts
models_df_sweden <- models_df_sweden %>%
  mutate(forecast = map(models, ~possibly(forecast::forecast, otherwise = NULL)(., h = 7))) %>%
  mutate(point_forecast = map(forecast, ~.$`mean`))%>%
  mutate(true_value = rerun(nrow(parameterdf), y_validation_vec_sweden_unscaled)) %>%
  mutate(mae = map2_dbl(point_forecast, true_value,
                        ~(mean(abs((.x - .y))))))

best_arima <- na.omit(models_df_sweden) %>% filter(mae == min(mae))
best_arima$order
#ARIMA(2,0,2)

#Test both models
#LSTM
lstm_sweden <- keras_model_sequential()

lstm_sweden %>%
  layer_lstm(units            = 90, 
             batch_size = 32,
             return_sequences = TRUE, 
             stateful         = FALSE,
             kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 345)) %>% 
  layer_dropout(0.4, seed = 345) %>% 
  layer_lstm(units            = 90, 
             return_sequences = FALSE, 
             stateful         = FALSE,
             kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 345)) %>% 
  layer_dropout(0.4, seed = 345) %>% 
  layer_dense(units = 1,
              kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 345))

lstm_sweden %>% 
  compile(loss = 'mae', optimizer= tf$keras$optimizers$Adam(learning_rate = 0.01))

for (epoch in 1:10) {
  lstm_sweden %>% fit(x          = x_train_validation_arr_sweden,
                     y          = y_train_validation_arr_sweden, 
                     verbose    = 1, 
                     epochs = 10,
                     batch_size = 32,
                     shuffle    = FALSE)
  
  lstm_sweden %>% reset_states()
  cat("Epoch: ", epoch)
  
}

# Make Predictions LSTM
prediction_lstm_sweden <- lstm_sweden %>% 
  predict(x_test_arr_sweden, batch_size = 32) %>%
  .[,1] 


#MAE LSTM
mean(abs((prediction_lstm_sweden * scale_history_sweden + center_history_sweden)- y_test_arr_sweden_unscaled))

#ARIMA
arima_sweden <- arima(c(y_train_arr_sweden_unscaled,y_validation_arr_sweden_unscaled), order = c(2,0,2))

#Predictions
predictions_arima_sweden <- forecast::forecast(arima_sweden, h = 7)

#MAE ARIMA
mean(abs(predictions_arima_sweden$mean- y_test_arr_sweden_unscaled))
