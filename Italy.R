source("ARIMA_RNN_Main.R")

#Train Validation Test Split

#italy
train_italy_cases <- italy_cases[1:(nrow(italy_cases) - 14),]
validation_italy_cases <- italy_cases[(nrow(train_italy_cases) + 1): (nrow(italy_cases) - 7),]
test_italy_cases <- italy_cases[(nrow(validation_italy_cases) + nrow(train_italy_cases) + 1) : nrow(italy_cases), ]


#Combine all tibbles to one tibble
italy_complete <- bind_rows(
  train_italy_cases %>% add_column(key = "train"),
  validation_italy_cases %>% add_column(key = "validation"),
  test_italy_cases%>% add_column(key = "test")) %>% 
  as_tbl_time(index = time)

#Scaling
italy_scaled <- scale(italy_complete$lag_2)
italy_complete_scaled <- na.omit(tibble(time = italy_complete$time,
                                          key = italy_complete$key,
                                          lag_2 = italy_scaled[,1]))


#center history for inverting after modelling
center_history_italy <- attributes(italy_scaled)[[2]]
scale_history_italy  <- attributes(italy_scaled)[[3]]

#modelling lstm: time series at lag 1 = input for LSTM

#Training Data Set: scaled
lag_train_italy <- italy_complete_scaled %>%
  mutate(value_lag = lag(lag_2, n = 1)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "train")

x_train_vec_italy <- lag_train_italy$value_lag
x_train_arr_italy <- array(data = x_train_vec_italy, dim = c(length(x_train_vec_italy), 1, 1))

y_train_vec_italy <- lag_train_italy$lag_2
y_train_arr_italy  <- array(data = y_train_vec_italy, dim = c(length(y_train_vec_italy), 1))

#Training Data Set: unscaled
lag_train_italy_unscaled <- italy_complete %>%
  mutate(value_lag = lag(lag_2, n = 1)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "train")

x_train_vec_italy_unscaled <- lag_train_italy_unscaled$value_lag
x_train_arr_italy_unscaled <- array(data = x_train_vec_italy_unscaled, dim = c(length(x_train_vec_italy_unscaled), 1, 1))

y_train_vec_italy_unscaled <- lag_train_italy_unscaled$lag_2
y_train_arr_italy_unscaled  <- array(data = y_train_vec_italy_unscaled, dim = c(length(y_train_vec_italy_unscaled), 1))

# Validation Set:scaled
lag_validation_italy <- italy_complete_scaled %>%
  mutate(value_lag = lag(lag_2, n = 1)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "validation")

x_validation_vec_italy <- lag_validation_italy$value_lag
x_validation_arr_italy <- array(data = x_validation_vec_italy, dim = c(length(x_validation_vec_italy), 1, 1))

y_validation_vec_italy <- lag_validation_italy$lag_2
y_validation_arr_italy  <- array(data = y_validation_vec_italy, dim = c(length(y_validation_vec_italy), 1))

# Validation Set:unscaled
lag_validation_italy_unscaled <- italy_complete%>%
  mutate(value_lag = lag(lag_2, n = 1)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "validation")

x_validation_vec_italy_unscaled <- lag_validation_italy_unscaled$value_lag
x_validation_arr_italy_unscaled <- array(data = x_validation_vec_italy_unscaled, dim = c(length(x_validation_vec_italy_unscaled), 1, 1))

y_validation_vec_italy_unscaled <- lag_validation_italy_unscaled$lag_2
y_validation_arr_italy_unscaled  <- array(data = x_validation_vec_italy_unscaled, dim = c(length(x_validation_vec_italy_unscaled), 1))


# Testing Set: scaled
lag_test_italy <- italy_complete_scaled %>%
  mutate(value_lag = lag(lag_2, n = 1)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "test")

x_test_vec_italy <- lag_test_italy$value_lag
x_test_arr_italy <- array(data = x_test_vec_italy, dim = c(length(x_test_vec_italy), 1, 1))

y_test_vec_italy <- lag_test_italy$lag_2
y_test_arr_italy  <- array(data = y_test_vec_italy, dim = c(length(y_test_vec_italy), 1))

# Testing Set: unscaled
lag_test_italy_unscaled <- italy_complete %>%
  mutate(value_lag = lag(lag_2, n = 1)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "test")

x_test_vec_italy_unscaled  <- lag_test_italy_unscaled $value_lag
x_test_arr_italy_unscaled  <- array(data = x_test_vec_italy_unscaled , dim = c(length(x_test_vec_italy_unscaled ), 1, 1))

y_test_vec_italy_unscaled  <- lag_test_italy_unscaled $lag_2
y_test_arr_italy_unscaled   <- array(data = y_test_vec_italy_unscaled , dim = c(length(y_test_vec_italy_unscaled ), 1))

#Train and Validation Combined:scaled
x_train_validation_arr_italy <- array(c(x_train_arr_italy, x_validation_arr_italy), dim = c((length(x_train_arr_italy)+
                                                                                                     length(x_validation_arr_italy)),1,1))

y_train_validation_arr_italy <- array(c(y_train_arr_italy,y_validation_arr_italy), dim = c((length(x_train_arr_italy)+
                                                                                                    length(x_validation_arr_italy)),1))

#Hyperparameter Tuner
#10 epochs
tuner_10_italy = RandomSearch(
  build_model,
  objective = 'val_mae',
  max_trials = 50,
  seed = 345,
  executions_per_trial = 1,
  directory = normalizePath("C:/"),
  project_name = "italy_models_10_ep_italy")


tuner_10_italy %>% fit_tuner(x_train_arr_italy,y_train_arr_italy,
                               epochs = 10, 
                               validation_data = list(x_validation_arr_italy,y_validation_arr_italy),
                               batch_size = 32)


#Plot tuner_10
plot_tuner(tuner_10_italy)


#100 epochs
tuner_100_italy = RandomSearch(
  build_model,
  objective = 'val_mae',
  max_trials = 50,
  seed = 345,
  executions_per_trial = 1,
  directory = normalizePath("C:/"),
  project_name = "italy_models_100_ep_italy")



tuner_100_italy %>% fit_tuner(x_train_arr_italy,y_train_arr_italy,
                                epochs = 100, 
                                validation_data = list(x_validation_arr_italy,y_validation_arr_italy),
                                batch_size = 32)


#Plot tuner_100
plot_tuner(tuner_100_italy)


#300 epochs
tuner_300_italy = RandomSearch(
  build_model,
  objective = 'val_mae',
  max_trials = 50,
  seed = 345,
  executions_per_trial = 1,
  directory = normalizePath("C:/"),
  project_name = "italy_models_300_ep_italy")



tuner_300_italy %>% fit_tuner(x_train_validation_arr_italy,y_train_validation_arr_italy,
                                epochs = 300, 
                                validation_data = list(x_validation_arr_italy,y_validation_arr_italy),
                                batch_size = 32)


#Plot tuner_300
plot_tuner(tuner_300_italy)

#Increasing the epoch size to 100 has low impact on mae: epochs: 10

#get best model
tuner_10_italy$results_summary()
#20 Neurons
#Learning Rate 0.001
#Dropout 0.5


#ARIMA Grid Search
models_df_italy <- parameterdf %>%
  mutate(models = future_map(.x = order, .y = NULL,
                             .f = ~possibly(arima, otherwise = NULL)(x = y_train_vec_italy_unscaled,
                                                                     order = .x)))
#add forecasts
models_df_italy <- models_df_italy %>%
  mutate(forecast = map(models, ~possibly(forecast::forecast, otherwise = NULL)(., h = 7))) %>%
  mutate(point_forecast = map(forecast, ~.$`mean`))%>%
  mutate(true_value = rerun(nrow(parameterdf), y_validation_vec_italy_unscaled)) %>%
  mutate(mae = map2_dbl(point_forecast, true_value,
                        ~(mean(abs((.x - .y))))))

best_arima <- models_df_italy %>% filter(mae == min(mae))
best_arima$order
#ARIMA(3,0,2)

#Test both models
#LSTM
lstm_italy <- keras_model_sequential()

lstm_italy %>%
  layer_lstm(units            = 20, 
             batch_size = 32,
             return_sequences = TRUE, 
             stateful         = FALSE,
             kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 345)) %>% 
  layer_dropout(0.5, seed = 345) %>% 
  layer_lstm(units            = 20, 
             return_sequences = FALSE, 
             stateful         = FALSE,
             kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 345)) %>% 
  layer_dropout(0.5, seed = 345) %>% 
  layer_dense(units = 1,
              kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 345))

lstm_italy %>% 
  compile(loss = 'mae', optimizer= tf$keras$optimizers$Adam(learning_rate = 0.001))

for (epoch in 1:10) {
  lstm_italy %>% fit(x          = x_train_validation_arr_italy,
                y          = y_train_validation_arr_italy, 
                verbose    = 1, 
                epochs = 10,
                batch_size = 32,
                shuffle    = FALSE)
  
  lstm_italy %>% reset_states()
  cat("Epoch: ", epoch)
  
}

# Make Predictions LSTM
prediction_lstm_italy <- lstm_italy %>% 
  predict(x_test_arr_italy, batch_size = 32) %>%
  .[,1] 


#MAE LSTM
mean(abs((prediction_lstm_italy * scale_history_italy + center_history_italy)- y_test_arr_italy_unscaled))

#ARIMA
arima_italy <- arima(c(y_train_arr_italy_unscaled,y_validation_arr_italy_unscaled), order = c(3,0,2))

#Predictions
predictions_arima_italy <- forecast::forecast(arima_italy, h = 7)

#MAE ARIMA
mean(abs(predictions_arima_italy$mean- y_test_arr_italy_unscaled))
