#FIRST: Make session reproducible
set.seed(345)
library(tensorflow)
reticulate::use_condaenv(condaenv = "tensorflow", required = TRUE)
library(keras)
tf$compat$v1$set_random_seed(345)
session_conf = tf$compat$v1$ConfigProto(intra_op_parallelism_threads= 1L, inter_op_parallelism_threads= 1L)
sess <- tf$compat$v1$Session(graph=tf$compat$v1$get_default_graph(), config=session_conf)
K <- tf$compat$v1$keras$backend
K$set_session(sess)

#Load Packages
library(kerastuneR)
library(dplyr)
library(stats)
library(forecast)
library(janitor)
library(readr)
library(tidyr)
library(ggplot2)
library(lubridate)
library(rsample)
library(tidyverse)
library(glue)
library(forcats)
library(timetk)
library(tidyquant)
library(tibbletime)
library(aTSA)
library(furrr)
library(tsibble)
library(ggthemes)
library(gridExtra)

#read data
all_cases <- read_csv("time_series_covid19_confirmed_global.csv")

#data preprocessing
all_cases <- all_cases %>% select(-Lat, -Long)
all_cases_long <- all_cases %>% pivot_longer(3:ncol(all_cases), names_to = "time", values_to = "cases")
all_cases_long$time <- mdy(all_cases_long$time)

#clean column names
all_cases_long <- all_cases_long %>% clean_names()

#select 3 European countries
european_countries <- all_cases_long %>% filter(country_region %in% c("Sweden", "Germany", "Italy"))

#differences of cases
european_countries <- european_countries %>% group_by(country_region) %>% mutate(log_cases = log(cases), lag_cases = c(NA,diff(cases)),
                                                                                 lag_2 = c(NA, NA, diff(cases, differences = 2)))
european_countries <- european_countries %>% ungroup()
european_countries_log <- european_countries %>% filter(!is.infinite(log_cases))

germany_cases <- european_countries %>% filter(country_region == "Germany")
italy_cases <- european_countries %>% filter(country_region == "Italy")
sweden_cases <- european_countries %>% filter(country_region == "Sweden")

#plot time series
cases_plot <- ggplot(data = european_countries, aes(x = time, y = cases, colour = country_region))+ geom_line()+
  xlab("")+ylab("Cases")+ 
  theme_economist_white(base_size = 11, base_family = "sans",
                        gray_bg = TRUE, horizontal = TRUE)+
  theme(legend.title=element_blank(),
        legend.text = element_text(size = 16))+
  scale_x_date(date_breaks = "1 month")+
  theme(axis.text = element_text(size=16),axis.title= element_text(size = 16))


#plot first differences of time series
cases_first_diff_plot <- ggplot(data = european_countries, aes(x = time, y = lag_cases, colour = country_region))+ geom_line()+
  xlab("")+ylab("Cases first Differences")+ 
  theme_economist_white(base_size = 11, base_family = "sans",
                        gray_bg = TRUE, horizontal = TRUE)+
  theme(legend.title=element_blank(),
        legend.text = element_text(size = 16))+
  scale_x_date(date_breaks = "1 month")+
  theme(axis.text = element_text(size = 16),axis.title= element_text(size = 16))

#plot second differences of time series
cases_second_diff_plot <- ggplot(data = european_countries, aes(x = time, y = lag_2, colour = country_region))+ geom_line()+
  xlab("")+ylab("Cases second Differences")+ 
  theme_economist_white(base_size = 16, base_family = "sans",
                        gray_bg = TRUE, horizontal = TRUE)+
  theme(legend.title=element_blank(),
        legend.text = element_text(size = 16))+
  scale_x_date(date_breaks = "1 month")+
  theme(axis.text = element_text(size = 16),axis.title= element_text(size = 16))

#geometric growth -> no stationarity -> log transformation

grid.arrange(cases_plot, cases_first_diff_plot, cases_second_diff_plot)

#Integration order
ndiffs(germany_cases$cases, max.d = 5, test = "kpss")
ndiffs(italy_cases$cases, max.d = 5, test = "kpss")
ndiffs(sweden_cases$cases, max.d = 5, test = "kpss")

#Autocorrelation Functions
#Germany
acf_germany <- ggAcf(na.omit(germany_cases%>% select(lag_2)), lag.max = 50)+ 
  theme_economist_white(base_size = 16, base_family = "sans",
                        gray_bg = TRUE, horizontal = TRUE)+
  ggtitle("Autocorrelation Function: second Differences Cases Germany")

#Italy
acf_italy <- ggAcf(na.omit(italy_cases%>% select(lag_2)), lag.max = 50)+ 
  theme_economist_white(base_size = 16, base_family = "sans",
                        gray_bg = TRUE, horizontal = TRUE)+
  ggtitle("Autocorrelation Function: second Differences Cases Italy")

#Sweden
acf_sweden <- ggAcf(na.omit(sweden_cases%>% select(lag_2)), lag.max = 50)+ 
  theme_economist_white(base_size = 16, base_family = "sans",
                        gray_bg = TRUE, horizontal = TRUE)+
  ggtitle("Autocorrelation Function: second Differences Cases Sweden")

grid.arrange(acf_germany, acf_italy, acf_sweden)

#Train Validation Test Split

                                         
#Fit LSTM
build_model = function(hp){
model <- keras_model_sequential()

model %>%
  layer_lstm(units            = hp$Int('units',
                                        min_value=10,
                                        max_value=100,
                                        step=10),
             return_sequences = TRUE,
             batch_size = 32,
             kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 345)) %>% 
  layer_dropout(hp$Float("dropout", min_value = 0, max_value = 0.5, step = 0.1), seed = 345) %>% 
  layer_lstm(units            = hp$Int('units',
                                       min_value=10,
                                       max_value=100,
                                       step=10), 
             return_sequences = FALSE,
             kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 345)) %>% 
  layer_dropout(hp$Float("dropout", min_value = 0, max_value = 0.5, step = 0.1), seed = 345) %>%  
  layer_dense(units = 1,
              kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 345)) %>% 
  compile(
    optimizer= tf$keras$optimizers$Adam(
      hp$Choice('learning_rate',
                values=c(1e-2, 1e-3, 1e-4,1e-5))),loss = 'binary_crossentropy',
    metrics = "mae")
return(model)}



#ARIMA Grid Search
parameter_list  <- list("P" = seq(0, 3),
                    "D" = 0,
                    "Q" = seq(0, 3)) %>%
  cross() %>%
  map(lift(c))

parameterdf <- tibble("order" = parameter_list)




