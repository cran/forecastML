## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(fig.width = 7.15, fig.height = 4)
knitr::opts_knit$set(fig.width = 7.15, fig.height = 4)

## ---- message = FALSE, warning = FALSE----------------------------------------
library(forecastML)
library(dplyr)
library(DT)
library(ggplot2)
library(xgboost)

data("data_buoy_gaps", package = "forecastML")

data_buoy_gaps$wind_spd <- cut(data_buoy_gaps$wind_spd, breaks = c(-1, 3, 5, 8, 10),
                               ordered_result = TRUE)  # Create the factor outcome.

DT::datatable(head(data_buoy_gaps), options = list(scrollX = TRUE))

## -----------------------------------------------------------------------------
data <- forecastML::fill_gaps(data_buoy_gaps, date_col = 1, frequency = '1 day', 
                              groups = 'buoy_id', static_features = c('lat', 'lon'))

print(list(paste0("The original dataset with gaps in data collection is ", nrow(data_buoy_gaps), " rows."), 
      paste0("The modified dataset with no gaps in data collection from fill_gaps() is ", nrow(data), " rows.")))

## -----------------------------------------------------------------------------
data$day <- lubridate::mday(data$date)
data$year <- lubridate::year(data$date)

## ---- message = FALSE, warning = FALSE----------------------------------------
p <- ggplot(data[!is.na(data$wind_spd), ], aes(x = date, y = 1, fill = wind_spd, color = wind_spd))
p <- p + geom_tile()
p <- p + facet_wrap(~ ordered(buoy_id), scales = "fixed")
p <- p + theme_bw() + theme(axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
  xlab(NULL) + ylab(NULL)
p

## -----------------------------------------------------------------------------
data$buoy_id <- as.numeric(factor(data$buoy_id))

## -----------------------------------------------------------------------------
outcome_col <- 1  # The column position of our 'wind_spd' outcome (after removing the 'date' column).

horizons <- c(1, 7, 30)  # Forecast 1, 1:7, and 1:30 days into the future.

lookback <- c(1:30, 360:370)  # Features from 1 to 30 days in the past and annually.

dates <- data$date  # Grouped time series forecasting requires dates.
data$date <- NULL  # Dates, however, don't need to be in the input data.

frequency <- "1 day"  # A string that works in base::seq(..., by = "frequency").

dynamic_features <- c("day", "year")  # Features that change through time but which will not be lagged.

groups <- "buoy_id"  # 1 forecast for each group or buoy.

static_features <- c("lat", "lon")  # Features that do not change through time.

## -----------------------------------------------------------------------------
type <- "train"  # Create a model-training dataset.

data_train <- forecastML::create_lagged_df(data, type = type, outcome_col = outcome_col,
                                           horizons = horizons, lookback = lookback,
                                           dates = dates, frequency = frequency,
                                           dynamic_features = dynamic_features,
                                           groups = groups, static_features = static_features, 
                                           use_future = FALSE)

DT::datatable(head(data_train$horizon_1), options = list(scrollX = TRUE))

## ---- message = FALSE, warning = FALSE----------------------------------------
p <- plot(data_train)  # plot.lagged_df() returns a ggplot object.
p <- p + geom_tile(NULL)  # Remove the gray border for a cleaner plot.
p

## ---- message = FALSE, warning = FALSE----------------------------------------
windows <- forecastML::create_windows(data_train, window_length = 0)

plot(windows, data_train)

## ---- message = FALSE, warning = FALSE----------------------------------------
plot(windows, data_train, group_filter = "buoy_id == 1") 

## -----------------------------------------------------------------------------
# The value of outcome_col can also be set in train_model() with train_model(outcome_col = 1).
model_function <- function(data, outcome_col = 1) {
  
  # xgboost cannot model factors directly so they'll be converted to numbers.
  data[] <- lapply(data, as.numeric)
  
  # xgboost cannot handle missing outcomes data so we'll remove this.
  data <- data[!is.na(data[, outcome_col]), ]
  
  data[, outcome_col] <- data[, outcome_col] - 1  # xgboost needs factors to start at 0.

  indices <- 1:nrow(data)
  
  set.seed(224)
  train_indices <- sample(1:nrow(data), ceiling(nrow(data) * .8), replace = FALSE)
  test_indices <- indices[!(indices %in% train_indices)]

  data_train <- xgboost::xgb.DMatrix(data = as.matrix(data[train_indices, 
                                                           -(outcome_col), drop = FALSE]),
                                     label = as.matrix(data[train_indices, 
                                                            outcome_col, drop = FALSE]))

  data_test <- xgboost::xgb.DMatrix(data = as.matrix(data[test_indices, 
                                                          -(outcome_col), drop = FALSE]),
                                    label = as.matrix(data[test_indices, 
                                                           outcome_col, drop = FALSE]))

  params <- list("objective" = "multi:softprob",
                 "eval_metric" = "mlogloss",
                 "num_class" = 4)  # Hard-coding the number of factor levels.

  watchlist <- list(train = data_train, test = data_test)
  
  set.seed(224)
  model <- xgboost::xgb.train(data = data_train, params = params, 
                              max.depth = 8, nthread = 2, nrounds = 30,
                              metrics = "rmse", verbose = 0, 
                              early_stopping_rounds = 5, 
                              watchlist = watchlist)

  return(model)
}

## -----------------------------------------------------------------------------
#future::plan(future::multiprocess)  # Multi-core or multi-session parallel training.

model_results <- forecastML::train_model(lagged_df = data_train,
                                         windows = windows,
                                         model_name = "xgboost",
                                         model_function = model_function, 
                                         use_future = FALSE)

## -----------------------------------------------------------------------------
summary(model_results$horizon_1$window_1$model)

## -----------------------------------------------------------------------------
# If 'model' is passed as a named list, the prediction model would be accessed with model$model or model["model"].
prediction_function_prob <- function(model, data_features) {
  
  # xgboost cannot model factors directly so they'll be converted to numbers.
  data_features[] <- lapply(data_features, as.numeric)
  
  x <- xgboost::xgb.DMatrix(data = as.matrix(data_features))
  data_pred <- data.frame("y_pred" = predict(model, x, reshape = TRUE))  # 'reshape' returns a wide data.frame.
  return(data_pred)
}

## -----------------------------------------------------------------------------
# We'll define a global variable with the factor levels.
factor_levels <- levels(data_buoy_gaps$wind_spd)

# If 'model' is passed as a named list, the prediction model would be accessed with model$model or model["model"].
prediction_function_level <- function(model, data_features) {
  
  # xgboost cannot model factors directly so they'll be converted to numbers.
  data_features[] <- lapply(data_features, as.numeric)
  
  x <- xgboost::xgb.DMatrix(data = as.matrix(data_features))
  data_pred <- data.frame("y_pred" = predict(model, x, reshape = TRUE))  # 'reshape' returns a wide data.frame.
  
  data_pred$y_pred <- apply(data_pred, 1, which.max)  # Find the column with the highest probability.
  data_pred$y_pred <- dplyr::recode(data_pred$y_pred, `1` = factor_levels[1], `2` = factor_levels[2], 
                                    `3` = factor_levels[3], `4` = factor_levels[4])

  data_pred$y_pred <- factor(data_pred$y_pred, levels = factor_levels, ordered = TRUE)

  data_pred <- data_pred[, "y_pred", drop = FALSE]
  return(data_pred)
}

## -----------------------------------------------------------------------------
data_pred_prob <- predict(model_results, prediction_function = list(prediction_function_prob), data = data_train)

data_pred_level <- predict(model_results, prediction_function = list(prediction_function_level), data = data_train)

## ---- message = FALSE, warning = FALSE----------------------------------------
plot(data_pred_prob, horizons = 7, group_filter = "buoy_id %in% c(1, 2)")

## ---- message = FALSE, warning = FALSE----------------------------------------
inspect_dates <- seq(as.Date("2018-01-01"), as.Date("2018-12-31"), by = "1 day")

plot(data_pred_level[data_pred_level$date_indices %in% inspect_dates, ], horizons = 7, group_filter = "buoy_id %in% c(1, 2)")

## ---- message = FALSE, warning = FALSE----------------------------------------
data_error <- forecastML::return_error(data_pred_level, metric = "mae")

plot(data_error, data_pred_level, type = "horizon", metric = "mae")
plot(data_error, data_pred_level, type = "global", metric = "mae")

## -----------------------------------------------------------------------------
type <- "forecast"  # Create a forecasting dataset for our predict() function.

data_forecast <- forecastML::create_lagged_df(data, type = type, outcome_col = outcome_col,
                                              horizons = horizons, lookback = lookback,
                                              dates = dates, frequency = frequency,
                                              dynamic_features = dynamic_features,
                                              groups = groups, static_features = static_features, 
                                              use_future = FALSE)

DT::datatable(head(data_forecast$horizon_1), options = list(scrollX = TRUE))

## -----------------------------------------------------------------------------
for (i in seq_along(data_forecast)) {
  data_forecast[[i]]$day <- lubridate::mday(data_forecast[[i]]$index)  # When dates are given, the 'index` is date-based.
  data_forecast[[i]]$year <- lubridate::year(data_forecast[[i]]$index)
}

## -----------------------------------------------------------------------------
data_forecasts_prob <- predict(model_results, prediction_function = list(prediction_function_prob), data = data_forecast)

data_forecasts_level <- predict(model_results, prediction_function = list(prediction_function_level), data = data_forecast)

## ---- message = FALSE, warning = FALSE----------------------------------------
plot(data_forecasts_prob, group_filter = "buoy_id %in% c(1, 2)")

## ---- message = FALSE, warning = FALSE----------------------------------------
plot(data_forecasts_level, group_filter = "buoy_id %in% c(1, 2)")

## ---- message = FALSE, warning = FALSE----------------------------------------
data_combined_prob <- forecastML::combine_forecasts(data_forecasts_prob)
data_combined_level <- forecastML::combine_forecasts(data_forecasts_level)

# Plot the final forecasts.
plot(data_combined_prob, group_filter = "buoy_id %in% c(1, 2)")
plot(data_combined_level, group_filter = "buoy_id %in% c(1, 2)")

