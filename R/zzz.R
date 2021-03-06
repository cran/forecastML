# nocov start
.onLoad <- function(...) {
  requireNamespace("dplyr")
}
#------------------------------------------------------------------------------
# Function in lagged_df.R. for create_lagged_df(method = "multi_outcome").
# The output is an nrow(data) by length(horizons) data.frame of outcome values.
forecastML_create_multi_outcome <- function(data, outcome_name, horizons, groups, outcome_levels) {

  outcome_indices <- purrr::map(1:nrow(data), function(x) {x + horizons})

  outcome_indices <- tibble::tibble(outcome_indices)

  outcome_indices <- dplyr::bind_cols(data[, groups, drop = FALSE], outcome_indices)

  outcome_indices$index <- 1:nrow(outcome_indices)

  outcome_indices <- outcome_indices %>%
    dplyr::group_by_at(dplyr::vars(!!groups)) %>%
    dplyr::mutate("max_group_index" = max(.data$index, na.rm = TRUE)) %>%
    dplyr::group_by(.data$index) %>%
    dplyr::mutate("outcome_indices" = list(ifelse(unlist(outcome_indices) > .data$max_group_index, NA, unlist(outcome_indices))))

  outcome_indices <- outcome_indices$outcome_indices

  data_outcomes <- lapply(outcome_indices, function(i) {

    if (all(is.na(i))) {

      data.frame(matrix(rep(NA, length(horizons)), nrow = 1))

    } else {

      data.frame(matrix(data[i, outcome_name, drop = TRUE], nrow = 1))
    }
  })

  data_outcomes <- suppressWarnings(dplyr::bind_rows(data_outcomes))

  if (!is.null(outcome_levels)) {
    data_outcomes[] <- lapply(data_outcomes, function(x) {factor(x, levels = outcome_levels)})
  }

  names(data_outcomes) <- paste0(outcome_name, "_", horizons)

  data_outcomes <- data_outcomes[, 1:length(horizons) , drop = FALSE]

  return(data_outcomes)
}
#------------------------------------------------------------------------------
# Error functions in return_error.R.
# For all error function args: x = 'residual', y = 'actual', and z = 'prediction'.
forecastML_mae <- function(x, ...) {
  error_var <- base::mean(base::abs(x), na.rm = TRUE)
  error_var <- if (is.infinite(error_var) || is.nan(error_var)) {NA} else {error_var}
}

forecastML_mape <- function(x, y, ...) {
  error_var <- base::mean(base::abs(x) / base::abs(y), na.rm = TRUE) * 100
  error_var <- if (is.infinite(error_var) || is.nan(error_var)) {NA} else {error_var}
}

forecastML_mdape <- function(x, y, ...) {
  error_var <- stats::median(base::abs(x) / base::abs(y), na.rm = TRUE) * 100
  error_var <- if (is.infinite(error_var) || is.nan(error_var)) {NA} else {error_var}
}

forecastML_smape <- function(x, y, z, ...) {
  error_var <- base::mean(2 * base::abs(x) / (base::abs(y) + base::abs(z)), na.rm = TRUE) * 100
  error_var <- if (is.infinite(error_var) || is.nan(error_var)) {NA} else {error_var}
}

forecastML_rmse <- function(x, ...) {
  error_var <- base::sqrt(base::mean(x^2, na.rm = TRUE))
  error_var <- if (is.infinite(error_var) || is.nan(error_var)) {NA} else {error_var}
}

# From the M5 competition. This metric is currently defined in return_error.R.
forecastML_rmsse <- function(...) {
  return(NA)
}
#------------------------------------------------------------------------------
# Function for ggplot2 faceting in train_model.R, return_error.R, and combine_forecasts.R. The input is (1) a formula
# with any of 'horizon', 'model', 'group', or '.' and (2) a string identifying the grouping columns,
# if any, in the input dataset. The output is a list containing (1) a formula where any
# groups in the modeling dataset are substituted for "~ group" and (2) facet names
# for use with ggplot2 geom objects. Plot aesthetics like color and group are adjusted automatically
# based on the facet names which helps avoid double encoding plot data.
forecastML_facet_plot <- function(facet, groups) {

  facet_names <- all.vars(facet)

  if (isTRUE(any(facet_names %in% "."))) {
    facet_names <- facet_names[!facet_names %in% "."]
  }

  if (!is.null(facet) && !all(facet_names %in% c("horizon", "model", "group"))) {
    stop("One or more of the plot facets is not in 'horizon', 'model', or 'group'.")
  }

  # Adjust the formula, substituting the group names from the data into the 'facet' input formula.
  if ("group" %in% facet_names) {

    rhs <- try(labels(stats::terms(facet)), silent = TRUE)

    if (methods::is(rhs, "try-error")) {
      rhs <- "."
    }

    lhs <- facet_names[!facet_names %in% rhs]

    if (length(groups) == 1) {

      lhs[lhs %in% "group"] <- groups
      rhs[rhs %in% "group"] <- groups

    } else {

      if (lhs %in% "group") {
        lhs <- c(lhs, groups)
        lhs <- lhs[!lhs %in% "group"]
      }

      if (rhs %in% "group") {
        rhs <- c(rhs, groups)
        rhs <- rhs[!rhs %in% "group"]
      }
    }

    facet <- as.formula(paste(paste(lhs, collapse = "+"), "~", paste(rhs, collapse = "+")))
    facet_names <- c(facet_names[!facet_names %in% "group"], groups)
  }
  return(list(facet, facet_names))
}
#------------------------------------------------------------------------------
# Used in lagged_df.R in create_lagged_df() for removing feature-specific feature lag indices when
# they do not support direct forecasting to a given forecast horizon.
forecastML_filter_lookback_control <- function(lookback_control, horizons, groups, group_cols,
                                               static_feature_cols, dynamic_feature_cols) {

  if (length(horizons) == 1) {  # A single-horizon, non-nested lookback_control of feature lags.

    lookback_control <- lapply(seq_along(lookback_control), function(i) {

      if (is.null(groups)) {  # Single time series

        if (i %in% dynamic_feature_cols) {

          lookback_control[[i]] <- 0

        } else {

          lookback_control[[i]][lookback_control[[i]] >= horizons]
        }

      } else {  # Multiple time series

        if (i %in% c(group_cols, static_feature_cols, dynamic_feature_cols)) {

          lookback_control[[i]] <- 0

        } else {

          if (!is.null(lookback_control[[i]])) {

            lookback_control[[i]] <- lookback_control[[i]][lookback_control[[i]] >= horizons]
          }
        }
        lookback_control[[i]]
      }
    })  # Impossible lags for lagged features have been removed.

  } else if (length(horizons) > 1) {  # A multiple-horizon, nested lookback_control of feature lags.

    lookback_control <- lapply(seq_along(lookback_control), function(i) {
      lapply(seq_along(lookback_control[[i]]), function(j) {

        if (is.null(groups)) {  # Single time series.

          if (j %in% dynamic_feature_cols) {

            lookback_control[[i]][[j]] <- 0

          } else {

            lookback_control[[i]][[j]][lookback_control[[i]][[j]] >= horizons[i]]
          }

        } else {  # Multiple time series.

          if (j %in% c(group_cols, static_feature_cols, dynamic_feature_cols)) {

            lookback_control[[i]][[j]] <- 0

          } else {

            if (!is.null(lookback_control[[i]][[j]])) {

              lookback_control[[i]][[j]] <- lookback_control[[i]][[j]][lookback_control[[i]][[j]] >= horizons[i]]
            }
          }
          lookback_control[[i]][[j]]
        }
      })
    })  # Impossible lags for lagged features have been removed.
  }  # Impossible lags in 'lookback_control' have been removed.
}
#------------------------------------------------------------------------------
# nocov end
