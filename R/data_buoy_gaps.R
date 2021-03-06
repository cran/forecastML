#' NOAA buoy weather data
#'
#' A dataset containing daily average sensor measurements of several environmental
#' conditions collected by 14 buoys in Lake Michigan from 2012 through 2018. This
#' dataset is identical to the \link{data_buoy} dataset except that there are gaps in
#' the daily sensor data. Running \code{fill_gaps()} on \code{data_buoy_gaps} will
#' produce \code{data_buoy}.
#'
#' @format A data.frame with 23,646 rows and 9 columns:
#' \describe{
#'   \item{date}{date}
#'   \item{wind_spd}{average daily wind speed in kts}
#'   \item{buoy_id}{the station ID for each buoy}
#'   \item{lat}{latitude}
#'   \item{lon}{longitude}
#'   \item{day}{day of year}
#'   \item{year}{calendar year}
#'   \item{air_temperature}{air temperature in degrees Fahrenheit}
#'   \item{sea_surface_temperature}{water temperature in degrees Fahrenheit}
#' }
#' @source \url{http://www.ndbc.noaa.gov/}
"data_buoy_gaps"
