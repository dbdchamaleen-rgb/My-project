# My-project

This Python code processes a time series dataset containing temperature, humidity, and noise measurements.
It converts the time column to a datetime format, resamples the data to daily averages, fills missing values,
and visualizes the daily average temperature trend.The plot below shows the daily average temperature trend over time.

second.py  defines and tests a moving_average() function that calculates a simple moving average for a given pandas Series using a specified window size. After verifying the function with a small unit test, it applies a 5-day moving average to the temperature column in df_daily. Finally, it plots both the daily temperature and the moving average on the same graph and saves the plot as second.pdf

