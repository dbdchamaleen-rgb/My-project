# My-project

This Python code processes a time series dataset containing temperature, humidity, and noise measurements.
It converts the time column to a datetime format, resamples the data to daily averages, fills missing values,
and visualizes the daily average temperature trend.The plot below shows the daily average temperature trend over time.

second.py  defines and tests a moving_average() function that calculates a simple moving average for a given pandas Series using a specified window size. After verifying the function with a small unit test, it applies a 5-day moving average to the temperature column in df_daily. Finally, it plots both the daily temperature and the moving average on the same graph and saves the plot as second.pdf

fourthplot.py code visualizes the data using a line plot, the blue line shows the temperature trend over time, red dots highlight extreme high days, and black dots mark extreme low days, while a green dashed line indicates the mean temperature.

The third.py code mostly focus about  a linear trend analysis using least-squares regression for each variable. The results (slope, R², and p-values) quantify the strength and direction of long-term changes.Several plots are generated for visualization such as Time-series plots showing daily average values for temperature, humidity, and noise and
Trendline overlays (in dashed green) showing the fitted linear regression for each variable.

The seventh.py shows the multi-panel figure of daily temperature, humidity, and noise levels with 5-day moving averages (red) highlighting trends. Mean values and ±1\sigma thresholds mark normal ranges, while black points indicate extreme deviations, clearly illustrating variability and anomalies in each variable.

The ninth.py  first calculates the mean, standard deviation, and upper/lower thresholds, then plots a histogram of daily temperatures with a fitted KDE curve to show the overall distribution.

The eight.py file visualizes how the number of extreme days changes with different threshold values 𝑘 for three variables,temperature, humidity, and noise. For each variable, it counts how many days exceed ±𝑘 standard deviations from the mean.
