# %%
# Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings("ignore")



# %%
# Load the dataset
df = pd.read_csv("C:/Users/dbdch/OneDrive/Desktop/Project progress/mongo_export.csv")


# Convert 'time' column to datetime
df['time'] = pd.to_datetime(df['time'])

# Set 'time' as the index
df.set_index('time', inplace=True)

# Select only useful columns
df = df[['temperature', 'humidity', 'noice']]

# Resample data to daily averages
df_daily = df.resample('D').mean()

# Fill missing values
df_daily = df_daily.ffill()

# Quick check
print(df_daily.head())


# %%
def moving_average(series, window=7):
    """
    Compute simple moving average.
    """
    return series.rolling(window=window).mean()

# Unit Test
def test_moving_average():
    test_data = pd.Series([1, 2, 3, 4, 5])
    expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0])
    result = moving_average(test_data, window=3)
    assert np.allclose(result.dropna(), expected.dropna()), "Moving average function failed!"

test_moving_average()
print("moving_average() passed all tests!")

# Apply function
df_daily['7_day_avg'] = moving_average(df_daily['temperature'], 7)

# Plot with moving average
plt.figure(figsize=(12, 6))
plt.plot(df_daily.index, df_daily['temperature'], color='b', label='Daily Temperature', linewidth=2)
plt.plot(df_daily.index, df_daily['7_day_avg'], color='red', linewidth=3, label='5-Day Moving Average')
plt.title("Temperature with 5-Day Moving Average")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.savefig('second.pdf')
#plt.show()



