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
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 6))
plt.plot(df_daily.index, df_daily['temperature'], color='steelblue', linewidth=2)
plt.title("Daily Average Temperature Over Time", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.show()