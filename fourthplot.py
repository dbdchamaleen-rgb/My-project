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
mean_temp = df_daily['temperature'].mean()
std_temp = df_daily['temperature'].std()

df_daily['Extreme_High'] = df_daily['temperature'] > (mean_temp + std_temp)
df_daily['Extreme_Low'] = df_daily['temperature'] < (mean_temp - std_temp)

high_days = df_daily['Extreme_High'].sum()
low_days = df_daily['Extreme_Low'].sum()

print(f"Average Temperature: {mean_temp:.2f} °C")
print(f"Standard Deviation: {std_temp:.2f}")
print(f"Extreme High Days: {high_days}")
print(f"Extreme Low Days: {low_days}")



# %%
plt.figure(figsize=(12,6))
plt.plot(df_daily.index, df_daily['temperature'], color='b', label='Temperature',linewidth=2)
plt.scatter(df_daily.loc[df_daily['Extreme_High']].index,
            df_daily.loc[df_daily['Extreme_High'], 'temperature'], color='red', label='Extreme Highs')
plt.scatter(df_daily.loc[df_daily['Extreme_Low']].index,
            df_daily.loc[df_daily['Extreme_Low'], 'temperature'], color='k', label='Extreme Lows')
plt.axhline(mean_temp, color='green', linestyle='--', label='Mean Temperature',linewidth=3)
plt.title("Extreme Temperature Days")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.savefig('fourth.pdf')
plt.show()



