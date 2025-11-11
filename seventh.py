# %%
#  Imports Libraries
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import warnings
warnings.filterwarnings("ignore")

#  defaults
# Set some defaults
params = {'figure.figsize': [12, 6]}
mpl.rcParams.update(params)
sns.set_style("whitegrid")

# %%
# 1) Load & pre-process

csv_path = "C:/Users/dbdch/OneDrive/Desktop/Project progress/mongo_export.csv" 

df = pd.read_csv(csv_path)

# expected columns 
expected = {'position','temperature','humidity','noice','time','createdAt','updatedAt','__v'}
missing = expected - set(df.columns)
if missing:
    print("Warning: dataset missing columns:", missing)
#  proceed with columns we have; require 'time' and the three variables
required = {'time','temperature','humidity','noice'}
if not required.issubset(df.columns):
    raise ValueError(f"Required columns missing: {required - set(df.columns)}")

# Parse time column to datetime and sort
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df = df.set_index('time').sort_index()

# convert to numeric
df = df[['temperature','humidity','noice']].apply(pd.to_numeric, errors='coerce')

# Resample to daily averages (daily means)
df_daily = df.resample('D').mean()

# Fill small gaps with forward/back fill
df_daily = df_daily.ffill().bfill()

# Quick QA
print("Date range:", df_daily.index.min(), "to", df_daily.index.max())
print("Rows (days):", len(df_daily))
print(df_daily.describe().T)


# %%
# 3) Moving average function with a small test
def moving_average(series, window=7, min_periods=1, center=False):
    """Return rolling moving average. Defaults to trailing window (center=False)."""
    return series.rolling(window=window, min_periods=min_periods, center=center).mean()

# Unit test (simple)
_test_series = pd.Series([1,2,3,4,5])
_expected = pd.Series([1.0, 1.5, 2.0, 3.0, 4.0])  # window=3, min_periods=1
_res = moving_average(_test_series, window=3, min_periods=1)
assert np.allclose(_res.values, _expected.values), "moving_average unit test failed"
print("moving_average unit test passed")

# %%
# 4) Identify extremes beyond ±k*std
def identify_extremes(series, k=1.0):
    """
    Returns:
      flags: boolean Series marking extremes (True if |value - mean| > k*std)
      mu, sigma, upper, lower
    """
    s = series.dropna().astype(float)
    mu = s.mean()
    sigma = s.std(ddof=1)
    upper = mu + k*sigma
    lower = mu - k*sigma
    flags = (series > upper) | (series < lower)
    return flags, mu, sigma, upper, lower

# Example for temperature at 1 sigma:
flags, mu, sigma, upper, lower = identify_extremes(df_daily['temperature'], k=1.0)
print(f"Temperature mean={mu:.3f}, std={sigma:.3f}, extremes (1sigma) count={flags.sum()}")


# %%
# 8) Multi-panel figure (stacked) showing daily series, 5-day MA, and extremes for each variable
def panel_plot_multi(df_daily, ks=[1.0], ma_window=5, savepath="multi_panel_figure.pdf"):
    vars_ = ['temperature','humidity','noice']
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12,10), sharex=True)
    for ax, var in zip(axes, vars_):
        ax.plot(df_daily.index, df_daily[var], color='b', linewidth=2, label=f'{var} (daily mean)')
        ax.plot(df_daily.index, moving_average(df_daily[var], window=ma_window), color='r', linewidth=3, label=f'{ma_window}-day MA')
        # annotate extremes for first k in ks (commonly k=1)
        k = ks[0]
        flags, mu, sigma, upper, lower = identify_extremes(df_daily[var], k=k)
        ax.scatter(df_daily.index[flags], df_daily[var][flags], color='black', s=18, label=f'Extremes (±{k}sigma, n={int(flags.sum())})')
        ax.axhline(mu, color='green', linestyle='--', linewidth=3, label=f'Mean ({mu:.2f})')
        ax.axhline(upper, color='m', linestyle='dotted', linewidth=3, label=f'Upper {k}sigma ({upper:.2f})')
        ax.axhline(lower, color='y', linestyle='-.', linewidth=3, label=f'Lower {k}sigma ({lower:.2f})')
        ax.set_ylabel(var.capitalize())
        ax.legend(loc='upper right')
    axes[-1].set_xlabel("Date")
    plt.suptitle("Daily Environment Variables — extremes and 5-day moving average")
    plt.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(savepath,dpi=300)
    plt.show()
    print(f"Saved figure to {savepath}")

# Run the multi-panel creation
panel_plot_multi(df_daily, ks=[1.0], ma_window=5, savepath="env_multi_panel_1sigma.pdf")



