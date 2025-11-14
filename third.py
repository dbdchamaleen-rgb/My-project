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
# 2) Statistical helper functions

def compute_statistics(series):
    """Return dictionary of descriptive stats (works with pandas Series)."""
    s = series.dropna().astype(float)
    return {
        'count': int(s.count()),
        'mean': float(s.mean()),
        'median': float(s.median()),
        'variance': float(s.var(ddof=1)),
        'std': float(s.std(ddof=1)),
        'min': float(s.min()),
        'max': float(s.max()),
        'q25': float(s.quantile(0.25)),
        'q75': float(s.quantile(0.75)),
        'skew': float(s.skew()),
        'kurtosis': float(s.kurtosis())
    }

# Example usage and print neatly
for col in ['temperature','humidity','noice']:
    stats = compute_statistics(df_daily[col])
    print(f"\n--- {col.upper()} ---")
    for k,v in stats.items():
        print(f"{k:8s}: {v:.4f}" if isinstance(v, float) else f"{k:8s}: {v}")


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
# 5) Sensitivity analysis: counts of extreme days as k varies
ks = [0.5, 1.0, 1.5, 2.0, 2.5]
sensitivity_rows = []
for col in ['temperature','humidity','noice']:
    for k in ks:
        flags, mu, sigma, upper, lower = identify_extremes(df_daily[col], k=k)
        sensitivity_rows.append({
            'variable': col,
            'k': k,
            'mean': mu,
            'std': sigma,
            'upper': upper,
            'lower': lower,
            'n_high': int((df_daily[col] > upper).sum()),
            'n_low': int((df_daily[col] < lower).sum()),
            'n_total_extremes': int(flags.sum())
        })
sensitivity_df = pd.DataFrame(sensitivity_rows)
print("\nSensitivity DataFrame (first rows):")
print(sensitivity_df.head(12).to_string(index=False))
# Save for later reading
sensitivity_df.to_csv("extremes_sensitivity_counts.csv", index=False)
print("Saved extremes_sensitivity_counts.csv")

# %%
# 6) Linear trend analysis (days since start)
df_trend = df_daily.copy()
df_trend['days'] = (df_trend.index - df_trend.index.min()).days

trend_results = []
for col in ['temperature','humidity','noice']:
    mask = df_trend[col].notna()
    x = df_trend['days'][mask].values
    y = df_trend[col][mask].values
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    trend_results.append({
        'variable': col,
        'slope_per_day': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    })
    print(f"\n{col.upper()} trend: slope={slope:.6f} per day, R²={r_value**2:.4f}, p={p_value:.4g}")
    # plot
    plt.figure(figsize=(12,6))
    plt.plot(df_trend.index, df_trend[col], color='b', linewidth=2, label='Daily mean')
    plt.plot(df_trend.index, intercept + slope * df_trend['days'], color='g', linewidth=3,linestyle='--', label=f'Linear trend (slope={slope:.4e})')
    plt.title(f"{col.capitalize()} with linear trend",fontsize=20)
    plt.xlabel("Date",fontsize=20)
    plt.ylabel(col.capitalize(),fontsize=20)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig('third.pdf')
    plt.show()

trend_df = pd.DataFrame(trend_results)
trend_df.to_csv("trend_results.csv", index=False)
print("Saved trend_results.csv")



