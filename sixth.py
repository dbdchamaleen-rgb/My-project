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
expected = {'position','temperature','humidity','noise','time','createdAt','updatedAt','__v'}
missing = expected - set(df.columns)
if missing:
    print("Warning: dataset missing columns:", missing)
#  proceed with columns we have; require 'time' and the three variables
required = {'time','temperature','humidity','noise'}
if not required.issubset(df.columns):
    raise ValueError(f"Required columns missing: {required - set(df.columns)}")

# Parse time column to datetime and sort
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df = df.set_index('time').sort_index()

# convert to numeric
df = df[['temperature','humidity','noise']].apply(pd.to_numeric, errors='coerce')

# Resample to daily averages (daily means)
df_daily = df.resample('D').mean()

# Fill small gaps with forward/back fill
df_daily = df_daily.ffill().bfill()

# Quick QA
print("Date range:", df_daily.index.min(), "to", df_daily.index.max())
print("Rows (days):", len(df_daily))
print(df_daily.describe().T)


# %%
# 7) Correlation and pairwise relationships
corr = df_daily[['temperature','humidity','noise']].corr()
print("\nCorrelation matrix:\n", corr)

plt.figure(figsize=(12,6))
sns.heatmap(corr, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap='coolwarm')
plt.title("Correlation matrix",fontsize=20)
plt.tight_layout()
plt.savefig('sixth.pdf')
plt.show()

# %%
# Pairwise scatter + KDE (sample if > 2000 rows)
sample = df_daily[['temperature','humidity','noice']].dropna()
if sample.shape[0] > 2000:
    sample = sample.sample(2000, random_state=42)
sns.pairplot(sample, kind='reg', diag_kind='kde', plot_kws={'line_kws':{'color':'red','alpha':0.6}})
plt.suptitle("Pairwise relationships (sampled)", y=1.02,fontsize=20)
plt.savefig('seventh.pdf')
plt.show()



