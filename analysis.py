# libraries loading 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats

# load the dataset
data = pd.read_csv('CCLACBM027NBOG.csv')

# ------------------- EDA --------------------------
# convert dates into date types
data['observation_date'] = pd.to_datetime(data['observation_date'])
data.set_index('observation_date', inplace = True)

# basic info about the dataset
print(data.head(10))
print(data.dtypes)
print(data.isnull().sum())

# descriptive statistics about the dataset
print (f"\nMin: ${data['CCLACBM027NBOG'].min()} B")
print (f"\nMax: ${data['CCLACBM027NBOG'].max()} B")
print (f"\nMean: ${data['CCLACBM027NBOG'].mean()} B")
print (f"\nMedian: ${data['CCLACBM027NBOG'].median()} B")
print (f"\nStandard deviation: ${data['CCLACBM027NBOG'].std()} B")
print (f"\nRange: ${data['CCLACBM027NBOG'].max() - data['CCLACBM027NBOG'].min()} B")

# ------------------ TIME SERIES VISUALIZATION ------

# 1) OVERALL TREND
plt.figure(figsize = (14, 6))
plt.plot(data.index, data['CCLACBM027NBOG'], linewidth = 2, color = 'steelblue')
plt.title('Consumer Credit Card Loans Over Time (2000 - 2025)', fontsize = 16, fontweight = 'bold')
plt.xlabel('Date', fontsize = 12)
plt.ylabel('Billions of Dollars', fontsize = 12)
plt.grid(True, alpha = 0.3)
plt.tight_layout()
plt.show()

# 2) CALCULATE GROWTH RATES
plt.close('all')
data['YoY_growth'] = data['CCLACBM027NBOG'].pct_change(12) * 100
data['MoM_growth'] = data['CCLACBM027NBOG'].pct_change() * 100

fig, axis = plt.subplots(2, 1, figsize = (14, 10))

# year by year growth plot
axis[0].plot(data.index, data['YoY_growth'], linewidth = 2, color = 'steelblue')
axis[0].axhline(y = 0, color = 'darkorange', linestyle = '--', linewidth = 1)
axis[0].set_title('Year by Year Growth Rate (%)', fontsize = 14, fontweight = 'bold')
axis[0].set_ylabel('Growth Rate (%)', fontsize = 12)
axis[0].grid (True, alpha = 0.3)

# month by month growth plot
axis[1].plot(data.index, data['MoM_growth'], linewidth = 2, color = 'steelblue')
axis[1].axhline(y = 0, color = 'darkorange', linestyle = '--', linewidth = 1)
axis[1].set_title('Month by Month Growth Rate (%)', fontsize = 14, fontweight = 'bold')
axis[1].set_ylabel('Growth Rate (%)', fontsize = 12)
axis[1].grid (True, alpha = 0.3)

plt.tight_layout()
plt.show()

# 3) DISTRIBUTION ANALYSIS
plt.close('all')
# histograms
plt.figure(figsize = (12, 5))
plt.subplot(1, 2, 1)
plt.hist(data['CCLACBM027NBOG'], bins = 30, edgecolor = 'black', color = 'pink')
plt.title('Distribution of Credit Card Loans', fontsize = 14, fontweight = 'bold')
plt.xlabel('Billions of Dollars', fontsize = 12)
plt.ylabel('Frequency', fontsize = 12)
plt.grid(True, alpha = 0.3)

plt.subplot(1, 2, 2)
plt.boxplot(data['CCLACBM027NBOG'], medianprops = dict(color = 'red', linewidth = 2))
plt.title ('Billions of Dollars', fontsize = 12)
plt.ylabel('Billions of Dollas', fontsize = 12)
plt.grid(True, alpha = 0.3)

plt.tight_layout()
plt.show()

# 4) SEASONALITY ANALYSIS

# MONTHS ------------------------------

plt.close('all')
# monthly patterns
data['month'] = data.index.month
data['year'] = data.index.year

# average by month
monthly_avg = data.groupby('month')['CCLACBM027NBOG'].mean()
monthly_std = data.groupby('month')['CCLACBM027NBOG'].std()

plt.figure(figsize = (12, 6))
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.plot(range(1, 13), monthly_avg.values, marker = 'o', linewidth = 2,
         markersize = 8, color = 'steelblue', label = 'Average')
plt.fill_between(range (1, 13),
                 (monthly_avg - monthly_std).values,
                 (monthly_avg + monthly_std).values,
                 alpha = 0.3, color = "steelblue")
plt.xticks(range(1, 13), months)
plt.title('Average Credit Card loans by Month', fontsize = 14, fontweight = 'bold')
plt.xlabel('Month', fontsize = 12)
plt.ylabel('Billions of Dollars', fontsize = 12)
plt.legend()
plt.grid(True, alpha = 0.3)
plt.tight_layout()
plt.show()

# SEASONS -----------------------------------------
plt.close('all')
# decompose the series
decomposition = seasonal_decompose(data['CCLACBM027NBOG'],
                                   model = 'additive',
                                   period = 12)
fig, axes = plt.subplots(2, 2, figsize = (14, 12))

# original
axes[0, 0].plot(data.index, data['CCLACBM027NBOG'], linewidth = 1.5)
axes[0, 0].set_ylabel('Original', fontsize = 12)
axes[0, 0].set_title('Seasonal Decomposition', fontsize = 14, fontweight = 'bold')

# Trend
axes[0, 1].plot(decomposition.trend.index, decomposition.trend, linewidth = 1.5, color = 'orange')
axes[0, 1].set_ylabel('Trend', fontsize = 12)

# Seasonal
axes[1, 0].plot(decomposition.seasonal.index, decomposition.seasonal, linewidth = 1.5, color = 'green')
axes[1, 0].set_ylabel('Seasonal', fontsize = 12)

# Residual
axes[1, 1].plot(decomposition.resid.index, decomposition.resid, linewidth = 1.5, color = 'red')
axes[1, 1].set_ylabel('Residual', fontsize = 12)
axes[1, 1].set_xlabel('Date', fontsize = 12)

plt.tight_layout()
plt.show()

# 5) STATIONARITY ANALYSIS
plt.close('all')
#Augmented Dickey-Fuller Test

def adf_test(series, name): 
    result = adfuller(series.dropna(), autolag='AIC')
    print(f"\n{name}:")
    print(f" ADF Statistic: {result[0]:.4f}")
    print(f" p-value: {result[1]:.4f}")
    print(f" Critical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.4f}")
    if result[1] <= 0.05:
        print(f"  STATIONARY (reject H0 at 5% level)")
        return True
    else:
       print(f"  NON-STATIONARY (fail to reject H0)")
       return False 
    
# Testing the original series
print("="*60)
print("STATIONARITY TESTS")
print("="*60)
is_stationary = adf_test(data['CCLACBM027NBOG'], "Original Series")

#Testing the first difference
data['first_diff'] = data['CCLACBM027NBOG'].diff() 

is_diff_stationary = adf_test(data['first_diff'], "First Differenced Series")

#Visualizing the differences
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

#Visualization of the orginal
axes[0].plot(data.index, data['CCLACBM027NBOG'], linewidth=2, color='steelblue')
axes[0].set_title('Original Series (Non-Stationary)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Billions of Dollars', fontsize=12)
axes[0].grid(True, alpha=0.3)

#Visualization of the first difference
axes[1].plot(data.index, data['first_diff'], linewidth=1.5, color='darkorange')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[1].set_title('First Difference (Stationary)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('Change (Billions)', fontsize=12)
axes[1].grid(True, alpha=0.3)


plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.show()

#6) AUTOCORRELATION ANALYSIS
plt.close('all')
#ACF and PACF Plots 

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

#ACF plot for original series
plot_acf(data['CCLACBM027NBOG'].dropna(), lags=40, ax=axes [0, 0])
axes[0, 0].set_title('ACF - Original Series', fontsize=12, fontweight='bold')
axes[0, 0].set_ylim(-1.1, 1.1)

#PACF plot for original series
plot_pacf(data['CCLACBM027NBOG'].dropna(), lags=40, ax=axes [0, 1], method='ywm')
axes[0, 1].set_title('PACF - Original Series', fontsize=12, fontweight='bold')
axes[0, 1].set_ylim(-1.1, 1.1)

#ACF plot for first differenced
plot_acf(data['first_diff'].dropna(), lags=40, ax=axes [1, 0])
axes[1, 0].set_title('ACF - First Differenced Series', fontsize=12, fontweight='bold')
axes[1, 0].set_ylim(-1.1, 1.1)

#PACF plot for first differenced 
plot_pacf(data['first_diff'].dropna(), lags=40, ax=axes [1, 1], method='ywm')
axes[1, 1].set_title('PACF - First Differenced Series', fontsize=12, fontweight='bold')
axes[1, 1].set_ylim(-1.1, 1.1) 

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.show()

# 7) STRUCTURAL BREAKS & KEY EVENTS
plt.close('all')

events = {
    '2008-09-15' : ('Lehman Brothers Collapse', 0.95),
    '2020-03-01' : ('COVID-19 Pandemic', 0.95),
    '2022-03-01' : ('Fed Rate Hikes Begin', 0.50)
}

plt.figure(figsize=(14, 7))
plt.plot(data.index, data['CCLACBM027NBOG'], linewidth=2, color='steelblue')

for date, (label, y_pos) in events.items():
    plt.axvline(x=pd.to_datetime(date), color='red', linestyle='--', alpha=0.7)
    plt.text(pd.to_datetime(date) + pd.Timedelta(days=60), plt.ylim()[1]*y_pos, label,
            rotation=90, verticalalignment='top', fontsize=10 )
    
plt.title('Credit Card Loans with Key Economic Events', fontsize= 14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Billions of Dollars', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 8) ROLLING STATISTICS
plt.close('all')
#Rolling mean and standard deviation

data['rolling_mean_12'] = data['CCLACBM027NBOG'].rolling(window=12).mean()
data['rolling_std_12'] = data['CCLACBM027NBOG'].rolling(window=12).std()

plt.figure(figsize=(14,7))
plt.plot(data.index, data['CCLACBM027NBOG'], label='Original', linewidth=2, alpha=0.7)
plt.plot(data.index, data['rolling_mean_12'], label='12-Month Rolling Mean',
         linewidth=2, color='red')
plt.fill_between(data.index,
                 data['rolling_mean_12'] - data['rolling_std_12'],
                 data['rolling_mean_12'] + data['rolling_std_12'],
                 alpha=0.2, color='red', label = '±1 Std Dev')

plt.title('Credit Card Loans with Rolling Statistics', fontsize= 14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Billions of Dollars', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 9)SUMMARY STATISTICS TABLE

summary = pd.DataFrame({
    'Metric' : ['Observation', 'Start Date', 'End Date', 'Mean', 'Std Dev',
                'Min', 'Max', 'Range', 'Total Growth'],
    'Value': [
        len(data),
        data.index[0].strftime('%Y-%m'),
        data.index[-1].strftime('%Y-%m'),
        f"${data['CCLACBM027NBOG'].mean():.2f}B",
        f"${data['CCLACBM027NBOG'].std():.2f}B",
        f"${data['CCLACBM027NBOG'].min():.2f}B",
        f"${data['CCLACBM027NBOG'].max():.2f}B",
        f"${data['CCLACBM027NBOG'].max() - data['CCLACBM027NBOG'].min() :.2f}B",
        f"${((data['CCLACBM027NBOG'].iloc[-1] - data['CCLACBM027NBOG'].iloc[0]) / data['CCLACBM027NBOG'].iloc[0]*100):.1f}%",
    ]            
})

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(summary.to_string(index=False))