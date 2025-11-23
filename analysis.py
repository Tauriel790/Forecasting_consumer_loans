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

# Split the data into train (80%) and test(20%)
train_size = int(len(data) * 0.8)
train = data[:train_size]
test = data[train_size:]

print (f"\nTrain set: {len(train)} observations ({train.index[0].strftime('%Y-%m')} to {train.index[-1].strftime('%Y-%m')})")
print (f"Test set: {len(test)} observations ({test.index[0].strftime('%Y-%m')} to {test.index[-1].strftime('%Y-%m')})")
print (f"Test set percentage: {len(test)/len(data) * 100:.1f}%")

# --------------------------------------------------------------------------------------------------------
# MODEL SELECTION - GRID RESEARCH FOR BEST ARIMA ORDER
# --------------------------------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SEARCHING FOR OPTIMAL ARIMA PARAMETERS")
print("=" * 70)

# Grid search for best ARIMA order
best_aic = np.inf
best_bic = np.inf
best_order = None
best_model = None

# Testing for different combinations
# We use d = 1 because we know from ADF test that first difference is stationary
orders_to_test = [
    (0, 1, 0), # Random walk
    (1, 1, 0), # AR(1) with differencing
    (0, 1, 1), # MA(1) with differencing
    (1, 1, 1), # ARIMA(1,1,1)
    (2, 1, 0), # AR(2) with differencing
    (0, 1, 2), # MA(2) with differencing
    (2, 1, 1), # ARIMA(2,1,1)
    (1, 1, 2), # ARIMA(1,1,2)
    (2, 1, 2), # ARIMA(2, 1, 2)
    (3, 1, 0), # AR(3) with diffencing
    (0, 1, 3), # MA(3) with differencing
]

results_list = []

print("\nTesting diffent ARIMA orders...")
print(f"{'Order':<15} {'AIC':<12} {'BIC':<12} {'Status'}")
print("-" * 50)

for order in orders_to_test:
    try:
        # Fit model
        model = ARIMA(train['CCLACBM027NBOG'], order = order)
        fitted = model.fit()

        # Store results
        results_list.append({
            'order': order,
            'aic': fitted.aic,
            'bic': fitted.bic
        })

        print(f"{str(order):<15} {fitted.aic:<12.2f} {fitted.bic:<12.2f}  {'✓'}")

    # Track best model by BIC (more conservative)
        if fitted.bic < best_bic:
            best_bic = fitted.bic
            best_aic = fitted.aic
            best_order = order
            best_model = fitted

    except Exception as e:
        print(f"{str(order):<15} {'Failed':<12} {'Failed':<12} {'✗'} - ERROR: {str(e)}")
        continue

print("\n" + "=" * 70)
print("BEST MODEL SELECTED")
print("=" * 70)
print(f"Best ARIMA order: {best_order}")
print(f"AIC: {best_aic:.2f}")
print(f"BIC: {best_bic:.2f}")

# -----------------------------------------------------------------------------------------------------
# MODEL DIAGNOSTICS
# -----------------------------------------------------------------------------------------------------
print("\n" + "=" *70)
print("MODEL SUMMARY")
print("=" * 70)
print(best_model.summary())

plt.close('all')
# Residuals diagnostics
residuals = best_model.resid

print("\n" + "=" * 70)
print("RESIDUAL DIAGNOSTICS")
print("=" * 70)
print(f"Mean of residuals: {residuals.mean():.4f}")
print(f"Std of residuals: {residuals.std():.4f}")

# Plot diagnostics
fig, axes = plt.subplots(2, 2, figsize = (12, 8))

# Residuals over time
axes[0, 0].plot(residuals, linewidth = 1)
axes[0, 0].axhline(y = 0, color = 'red', linestyle = '--', linewidth = 1)
axes[0, 0].set_title(f'ARIMA{best_order} Residuals', fontsize = 12, fontweight = 'bold')

axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].grid(True, alpha = 0.3)

# Histogram of residuals
axes[0, 1].hist(residuals, bins = 30, edgecolor = 'black', alpha = 0.7, color = 'steelblue')
axes[0, 1].set_title('Distribution of Residuals', fontsize = 12, fontweight = 'bold')
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha = 0.3)

# ACF of residuals
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals, lags = 40, ax = axes[1, 0])
axes[1, 0].set_title ("ACF of Residuals", fontsize = 12, fontweight = 'bold')
axes[1, 0].set_ylim(-0.3, 0.3)

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist = 'norm', plot = axes[1, 1])
axes[1, 1].set_title('Q-Q Plot', fontsize = 12, fontweight = 'bold')
axes[1, 1].grid(True, alpha = 0.3)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# ROLLING WINDOW FORECAST
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("GENERATING ONE-STEP-AHEAD FORECASTS")
print("=" * 70)

# Rolling window forecasts
arima_forecasts = []
arima_actuals = []

print(f"Generating {len(test)} one-step-ahead forecasts using rolling window...")

for i in range(len(test)):
    # Extend training data with observations up to current point
    train_extended = pd.concat([train, test[:i]])

    # Fit ARIMA model
    model = ARIMA(train_extended['CCLACBM027NBOG'], order = best_order)
    fitted = model.fit()

    # One-step-ahead forecast
    forecast = fitted.forecast(steps = 1)
    arima_forecasts.append(forecast.iloc[0])
    arima_actuals.append(test.iloc[i]['CCLACBM027NBOG'])
    # Progress indicator
    if (i + 1) % 10 == 0:
        print(f"Completed {i + 1} / {len(test)} forecasts")


print(test)
print(len(arima_actuals))
print("Forecasts complete")

print(len(arima_actuals))
# Convert to arrays
arima_forecasts = np.array(arima_forecasts)
arima_actuals = np.array(arima_actuals)

# -----------------------------------------------------------------------------------
# FORECAST EVALUATION METRICS
# -----------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FORECAST EVALUATION METRICS")
print("=" * 70)

# Calculate metrics 
rmse = np.sqrt(mean_squared_error(arima_actuals, arima_forecasts))
mae = mean_absolute_error(arima_actuals, arima_forecasts)
mape = np.mean(np.abs((arima_actuals - arima_forecasts) / arima_actuals)) * 100
mse = mean_squared_error(arima_actuals, arima_forecasts)

print(f"\nARIMA{best_order} Performance:")
print(f"  RMSE (Root Mean Squared Error): {rmse:.4f} billion dollars")
print(f"  MAE (Mean Absolute Error): {mae:.4f} billion dollars")
print(f"  MAPE (Mean Absolute Percentage Error): {mape:.4f}%")
print(f"  MSE (Mean Squared Error): {mse:.4f}")

#-------------------------------------------------------------------------------------------
# VISUALIZATION  
#-------------------------------------------------------------------------------------------
plt.close('all')

fig, ax = plt.subplots(3, 1, figsize=(14, 14), constrained_layout=True)

# Plot 1: Full series with forecasts
ax[0].plot(train.index, train['CCLACBM027NBOG'], label='Training Data', color='steelblue', linewidth=2)
ax[0].plot(test.index, arima_actuals, label='Actual Test Data', color='black', linewidth=2)
ax[0].plot(test.index, arima_forecasts, label=f'ARIMA{best_order} Forecasts', color='darkorange', linestyle='--', linewidth=2, alpha=0.8)
ax[0].axvline(x=train.index[-1], color='gray', linestyle=':', linewidth=2)
ax[0].set_title('Credit Card Loans: ARIMA Forecast vs Actual', fontsize=14, fontweight='bold')
ax[0].set_ylabel('Billions of Dollars', fontsize=12)
ax[0].legend(loc='best', fontsize=10)
ax[0].grid(True, alpha=0.3)

# Plot 2: Test period zoom-in
ax[1].plot(test.index, arima_actuals, label='Actual', color='black', linewidth=2.5, marker='o', markersize=4)
ax[1].plot(test.index, arima_forecasts, label=f'ARIMA{best_order} Forecast', color='darkorange', linestyle='--', linewidth=2, marker='s', markersize=4, alpha=0.8)
ax[1].set_title('Test Period Detailed Comparison', fontsize=14, fontweight='bold')
ax[1].set_ylabel('Billions of Dollars', fontsize=12)
ax[1].legend(loc='best', fontsize=10)
ax[1].grid(True, alpha=0.3)

# Plot 3: Forecast errors
forecast_errors = arima_actuals - arima_forecasts
ax[2].plot(test.index, forecast_errors, label='Forecast Errors', color='red', linewidth=2, marker='o', markersize=4)
ax[2].axhline(y=0, color='black', linestyle='-', linewidth=1)
ax[2].set_title('Forecast Errors (Actual - Predicted)', fontsize=14, fontweight='bold')
ax[2].fill_between(test.index, 0, forecast_errors, alpha=0.3, color='red')
ax[2].set_ylabel('Error (Billions)', fontsize=12)
ax[2].set_xlabel('Date', fontsize=12)
ax[2].legend(loc='best', fontsize=10)
ax[2].grid(True, alpha=0.3)

# Rotate x-axis labels for all subplots
for axis in ax:
    axis.tick_params(axis='x', rotation=45)

plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# SAVE MODEL SUMMARY
# ---------------------------------------------------------------------------------------------------------------------
results_df = pd.DataFrame({
    'Date': test.index,
    'Actual': arima_actuals,
    'ARIMA_Forecast': arima_forecasts,
    'Error': forecast_errors,
    'Absolute_Error': np.abs(forecast_errors),
    'Percentage_Error': np.abs(forecast_errors / arima_actuals) * 100
})

results_df.to_csv('arima_forecast_results.csv', index = False)
print("\n Results saved to 'arima_forecast_results.csv")

# Save model summary
with open('arima_model_summary.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("ARIMA MODEL RESULTS\n")
    f.write("="*70 + "\n\n")
    f.write(f"Best Model: ARIMA{best_order}\n")
    f.write(f"AIC: {best_aic:.2f}\n")
    f.write(f"BIC: {best_bic:.2f}\n\n")
    f.write("Forecast Performance:\n")
    f.write(f"  RMSE: {rmse:.4f}\n")
    f.write(f"  MAE: {mae:.4f}\n")
    f.write(f"  MAPE: {mape:.4f}%\n\n")
    f.write("="*70 + "\n")
    f.write("Full Model Summary:\n")
    f.write("="*70 + "\n")
    f.write(str(best_model.summary()))

print("✓ Model summary saved to 'arima_model_summary.txt'")

print("\n" + "="*70)
print("ARIMA ANALYSIS COMPLETE!")
print("="*70)

# -------------------------------------------------------------------------------------------------------
# SARIMA MODEL 
# -------------------------------------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SARIMA MODEL IMPLEMENTATION")
print("=" * 80)

#Check for train and test before
print(f"\nUsing the same train/test split:")
print (f"Train set: {len(train)} observations")
print (f"Test set: {len(test)} observations")

#---------------------------------------------------------------------------------------
# SARIMA MODEL SELECTION - GRID SEARCH
#---------------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SEARCHING FOR OPTIMAL SARIMA PARAMETERS")
print("=" * 80)

best_aic_sarima = np.inf
best_bic_sarima = np.inf
best_order_sarima = None
best_seasonal_order = None
best_model_sarima = None

# Grid search for best SARIMA order
# Non-seasonal: (p, d, q) - keep d = 1 from sationarity test
#Seasonal: (P, D, Q, s) - m = 12 for monthly data 

#Test fewer combinations to save time 

orders_to_test_sarima = [   

    #(p, d, q), (P, D, Q, m)
    ((0, 1, 1), (1, 0, 1, 12)),  # Based on your ARIMA(0,1,1) + seasonal      
    ((0, 1, 1), (0, 1, 1, 12)), # Seasonal differencing     
    ((0, 1, 1), (1, 1, 0, 12)),  # Seasonal AR only     
    ((1, 1, 0), (1, 0, 1, 12)),  #  AR + seasonal MA     
    ((1, 1, 1), (1, 0, 1, 12)),  #  Full non-seasonal + seasonal     
    ((1, 1, 1), (1, 1, 1, 12)),  #  Full model with seasonal diff     
    ((0, 1, 2), (1, 0, 1, 12)),  #  Higher order MA     
    ((2, 1, 0), (1, 0, 1, 12)),  #  Higher order AR     
    ((1, 1, 0), (1, 1, 1, 12)),  #  AR + full seasonal     
    ((0, 1, 1), (2, 0, 1, 12)),  # Higher seasonal AR 
    ]

print("\nTesting different SARIMA orders...")
print(f"{'Non-seasonal':<15} {'Seasonal':<20} {'AIC':<12} {'BIC':<12} {'Status'}")
print("-" * 75)

for order, seasonal_order in orders_to_test_sarima:
    try:
        # Fit SARIMA model
        model = SARIMAX(train['CCLACBM027NBOG'],
                        order = order,
                        seasonal_order = seasonal_order,
                        enforce_stationarity = False,
                        enforce_invertibility = False)
        fitted = model.fit(disp = False, maxiter=200)

        print(f"{str(order):<15} {str(seasonal_order):<20} {fitted.aic:<12.2f} {fitted.bic:<12.2f}  {'✓'}")

        # Track best model by BIC
        if fitted.bic < best_bic_sarima:
            best_bic_sarima = fitted.bic
            best_aic_sarima = fitted.aic
            best_order_sarima = order
            best_seasonal_order = seasonal_order
            best_model_sarima = fitted

    except Exception as e:
        print(f"{str(order):<15} {str(seasonal_order):<20} {'Failed':<12} {'Failed':<12} {'✗'} ")
        continue

print("\n" + "=" * 80)
print("BEST SARIMA MODEL SELECTED")
print("=" * 80)
print(f"Best SARIMA order: {best_order_sarima}")
print(f"Best seasonal order: {best_seasonal_order}")
print(f"AIC: {best_aic_sarima:.2f}")
print(f"BIC: {best_bic_sarima:.2f}")

# Comparison with ARIMA
print("\n" + "=" * 80)
print("COMPARISON WITH ARIMA")
print("=" * 80)
print(f"ARIMA(0, 1, 1) BIC: {best_bic:.2f}")
print(f"SARIMA{best_order_sarima}{best_seasonal_order} BIC: {best_bic_sarima:.2f}")
if best_bic_sarima < best_bic:
    print(f"SARIMA is better by {best_bic - best_bic_sarima:.2f} BIC points.")
else:
    print(f"ARIMA still better by {best_bic_sarima - best_bic:.2f} BIC points.")   

#-------------------------------------------------------------------------------------
# SARIMA MODEL DIAGNOSTICS
#-------------------------------------------------------------------------------------
print("\n" + "=" *80)
print("SARIMA MODEL SUMMARY")
print("=" * 80)
print(best_model_sarima.summary())

# Residual diagnostics
residuals_sarima = best_model_sarima.resid

print("\n" + "=" * 80)
print("SARIMA RESIDUAL DIAGNOSTICS")
print("=" * 80)
print(f"Mean of residuals: {residuals_sarima.mean():.4f}")
print(f"Std of residuals: {residuals_sarima.std():.4f}")

# Plot diagnostics
plt.close('all')
fig, axes = plt.subplots(2, 2, figsize = (12, 8))

# Residuals over time
axes[0, 0].plot(residuals_sarima, linewidth = 1)
axes[0, 0].axhline(y = 0, color = 'red', linestyle = '--', linewidth = 1)
axes[0, 0].set_title(f'SARIMA{best_order_sarima}{best_seasonal_order} Residuals', 
                     fontsize = 12, fontweight = 'bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].grid(True, alpha = 0.3)

# Histogram of residuals
axes[0, 1].hist(residuals_sarima, bins = 30, edgecolor = 'black', alpha = 0.7, color = 'green')
axes[0, 1].set_title('Distribution of Residuals', fontsize = 12, fontweight = 'bold')
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha = 0.3)

# ACF of residuals
plot_acf(residuals_sarima, lags = 40, ax = axes[1, 0])
axes[1, 0].set_title ("ACF of Residuals", fontsize = 12, fontweight = 'bold')
axes[1, 0].set_ylim(-0.3, 0.3)

# Q-Q plot
stats.probplot(residuals_sarima, dist = 'norm', plot = axes[1, 1])
axes[1, 1].set_title('Q-Q Plot', fontsize = 12, fontweight = 'bold')
axes[1, 1].grid(True, alpha = 0.3)

plt.tight_layout()
plt.show()

print("\nSARIMA diagnostic plots complete.")

# ---------------------------------------------------------------------------------------
# SARIMA ONE-STEP-AHEAD FORECASTING
# ---------------------------------------------------------------------------------------

#SARIMA ROLLING WINDOW FORECAST

print("\n" + "=" * 80)
print("GENERATING SARIMA ONE-STEP-AHEAD FORECASTS")
print("=" * 80)

# Rolling window forecasts
sarima_forecasts = []
sarima_actuals = []

print(f"Generating {len(test)} one-step-ahead forecasts using rolling window...")

for i in range(len(test)):
    # Extend training data with observations up to current point
    train_extended = pd.concat([train, test[:i]])

    # Fit SARIMA model
    model = SARIMAX(train_extended['CCLACBM027NBOG'],
                    order = best_order_sarima,
                    seasonal_order = best_seasonal_order,
                    enforce_stationarity = False,
                    enforce_invertibility = False)
    fitted = model.fit(disp = False, maxiter=200)

    # One-step-ahead forecast
    forecast = fitted.forecast(steps = 1)

    #Extract scalar values
    forecast_value = float(forecast.iloc[0])
    actual_value = float(test.iloc[i]['CCLACBM027NBOG'])

    sarima_forecasts.append(forecast_value)
    sarima_actuals.append(actual_value)

    # Progress indicator
    if (i + 1) % 10 == 0:
        print(f"Completed {i + 1} / {len(test)} forecasts")
    
print("SARIMA forecasts complete.")

# Convert to numpy arrays
sarima_forecasts = np.array(sarima_forecasts)
sarima_actuals = np.array(sarima_actuals)

#Verify
print(f"\nSARIMA forecast array shape: {sarima_forecasts.shape}")
print(f"SARIMA actuals array shape: {sarima_actuals.shape}")

# -----------------------------------------------------------------------------------
# SARIMA FORECAST EVALUATION 
# -----------------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SARIMA FORECAST EVALUATION METRICS")
print("=" * 80)

# Calculate metrics
rmse_sarima = np.sqrt(mean_squared_error(sarima_actuals, sarima_forecasts))
mae_sarima = mean_absolute_error(sarima_actuals, sarima_forecasts)
mape_sarima = np.mean(np.abs((sarima_actuals - sarima_forecasts) / sarima_actuals)) * 100
mse_sarima = mean_squared_error(sarima_actuals, sarima_forecasts)

print(f"\nSARIMA{best_order_sarima}{best_seasonal_order} Performance:")
print(f"  RMSE: {rmse_sarima:.4f} billion dollars")
print(f"  MAE: {mae_sarima:.4f} billion dollars")
print(f"  MAPE: {mape_sarima:.4f}%")
print(f"  MSE: {mse_sarima:.4f}")

# ---------------------------------------------------------------------------------------
# COMPARE ARIMA vs SARIMA 
# ---------------------------------------------------------------------------------------

print("\n" + "=" * 80)
print("ARIMA vs SARIMA COMPARISON")
print("=" * 80)

comparison_df = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'MAPE (%)', 'MSE'],
    'ARIMA(0,1,1)': [rmse, mae, mape, mse],
    f'SARIMA{best_order_sarima}{best_seasonal_order}': [rmse_sarima, mae_sarima, mape_sarima, mse_sarima]
})

print(comparison_df.to_string(index = False))

# Calculate improvement
rmse_improvement = ((rmse - rmse_sarima)/rmse) * 100
mae_improvement = ((mae - mae_sarima)/mae) * 100
mape_improvement = ((mape - mape_sarima)/mape) * 100

print("\n" + "=" * 80)
print("IMPROVEMENT OVER ARIMA")
print("=" * 80)
if rmse_improvement > 0:
    print(f"SARIMA improved RMSE by {rmse_improvement:.2f}% over ARIMA.")
else:
    print(f"SARIMA RMSE worse by {abs(rmse_improvement):.2f}% compared to ARIMA.")

if mae_improvement > 0:
    print(f"SARIMA improved MAE by {mae_improvement:.2f}% over ARIMA.")
else:
    print(f"SARIMA MAE worse by {abs(mae_improvement):.2f}% compared to ARIMA.")

if mape_improvement > 0:
    print(f"SARIMA improved MAPE by {mape_improvement:.2f}% over ARIMA.")
else:
    print(f"SARIMA MAPE worse by {abs(mape_improvement):.2f}% compared to ARIMA.")
# -------------------------------------------------------------------------------------

# Count which model wins on each metric!!!!!!(NOT NECESSARY)!!!!
sarima_better_count = 0
if rmse_sarima < rmse: 
    sarima_better_count += 1
if mae_sarima < mae: 
    sarima_better_count += 1
if mape_sarima < mape: 
    sarima_better_count += 1
if best_bic_sarima < best_bic: 
    sarima_better_count += 1

if sarima_better_count >= 3:
    print("Overall, SARIMA is the superior model for this dataset.")

#-------------------------------------------------------------------------------------
# SARIMA VISUALIZATION
#-------------------------------------------------------------------------------------
plt.close('all')

print("\n" + "=" * 80)
print("GENERATING SARIMA FORECAST PLOTS")
print("=" * 80)

fig, axes = plt.subplots(3, 1, figsize=(14, 14))

# Plot 1: Full series with both forecasts
axes[0].plot(train.index, train['CCLACBM027NBOG'],
            label='Training Data', color='steelblue', linewidth=2, alpha=0.7)
axes[0].plot(test.index, sarima_actuals,
            label='Actual Test Data', color='black', linewidth=2.5)
axes[0].plot(test.index, arima_forecasts,
             label='ARIMA{best_order} Forecast',linewidth=2, 
             linestyle='--', color='darkorange', alpha=0.7)
axes[0].plot(test.index, sarima_forecasts,
             label=f'SARIMA{best_order_sarima}{best_seasonal_order} Forecast',
             linewidth=2, linestyle='--', color='green', alpha=0.8)
axes[0].axvline(x=train.index[-1], color='gray', linestyle=':', linewidth=2)
axes[0].set_title('Credit Card Loans: ARIMA vs SARIMA Forecasts', 
                  fontsize=14, fontweight='bold')
axes[0].set_ylabel('Billions of Dollars', fontsize=12)
axes[0].legend(loc='best', fontsize=10) 
axes[0].grid(True, alpha=0.3)

# Plot 2: Test period comparison
axes[1].plot(test.index, sarima_actuals,
             label='Actual', color='black', linewidth=2.5, marker='o', markersize=4)
axes[1].plot(test.index, arima_forecasts,  
             label='ARIMA Forecast', color='darkorange', linestyle='--',
             linewidth=2, marker='s', markersize=3, alpha=0.7)
axes[1].plot(test.index, sarima_forecasts,  
             label='SARIMA Forecast', color='green', linestyle='--', linewidth =2,
             marker ='^', markersize=3, alpha=0.8)
axes[1].set_title('Test Period: ARIMA vs Sarıma Detailed Comparison',
                  fontsize=14, fontweight='bold')
axes[1].set_ylabel('Billions of Dollars', fontsize=12)
axes[1].legend(loc='best', fontsize=10)
axes[1].grid(True, alpha=0.3)

#Plot 3: Forecast Error Comparison
arima_errors = arima_actuals - arima_forecasts
sarima_errors = sarima_actuals - sarima_forecasts

axes[2].plot(test.index, arima_errors,
             label='ARIMA Error', linewidth=2, color='darkorange',
             marker='o', markersize=4, alpha=0.7)
axes[2].plot(test.index, sarima_errors,
             label='SARIMA Error', linewidth=2, color='green',
             marker='s', markersize=4, alpha=0.8)
axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[2].set_title('Forecast Error Comparison (Actual - Predicted)',
                  fontsize=14, fontweight='bold')
axes[2].set_xlabel('Date', fontsize=12)
axes[2].set_ylabel('Error (Billions)', fontsize=12)
axes[2].legend(loc='best', fontsize=10)
axes[2].grid(True, alpha=0.3)

#Adjust spacing

plt.tight_layout()
plt.subplots_adjust(
    left=0.08,
    right=0.96,
    top=0.96,
    bottom=0.08,
    hspace=0.70

)

plt.show()

#-------------------------------------------------------------------------------------
# SAVE SARIMA RESULTS
#---------------------------------------------------------------------------------------------------

# Save forecast results
results_sarima_df = pd.DataFrame({
    'Date': test.index,
    'Actual': sarima_actuals,
    'ARIMA_Forecast': arima_forecasts,
    'SARIMA_Forecast': sarima_forecasts,
    'ARIMA_Error': arima_errors,
    'SARIMA_Error': sarima_errors,
    'ARIMA_Abs_Error': np.abs(arima_errors),
    'SARIMA_Abs_Error': np.abs(sarima_errors)
})

results_sarima_df.to_csv('sarima_comparison_results.csv', index=False)
print("\nResults are saved to 'sarima_comparison_results.csv'")

#Save model summary
with open('sarima_model_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("SARIMA MODEL RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Best SARIMA Model: {best_order_sarima}{best_seasonal_order}\n")
    f.write(f"AIC: {best_aic_sarima:.2f}\n")
    f.write(f"BIC: {best_bic_sarima:.2f}\n\n")
    f.write("Forecast Performance:\n")
    f.write(f"  RMSE: {rmse_sarima:.4f}\n")
    f.write(f"  MAE: {mae_sarima:.4f}\n")
    f.write(f"  MAPE: {mape_sarima:.4f}%\n\n")
    f.write("="*80 + "\n")
    f.write("COMPARISON WITH ARIMA(0,1,1):\n")
    f.write("="*80 + "\n")
    f.write(f"ARIMA RMSE: 13.9185\n")
    f.write(f"SARIMA RMSE: {rmse_sarima:.4f}\n")
    f.write(f"Improvement: {rmse_improvement:.2f}%\n\n")
    f.write("="*80 + "\n")
    f.write("Full Model Summary:\n")
    f.write("="*80 + "\n")
    f.write(str(best_model_sarima.summary()))
 
print("✓ Model summary saved to 'sarima_model_summary.txt'")
 
print("\n" + "="*80)
print("SARIMA ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. sarima_comparison_results.csv - Detailed forecast data")
print("  2. sarima_model_summary.txt - Model summary and metrics")

#-------------------------------------------------------------------------------------
# SARIMAX MODEL WITH EXOGENOUS VARIABLES
#-------------------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SARIMAX MODEL WITH EXOGENOUS VARIABLES")
print("=" * 80)

# Load exogenous variables
fedfunds = pd.read_csv('FEDFUNDS.csv')
unrate = pd.read_csv('UNRATE.csv')

# Convert dates and set index
fedfunds['observation_date'] = pd.to_datetime(fedfunds['observation_date'])
fedfunds.set_index('observation_date', inplace=True)

unrate['observation_date'] = pd.to_datetime(unrate['observation_date'])
unrate.set_index('observation_date', inplace=True)

# Rename columns for clarity
fedfunds.columns = ['FedFunds_Rate']
unrate.columns = ['Unemployment_Rate']

print("Exogenous variables loaded successfully.")
print(f" Federal Funds Rate : {len(fedfunds)} observations")
print(f" Unemployment Rate  : {len(unrate)} observations")

# Merge exogenous variables with main dataset
data_exog = data[['CCLACBM027NBOG']].copy()
data_exog = data_exog.join(fedfunds, how='left')
data_exog = data_exog.join(unrate, how='left')

# Forward fill missing values
data_exog.fillna(method='ffill', inplace=True)

print("\nMerged dataset shape:", data_exog.shape)
print("\nFirst few rows:")
print(data_exog.head())

# Check for missing values
print("\nMissing values:")
print(data_exog.isnull().sum())

#-------------------------------------------------------------------------------------
# EDA FOR EXOGENOUS VARIABLES
#-------------------------------------------------------------------------------------

plt.close('all')
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Consumer Credit
axes[0].plot(data_exog.index, data_exog['CCLACBM027NBOG'], 
             color='steelblue', linewidth=2)
axes[0].set_title('Consumer Credit Card Loans', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Billions of Dollars', fontsize=10)
axes[0].grid(True, alpha=0.3)

# Federal Funds Rate
axes[1].plot(data_exog.index, data_exog['FedFunds_Rate'], 
             color='darkorange', linewidth=2)
axes[1].set_title('Federal Funds Rate', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Percentage (%)', fontsize=10)
axes[1].grid(True, alpha=0.3)

# Unemployment Rate
axes[2].plot(data_exog.index, data_exog['Unemployment_Rate'], 
             color='green', linewidth=2)
axes[2].set_title('Unemployment Rate', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Percentage (%)', fontsize=10)
axes[2].grid(True, alpha=0.3)

# Fix overlapping dates
fig.autofmt_xdate()
 
# Reduce space between subplots
fig.subplots_adjust(hspace=0.35)
plt.show()  

# Correlation analysis
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

correlation_matrix = data_exog.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Visualize correlation 
plt.close('all')
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square =True, linewidth=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix (Consumer Credits & Exogenous Variables)', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

#-------------------------------------------------------------------------------------
# SPLIT DATA WITH EXOGENOUS VARIABLES
#------------------------------------------------------------------------------------

# Train-test split (80/20)
train_exog = data_exog[:train_size]
test_exog = data_exog[train_size:]

print("\n" + "=" * 80)
print("TRAIN-TEST SPLIT WITH EXOGENOUS VARIABLES")
print("=" * 80)
print(f"Train set: {len(train_exog)} observations")
print(f"Test set: {len(test_exog)} observations")

# Separate target and exogenous variables
y_train = train_exog['CCLACBM027NBOG']
x_train = train_exog[['FedFunds_Rate', 'Unemployment_Rate']]

y_test = test_exog['CCLACBM027NBOG']
x_test = test_exog[['FedFunds_Rate', 'Unemployment_Rate']]

#-------------------------------------------------------------------------------------
# SARIMAX MODEL SELECTION
#------------------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SEARCHING FOR OPTIMAL SARIMAX PARAMETERS")
print("=" * 80)

best_aic_sarimax = np.inf
best_bic_sarimax = np.inf
best_order_sarimax = None
best_seasonal_order_sarimax = None
best_model_sarimax = None

#Test similar orders as SARIMA, but now with exogenous variables

orders_to_test_sarimax = [
    # (p, d, q), (P, D, Q, m)
    ((0, 1, 1), (1, 0, 1, 12)),
    ((0, 1, 1), (0, 1, 1, 12)),
    ((0, 1, 1), (1, 1, 0, 12)),
    ((1, 1, 0), (1, 0, 1, 12)),
    ((1, 1, 1), (1, 0, 1, 12)),
    ((1, 1, 1), (1, 1, 1, 12)),
    ((0, 1, 2), (1, 0, 1, 12)),
    ((2, 1, 0), (1, 0, 1, 12)),
    ((1, 1, 0), (1, 1, 1, 12)),
    ((0, 1, 1), (2, 0, 1, 12)), 
]

print("\nTesting SARIMAX orders with exogenous variables...")
print(f"{'Non-seasonal':<15} {'Seasonal':<20} {'AIC':<12} {'BIC':<12} {'Status'}")
print("-" * 75)

for order, seasonal_order in orders_to_test_sarimax:
    try:
        # Fit SARIMAX model with exogenous variables
        model = SARIMAX(y_train,
                        exog = x_train,
                        order = order,
                        seasonal_order = seasonal_order,
                        enforce_stationarity = False,
                        enforce_invertibility = False)
        fitted = model.fit(disp = False, maxiter=200)

        print(f"{str(order):<15} {str(seasonal_order):<20} {fitted.aic:<12.2f} {fitted.bic:<12.2f}  {'✓'}")

        # Track best model by BIC
        if fitted.bic < best_bic_sarimax:
            best_bic_sarimax = fitted.bic
            best_aic_sarimax = fitted.aic
            best_order_sarimax = order
            best_seasonal_order_sarimax = seasonal_order
            best_model_sarimax = fitted

    except Exception as e:
        print(f"{str(order):<15} {str(seasonal_order):<20} {'Failed':<12} {'Failed':<12} {'✗'} ")
        continue

print("\n" + "=" * 80)
print("BEST SARIMAX MODEL SELECTED")
print("=" * 80)
print(f"Best SARIMAX order: {best_order_sarimax}")
print(f"Best SARIMAX seasonal order: {best_seasonal_order_sarimax}")
print(f"Best SARIMAX AIC: {best_aic_sarimax:.2f}")
print(f"Best SARIMAX BIC: {best_bic_sarimax:.2f}")

#-------------------------------------------------------------------------------------
# SARIMAX MODEL DIAGNOSTICS
#-------------------------------------------------------------------------------------

print("\n" + "=" *80)
print("SARIMAX MODEL SUMMARY")
print("=" * 80)
print(best_model_sarimax.summary())

# Coefficient interpretation
print("\n" + "=" * 80)
print("EXOGENOUS VARIABLE COEFFICIENTS")
print("=" * 80)
params = best_model_sarimax.params
print("\nFederal Funds Rate Coefficient:", params.get('FedFunds_Rate', 'N/A'))
print("Unemployment Rate Coefficient:", params.get('Unemployment_Rate', 'N/A'))
print("\nInterpretation:")
print(" - Positive coefficient: Variable increase → Credit increases")
print(" - Negative coefficient: Variable increase → Credit decreases")

# Residual diagnostics
residuals_sarimax = best_model_sarimax.resid

print("\n" + "=" * 80)
print("SARIMAX RESIDUAL DIAGNOSTICS")
print("=" * 80)
print(f"Mean of residuals: {residuals_sarimax.mean():.4f}")
print(f"Std of residuals: {residuals_sarimax.std():.4f}")

# Plot diagnostics
plt.close('all')
fig, axes = plt.subplots(2, 2, figsize = (12, 8))

# Residuals over time
axes[0, 0].plot(residuals_sarimax, linewidth = 1)
axes[0, 0].axhline(y = 0, color = 'red', linestyle = '--', linewidth = 1)
axes[0, 0].set_title(f'SARIMAX{best_order_sarimax}{best_seasonal_order_sarimax} Residuals', 
                     fontsize = 12, fontweight = 'bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Residuals')  
axes[0, 0].grid(True, alpha = 0.3)

# Histogram of residuals
axes[0, 1].hist(residuals_sarimax, bins = 30, edgecolor = 'black', 
                alpha = 0.7, color = 'purple')
axes[0, 1].set_title('Distribution of Residuals', fontsize = 12, fontweight = 'bold')
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha = 0.3)  

# ACF of residuals
plot_acf(residuals_sarimax, lags = 40, ax = axes[1, 0])
axes[1, 0].set_title ("ACF of Residuals", fontsize = 12, fontweight = 'bold')
axes[1, 0].set_ylim(-0.3, 0.3)

# Q-Q plot
stats.probplot(residuals_sarimax, dist = 'norm', plot = axes[1, 1])
axes[1, 1].set_title('Q-Q Plot', fontsize = 12, fontweight = 'bold')
axes[1, 1].grid(True, alpha = 0.3)

plt.tight_layout()
plt.show()
print("\nSARIMAX diagnostic plots complete.")

# ---------------------------------------------------------------------------------------
# SARIMAX ONE-STEP-AHEAD FORECASTING    
# ---------------------------------------------------------------------------------------

print("\n" + "=" * 80)
print("GENERATING SARIMAX ONE-STEP-AHEAD FORECASTS")
print("=" * 80)

sarimax_forecasts = []
sarimax_actuals = []

print(f"Generating {len(test_exog)} one-step-ahead forecasts using rolling window with exogenous variables...")

for i in range(len(test_exog)):
   
    # Extend training data 
    y_train_extended = pd.concat([y_train, y_test[:i]])
    x_train_extended = pd.concat([x_train, x_test[:i]])

    #Get next observation's exogenous variables
    x_next = x_test.iloc[i:i+1]

    # Fit SARIMAX model
    model = SARIMAX(y_train_extended,
                    exog = x_train_extended,
                    order = best_order_sarimax,
                    seasonal_order = best_seasonal_order_sarimax,
                    enforce_stationarity = False,
                    enforce_invertibility = False)
    fitted = model.fit(disp = False, maxiter=200)

    # One-step-ahead forecast
    forecast = fitted.forecast(steps = 1, exog = x_next)

    forecast_value = float(forecast.iloc[0])
    actual_value = float(y_test.iloc[i])

    sarimax_forecasts.append(forecast_value)
    sarimax_actuals.append(actual_value)

    if (i + 1) % 10 == 0:
        print(f"Completed {i + 1} / {len(test_exog)} forecasts")

print("SARIMAX forecasts complete.")

# Convert to numpy arrays
sarimax_forecasts = np.array(sarimax_forecasts)
sarimax_actuals = np.array(sarimax_actuals)

#-------------------------------------------------------------------------------------------
# SARIMAX FORECAST EVALUATION
#-------------------------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SARIMAX FORECAST EVALUATION METRICS")
print("=" * 80)

# Calculate metrics
rmse_sarimax = np.sqrt(mean_squared_error(sarimax_actuals, sarimax_forecasts))
mae_sarimax = mean_absolute_error(sarimax_actuals, sarimax_forecasts)
mape_sarimax = np.mean(np.abs((sarimax_actuals - sarimax_forecasts) / sarimax_actuals)) * 100
mse_sarimax = mean_squared_error(sarimax_actuals, sarimax_forecasts)

print(f"\nSARIMAX{best_order_sarimax}{best_seasonal_order_sarimax} Performance:")
print(f"  RMSE: {rmse_sarimax:.4f} billion dollars")
print(f"  MAE: {mae_sarimax:.4f} billion dollars")
print(f"  MAPE: {mape_sarimax:.4f}%")           
print(f"  MSE: {mse_sarimax:.4f}")

# ---------------------------------------------------------------------------------------
# COMPREHENSIVE MODEL COMPARISON
# ---------------------------------------------------------------------------------------

print("\n" + "=" * 80)
print("COMPREHENSIVE MODEL COMPARISON")
print("=" * 80)

comparison_full = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'MAPE (%)', 'MSE','BIC'],
    'ARIMA(0,1,1)': [rmse, mae, mape, mse, best_bic],
    f'SARIMA{best_order_sarima}': [rmse_sarima, mae_sarima, mape_sarima, mse_sarima, best_bic_sarima],
    f'SARIMAX{best_order_sarimax}': [rmse_sarimax, mae_sarimax, mape_sarimax, mse_sarimax, best_bic_sarimax]
})

print("\n", comparison_full.to_string(index = False))

# Calculate improvements
print("\n" + "=" * 80)
print("SARIMAX IMPROVEMENTS OVER BASELINE MODELS")
print("=" * 80)

rmse_improvement_vs_arima = ((rmse - rmse_sarimax)/rmse) * 100
rmse_improvement_vs_sarima = ((rmse_sarima - rmse_sarimax)/rmse_sarima) * 100

mae_improvement_vs_arima = ((mae - mae_sarimax)/mae) * 100
mae_improvement_vs_sarima = ((mae_sarima - mae_sarimax)/mae_sarima) * 100

mape_improvement_vs_arima = ((mape - mape_sarimax)/mape) * 100
mape_improvement_vs_sarima = ((mape_sarima - mape_sarimax)/mape_sarima) * 100

print("\n" + "=" * 80)
print("SARIMAX IMPROVEMENTS OVER BASELINE MODELS")
print("=" * 80)

print(f"\nRMSE Improvement over ARIMA: {rmse_improvement_vs_arima:.2f}%")
print(f"RMSE Improvement over SARIMA: {rmse_improvement_vs_sarima:.2f}%")

print(f"MAE Improvement over ARIMA: {mae_improvement_vs_arima:.2f}%")
print(f"MAE Improvement over SARIMA: {mae_improvement_vs_sarima:.2f}%") 

print(f"\nMAPE Improvement over ARIMA: {mape_improvement_vs_arima:.2f}%")
print(f"MAPE Improvement over SARIMA: {mape_improvement_vs_sarima:.2f}%")

# Detailed improvement assessment !!!!!KINDA UNNECESSARY!!!!!

print("\n" + "=" * 80)
print("IMPROVEMENT ASSESSMENT")
print("=" * 80)

if rmse_improvement_vs_arima > 0:
    print(f"✓ SARIMAX improved RMSE by {rmse_improvement_vs_arima:.2f}% over ARIMA.")
else:
    print(f"✗ SARIMAX RMSE worse by {abs(rmse_improvement_vs_arima):.2f}% compared to ARIMA.")

if rmse_improvement_vs_sarima > 0:
    print(f"✓ SARIMAX improved RMSE by {rmse_improvement_vs_sarima:.2f}% over SARIMA.")
else:
    print(f"✗ SARIMAX RMSE worse by {abs(rmse_improvement_vs_sarima):.2f}% compared to SARIMA.")

if mae_improvement_vs_arima > 0:
    print(f"✓ SARIMAX improved MAE by {mae_improvement_vs_arima:.2f}% over ARIMA.")
else:
    print(f"✗ SARIMAX MAE worse by {abs(mae_improvement_vs_arima):.2f}% compared to ARIMA.")

if mae_improvement_vs_sarima > 0:
    print(f"✓ SARIMAX improved MAE by {mae_improvement_vs_sarima:.2f}% over SARIMA.")
else:
    print(f"✗ SARIMAX MAE worse by {abs(mae_improvement_vs_sarima):.2f}% compared to SARIMA.")

if mape_improvement_vs_arima > 0:
    print(f"✓ SARIMAX improved MAPE by {mape_improvement_vs_arima:.2f}% over ARIMA.")
else:
    print(f"✗ SARIMAX MAPE worse by {abs(mape_improvement_vs_arima):.2f}% compared to ARIMA.")

if mape_improvement_vs_sarima > 0:
    print(f"✓ SARIMAX improved MAPE by {mape_improvement_vs_sarima:.2f}% over SARIMA.")
else:
    print(f"✗ SARIMAX MAPE worse by {abs(mape_improvement_vs_sarima):.2f}% compared to SARIMA.")

# Finding the best model overall
# Count which model wins on each metric
model_scores = {
    'ARIMA': 0,
    'SARIMA': 0,
    'SARIMAX': 0
}

# Score based on RMSE
if rmse <= rmse_sarima and rmse <= rmse_sarimax:
    model_scores['ARIMA'] += 1
elif rmse_sarima <= rmse and rmse_sarima <= rmse_sarimax:
    model_scores['SARIMA'] += 1
else:
    model_scores['SARIMAX'] += 1

# Score based on MAE
if mae <= mae_sarima and mae <= mae_sarimax:
    model_scores['ARIMA'] += 1
elif mae_sarima <= mae and mae_sarima <= mae_sarimax:
    model_scores['SARIMA'] += 1
else:
    model_scores['SARIMAX'] += 1

# Score based on MAPE
if mape <= mape_sarima and mape <= mape_sarimax:
    model_scores['ARIMA'] += 1
elif mape_sarima <= mape and mape_sarima <= mape_sarimax:
    model_scores['SARIMA'] += 1
else:
    model_scores['SARIMAX'] += 1

# Score based on BIC (lower is better)
if best_bic <= best_bic_sarima and best_bic <= best_bic_sarimax:
    model_scores['ARIMA'] += 1
elif best_bic_sarima <= best_bic and best_bic_sarima <= best_bic_sarimax:
    model_scores['SARIMA'] += 1
else:
    model_scores['SARIMAX'] += 1

print("\n" + "=" * 80)
print("MODEL PERFORMANCE SCORECARD")
print("=" * 80)
print(f"ARIMA wins on {model_scores['ARIMA']}/4 metrics")
print(f"SARIMA wins on {model_scores['SARIMA']}/4 metrics")
print(f"SARIMAX wins on {model_scores['SARIMAX']}/4 metrics")

# Determine overall winner
best_model_overall = max(model_scores, key=model_scores.get)
print(f"\n{'='*80}")
print(f"RECOMMENDED MODEL: {best_model_overall}")
print(f"{'='*80}")

if best_model_overall == 'SARIMAX':
    print("SARIMAX is the superior model, winning on the most evaluation criteria.")
    print("The exogenous variables (Fed Funds Rate, Unemployment) add predictive value.")
elif best_model_overall == 'SARIMA':
    print("SARIMA is the superior model, winning on the most evaluation criteria.")
    print("Seasonal patterns are important; exogenous variables don't add enough value.")
else:
    print("ARIMA (simple benchmark) is the superior model.")
    print("More complex models don't improve forecast accuracy significantly.")

#-------------------------------------------------------------------------------------
# VISUALIZATION: SARIMAX vs OTHER MODELS
#-------------------------------------------------------------------------------------

plt.close('all')
print("\n" + "=" * 80)
print("GENERATING COMPREHENSIVE FORECAST COMPARISON PLOTS")
print("=" * 80)

fig, axes = plt.subplots(4, 1, figsize=(10, 12))     

# Plot 1: Full series with all forecasts
axes[0].plot(train.index, train['CCLACBM027NBOG'],
            label='Training Data', color='steelblue', linewidth=2, alpha=0.7)
axes[0].plot(test.index, sarimax_actuals,
            label='Actual Test Data', color='black', linewidth=2.5)
axes[0].plot(test.index, arima_forecasts,
             label=f'ARIMA{best_order}',linewidth=2, 
             linestyle='--', color='darkorange', alpha=0.7)
axes[0].plot(test.index, sarima_forecasts,
             label=f'SARIMA{best_order_sarima}{best_seasonal_order}',
                linewidth=2, linestyle='--', color='green', alpha=0.7)
axes[0].plot(test.index, sarimax_forecasts,
             label=f'SARIMAX{best_order_sarimax}{best_seasonal_order_sarimax}',
             linewidth=2, linestyle='--', color='purple', alpha=0.8)
axes[0].axvline(x=train.index[-1], color='gray', linestyle=':', linewidth=2)
axes[0].set_title('All Models: Forecast Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Billions', fontsize=12)
axes[0].legend(loc='best', fontsize=9)
axes[0].grid(True, alpha=0.3)

# Plot 2: Test period detailed view
axes[1].plot(test.index, sarimax_actuals,
             label='Actual', color='black', linewidth=2.5, marker='o', markersize=4)
axes[1].plot(test.index, arima_forecasts,  
             label='ARIMA', color='darkorange', linestyle='--', linewidth=2, 
             marker='s', markersize=3, alpha=0.7)
axes[1].plot(test.index, sarima_forecasts,  
             label='SARIMA', color='green', linestyle='--', linewidth=2,
             marker='^', markersize=3, alpha=0.7)   
axes[1].plot(test.index, sarimax_forecasts,  
             label='SARIMAX', color='purple', linestyle='--', linewidth=2,
             marker='d', markersize=3, alpha=0.8)
axes[1].set_title('Test Period: Detailed Model Comparison', 
                  fontsize=14, fontweight='bold')
axes[1].set_ylabel('Billions', fontsize=12)
axes[1].legend(loc='best', fontsize=9)
axes[1].grid(True, alpha=0.3)

# Plot 3: Forecast Error Comparison
sarimax_errors= sarimax_actuals - sarimax_forecasts

axes[2].plot(test.index, arima_errors,
             label='ARIMA Error', linewidth=2, color='darkorange', 
             marker= 'o', markersize = 3, alpha=0.7)
axes[2].plot(test.index, sarima_errors,
             label='SARIMA Error', linewidth=2, color='green',
             marker='s', markersize=3, alpha=0.7)
axes[2].plot(test.index, sarimax_errors,
             label='SARIMAX Error', linewidth=2, color='purple',
             marker='d', markersize=3, alpha=0.8)
axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[2].set_title('Forecast Error: Model Comparison', 
                  fontsize=14, fontweight='bold')
axes[2].set_ylabel('Error (Billions)', fontsize=12)
axes[2].legend(loc='best', fontsize=9)
axes[2].grid(True, alpha=0.3)

# Plot 4: Absolute Error Comparison
axes[3].bar(np.arange(len(test)) - 0.2, np.abs(arima_errors), width=0.2,
            label='ARIMA Abs Error', color='darkorange', alpha=0.7)
axes[3].bar(np.arange(len(test)) , np.abs(sarima_errors), width=0.2,
            label='SARIMA Abs Error', color='green', alpha=0.7)
axes[3].bar(np.arange(len(test)) + 0.2, np.abs(sarimax_errors), width=0.2,
            label='SARIMAX Abs Error', color='purple', alpha=0.7)
axes[3].set_title('Absolute Forecast Error: Model Comparison', 
                  fontsize=14, fontweight='bold')
axes[3].set_xlabel('Forecast Period', fontsize=12)
axes[3].set_ylabel('Absolute Error(Billions)', fontsize=12)
axes[3].legend(loc='best', fontsize=9)
axes[3].grid(True, alpha=0.3)

# Make date labels readable
fig.autofmt_xdate(rotation=45, ha='right')
plt.subplots_adjust(
    left=0.08,
    right=0.92,
    top=0.96,
    bottom=0.10,
    hspace=0.70

)
plt.show()

# -------------------------------------------------------------------------------------
# SAVE SARIMAX RESULTS
#---------------------------------------------------------------------------------------------------

# Save forecast results
results_sarimax_df = pd.DataFrame({
    'Date': test.index,
    'Actual': sarimax_actuals,
    'ARIMA_Forecast': arima_forecasts,
    'SARIMA_Forecast': sarima_forecasts,
    'SARIMAX_Forecast': sarimax_forecasts,
    'ARIMA_Error': arima_errors,
    'SARIMA_Error': sarima_errors,
    'SARIMAX_Error': sarimax_errors,
    'ARIMA_Abs_Error': np.abs(arima_errors),
    'SARIMA_Abs_Error': np.abs(sarima_errors),
    'SARIMAX_Abs_Error': np.abs(sarimax_errors),
    'FedFunds_Rate': x_test['FedFunds_Rate'].values,
    'Unemployment_Rate': x_test['Unemployment_Rate'].values
})

results_sarimax_df.to_csv('sarimax_full_comparison_results.csv', index=False)
print("\nResults saved to 'sarimax_full_comparison_results.csv'")

# Save model summary
with open('sarimax_model_summary.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("SARIMAX MODEL RESULTS WITH EXOGENOUS VARIABLES\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Best SARIMAX Model: {best_order_sarimax}{best_seasonal_order_sarimax}\n")
    f.write(f"Exogenous Variables: Federal Funds Rate, Unemployment Rate\n\n")
    f.write(f"AIC: {best_aic_sarimax:.2f}\n")
    f.write(f"BIC: {best_bic_sarimax:.2f}\n\n")
    f.write("Forecast Performance:\n")
    f.write(f"  RMSE: {rmse_sarimax:.4f}\n")
    f.write(f"  MAE: {mae_sarimax:.4f}\n")
    f.write(f"  MAPE: {mape_sarimax:.4f}%\n\n")
    f.write("=" * 80 + "\n")
    f.write("COMPARISON WITH OTHER MODELS:\n")
    f.write("=" * 80 + "\n")
    f.write(f"ARIMA RMSE: {rmse:.4f}\n")
    f.write(f"SARIMA RMSE: {rmse_sarima:.4f}\n")
    f.write(f"SARIMAX RMSE: {rmse_sarimax:.4f}\n\n")
    f.write(f"Improvement over ARIMA: {rmse_improvement_vs_arima:.2f}%\n")
    f.write(f"Improvement over SARIMA: {rmse_improvement_vs_sarima:.2f}%\n\n")
    f.write(f"Overall Best Model: {best_model_overall}\n")
    f.write(f"Model Scores:\nARIMA: {model_scores['ARIMA']}/4, \nSARIMA: {model_scores['SARIMA']}/4, \nSARIMAX: {model_scores['SARIMAX']}/4\n\n")
    f.write("=" * 80 + "\n")
    f.write("Full Model Summary:\n")
    f.write("=" * 80 + "\n")
    f.write(str(best_model_sarimax.summary()))

print("✓ Model summary saved to 'sarimax_model_summary.txt'")

print("\n" + "=" * 80)
print("SARIMAX ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  1. sarimax_full_comparison_results.csv - Detailed forecast data")
print("  2. sarimax_model_summary.txt - Model summary and metrics")
print("\nKey Findings:")
print(f"  - Best performing model: {best_model_overall}")
print(f"  - ARIMA RMSE: {rmse:.4f}")
print(f"  - SARIMA RMSE: {rmse_sarima:.4f}")
print(f"  - SARIMAX RMSE: {rmse_sarimax:.4f}")
print(f"  - SARIMAX includes Fed Funds Rate & Unemployment Rate as predictors")