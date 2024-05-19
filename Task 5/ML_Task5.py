import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the dataset
path = r'C:\\Users\\Zaryab\\OneDrive\\Documents\\GitHub\\AICP-internship\\Instagram-reach.csv'
df = pd.read_csv(path)

# Check for null values and column information
print(df.isnull().sum())
print(df.info())

# Descriptive statistics
print(df.describe())

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Analyze the trend of Instagram reach over time using a line chart
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Instagram reach'], marker='o')
plt.title('Trend of Instagram Reach Over Time')
plt.xlabel('Date')
plt.ylabel('Instagram Reach')
plt.grid(True)
plt.show()

# Analyze Instagram reach for each day using a bar chart
daily_reach = df.groupby(df['Date'].dt.date)['Instagram reach'].sum()
plt.figure(figsize=(12, 6))
daily_reach.plot(kind='bar')
plt.title('Instagram Reach for Each Day')
plt.xlabel('Date')
plt.ylabel('Instagram Reach')
plt.show()

# Analyze the distribution of Instagram reach using a box plot
plt.figure(figsize=(8, 6))
plt.boxplot(df['Instagram reach'])
plt.title('Distribution of Instagram Reach')
plt.ylabel('Instagram Reach')
plt.show()

# Create a day column and analyze reach based on the days of the week
df['Day'] = df['Date'].dt.day_name()
day_stats = df.groupby('Day')['Instagram reach'].agg(['mean', 'median', 'std'])
print(day_stats)

# Create a bar chart to visualize the reach for each day of the week
avg_reach_per_day = df.groupby('Day')['Instagram reach'].mean()
avg_reach_per_day = avg_reach_per_day.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.figure(figsize=(10, 6))
avg_reach_per_day.plot(kind='bar')
plt.title('Average Instagram Reach by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Instagram Reach')
plt.show()

# Check the trends and seasonal patterns of Instagram reach
df.set_index('Date', inplace=True)
decomposition = seasonal_decompose(df['Instagram reach'], model='additive')
decomposition.plot()
plt.show()

# Determine p, d, q values for SARIMA model
plot_acf(df['Instagram reach'])
plot_pacf(df['Instagram reach'])
plt.show()

# Define the SARIMA model (example values for p, d, q, P, D, Q, S)
p, d, q = 1, 1, 1
P, D, Q, S = 1, 1, 1, 12

sarima_model = SARIMAX(df['Instagram reach'], order=(p, d, q), seasonal_order=(P, D, Q, S))
sarima_result = sarima_model.fit()

# Summary of the model
print(sarima_result.summary())

# Forecasting future values
forecast = sarima_result.get_forecast(steps=30)
forecast_df = forecast.conf_int()
forecast_df['Forecast'] = forecast.predicted_mean

# Plotting the forecast
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Instagram reach'], label='Historical')
plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='red')
plt.fill_between(forecast_df.index, forecast_df.iloc[:, 0], forecast_df.iloc[:, 1], color='pink')
plt.title('Instagram Reach Forecast')
plt.xlabel('Date')
plt.ylabel('Instagram Reach')
plt.legend()
plt.show()
