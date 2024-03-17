import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

# Importing the dataset
file_path = r"E:\Downloads\transaction_anomalies_dataset.csv"

data = pd.read_csv(file_path)

# Checking for null values
null_values = data.isnull().sum()
print("Null values:\n", null_values)

# Checking column info
print("\nColumn Info:")
print(data.info())

# Descriptive statistics
print("\nDescriptive Statistics:")
print(data.describe())


# Check distribution of transaction amounts by account type
data.boxplot(column='Transaction_Amount', by='Account_Type')
plt.title('Distribution of Transaction Amounts by Account Type')
plt.xlabel('Account Type')
plt.ylabel('Transaction Amount')
plt.show()



# Grouping data by age and calculating average transaction amount
average_transaction_by_age = data.groupby('Age')['Transaction_Amount'].mean()
average_transaction_by_age.plot(kind='line', marker='o')
plt.title('Average Transaction Amount by Age')
plt.xlabel('Age')
plt.ylabel('Average Transaction Amount')
plt.grid(True)
plt.show()



# Counting transactions by day of the week
transactions_by_day_of_week = data['Day_of_Week'].value_counts()

# Plotting the count of transactions by day of the week
transactions_by_day_of_week.plot(kind='bar', color='skyblue')
plt.title('Count of Transactions by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Transactions')

plt.grid(axis='y', linestyle='-', alpha=0.7)

plt.show()


# Selecting only numeric columns
numeric_data = data.select_dtypes(include='number')

# Compute the correlation matrix
correlation_matrix = numeric_data.corr()

# Plotting the correlation matrix as a heatmap
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f")
plt.title('Correlation Matrix (Numeric Columns Only)')
plt.show()


sns.boxplot(x='Transaction_Amount', data=data, color='skyblue')
plt.title('Box Plot of Transaction Amount')
plt.xlabel('Transaction Amount')
plt.show()

# Calculate mean and standard deviation
mean_transaction_amount = data['Transaction_Amount'].mean()
std_transaction_amount = data['Transaction_Amount'].std()

# Define the range for anomalies (e.g., 3 standard deviations from the mean)
lower_bound = mean_transaction_amount - 3 * std_transaction_amount
upper_bound = mean_transaction_amount + 3 * std_transaction_amount

# Count the number of anomalies
anomalies_count = ((data['Transaction_Amount'] < lower_bound) | (data['Transaction_Amount'] > upper_bound)).sum()

# Calculate the ratio of anomalies in the data
total_samples = len(data)
ratio_anomalies = anomalies_count / total_samples

print("Number of anomalies:", anomalies_count)
print("Ratio of anomalies in the data:", ratio_anomalies)


# Selecting relevant features
features = data[['Transaction_Amount', 'Frequency_of_Transactions', 'Time_Since_Last_Transaction']]

# Fitting the Isolation Forest model
isolation_forest = IsolationForest(contamination=0.1)  # Adjust contamination as needed
isolation_forest.fit(features)

# Getting predictions
predictions = isolation_forest.predict(features)

# Converting predictions to binary values
binary_predictions = [1 if pred == -1 else 0 for pred in predictions]

# Adding binary predictions to the DataFrame
data['Anomaly'] = binary_predictions

# Displaying the DataFrame with binary predictions
print(data.head())


# Selecting relevant features
features = data[['Transaction_Amount', 'Frequency_of_Transactions', 'Time_Since_Last_Transaction']]

# Fitting the Isolation Forest model
isolation_forest = IsolationForest(contamination=0.1)  # Adjust contamination as needed
predictions = isolation_forest.fit_predict(features)

# Generating classification report
classification_rep = classification_report(predictions, predictions)

print("Classification Report:")
print(classification_rep)

# Selecting relevant features
features = data[['Transaction_Amount', 'Frequency_of_Transactions', 'Time_Since_Last_Transaction']]

# Fitting the Isolation Forest model
isolation_forest = IsolationForest(contamination=0.1)  # Adjust contamination as needed
predictions = isolation_forest.fit_predict(features)

# Adding the predictions to the DataFrame
data['Anomaly'] = ['Anomaly' if pred == -1 else 'Normal' for pred in predictions]

# Displaying the result
print(data)