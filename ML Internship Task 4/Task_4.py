# Step 1: Import data and check null values, column info, and descriptive statistics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


# Importing data
df = pd.read_csv(r"C:\Users\Zaryab\OneDrive\Documents\GitHub\ML Internship Task 4\userbehaviour.csv")

# Check for null values
print("Null values:\n", df.isnull().sum())

# Column info
print("\nColumn info:\n", df.info())

# Descriptive statistics
print("\nDescriptive statistics:\n", df.describe())

# Step 2: Analyze screen time of users
print("\nHighest Screen Time:", df['Average Screen Time'].max())
print("Lowest Screen Time:", df['Average Screen Time'].min())
print("Average Screen Time:", df['Average Screen Time'].mean())

# Step 3: Analyze amount spent by users
print("\nHighest Amount Spent:", df['Average Spent on App (INR)'].max())
print("Lowest Amount Spent:", df['Average Spent on App (INR)'].min())
print("Average Amount Spent:", df['Average Spent on App (INR)'].mean())

# Step 4: Investigate relationship between spending capacity and screen time for active users
active_users = df[df['Status'] == 'Installed']
uninstalled_users = df[df['Status'] == 'Uninstalled']

# Scatter plot for active users
plt.scatter(active_users['Average Spent on App (INR)'], active_users['Average Screen Time'], color='blue', label='Active Users')
# Scatter plot for uninstalled users
plt.scatter(uninstalled_users['Average Spent on App (INR)'], uninstalled_users['Average Screen Time'], color='red', label='Uninstalled Users')
plt.xlabel('Average Spent on App (INR)')
plt.ylabel('Average Screen Time')
plt.title('Relationship between Spending Capacity and Screen Time')
plt.legend()
plt.show()

# Step 5: Examine relationship between user ratings and average screen time
rating_vs_screen_time = df.groupby('Ratings')['Average Screen Time'].mean()
print("\nRelationship between Ratings and Average Screen Time:\n", rating_vs_screen_time)

plt.scatter(df['Ratings'], df['Average Screen Time'])
plt.xlabel('Ratings')
plt.ylabel('Average Screen Time')
plt.title('Relationship between User Ratings and Average Screen Time')
plt.show()

# Selecting features for segmentation
features_for_segmentation = df[['Average Screen Time', 'Average Spent on App (INR)', 'Ratings']]

# Perform K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
df['Segment'] = kmeans.fit_predict(features_for_segmentation)

# Visualize segments
plt.scatter(df['Average Screen Time'], df['Average Spent on App (INR)'], c=df['Segment'], cmap='viridis')
plt.xlabel('Average Screen Time')
plt.ylabel('Average Spent on App (INR)')
plt.title('App User Segmentation')
plt.colorbar(label='Segment')
plt.show()
print("Number of Segments:", df['Segment'].nunique())


# Step 6: Check correlation between different metrics
numeric_cols = ['Average Screen Time', 'Average Spent on App (INR)', 'New Password Request', 'Last Visited Minutes']
numeric_df = df[numeric_cols]

# Compute correlation matrix
correlation_matrix = numeric_df.corr()

# Plot heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Step 7: Detect anomalies using Isolation Forest algorithm

features = df[['Average Screen Time', 'Average Spent on App (INR)', 'Last Visited Minutes']]
clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(features)
df['Anomaly'] = clf.predict(features)

# Visualize anomalies
plt.scatter(df.index, df['Average Screen Time'], c=df['Anomaly'], cmap='viridis')
plt.xlabel('Index')
plt.ylabel('Average Screen Time')
plt.title('Anomaly Detection')
plt.colorbar(label='Anomaly')
plt.show()
