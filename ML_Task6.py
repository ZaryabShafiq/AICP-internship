import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Step 1: Load the dataset
path = r'C:\\Users\\Zaryab\\OneDrive\\Documents\\GitHub\\AICP-internship\\user_profiles_for_ads.csv'
df = pd.read_csv(path)
# Step 2: Check Null Values, Column Info, and Descriptive Statistics
print(df.isnull().sum())
print(df.info())
print(df.describe())

# Step 3: Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'])
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

device_counts = df['Device Usage'].value_counts()
print(device_counts)

plt.figure(figsize=(10, 6))
sns.countplot(x='Device Usage', data=df)
plt.title('Device Usage Distribution')
plt.xlabel('Device Usage')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Likes and Reactions', y='Followed Accounts', data=df)
plt.title('Likes and Reactions vs Followed Accounts')
plt.xlabel('Likes and Reactions')
plt.ylabel('Followed Accounts')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Time Spent Online (hrs/weekday)', y='Time Spent Online (hrs/weekend)', data=df)
plt.title('Time Spent Online (Weekday vs Weekend)')
plt.xlabel('Time Spent Online (Weekday)')
plt.ylabel('Time Spent Online (Weekend)')
plt.show()

ad_metrics = df[['Click-Through Rates (CTR)', 'Conversion Rates', 'Ad Interaction Time (sec)']]
print(ad_metrics.describe())

common_interests = df['Top Interests'].value_counts().head(10)
print(common_interests)

# Step 7: User Segmentation (Clustering)
age_mapping = {'18-24': 21, '25-34': 29.5, '35-44': 39.5, '45-54': 49.5, '55-64': 59.5, '65+': 70}
df['Age'] = df['Age'].map(age_mapping)

numeric_columns = df.select_dtypes(include=np.number).columns
features = df[numeric_columns]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Step 8: Cluster Analysis and Naming
cluster_analysis = df.groupby('Cluster').mean()

print(cluster_analysis)

# Step 9: Visualization
features = numeric_columns

for cluster in range(5):
    cluster_data = cluster_analysis.loc[cluster, features].tolist()
    labels = np.array(features)
    stats = cluster_data
    
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    stats = np.concatenate((stats,[stats[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, stats, color='red', alpha=0.25)
    ax.plot(angles, stats, color='red', linewidth=2)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    plt.title(f'Cluster {cluster}')
    
    plt.show()

# Step 10: Summary
print("Summary of User Profiling and Segmentation Analysis:")
print("---------------------------------------------------")
print("1. Exploratory Data Analysis (EDA) provided insights into demographic variables, device usage, and user behavior.")
print("2. Common interests among users were identified.")
print("3. User segmentation was performed using KMeans clustering, resulting in five distinct user segments.")
print("4. Radar charts were created to visually represent each segment's profile.")