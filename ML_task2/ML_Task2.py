import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the data
tips_df = pd.read_csv('C:\\Users\\Zaryab\\OneDrive\\Desktop\\ML_task2\\tips.csv')

# Q.1: Import data and check null values, check column info and the descriptive statistics of the data.
print("Null values:\n", tips_df.isnull().sum())
print("\nColumn info:\n", tips_df.info())
print("\nDescriptive statistics:\n", tips_df.describe())

# Q.2: Tips given according to total bill, number of people at a table, and day of the week
fig1 = px.scatter(tips_df, x='total_bill', y='tip', color='size', facet_row='day', title='Tips vs Total Bill, Grouped by Party Size and Day')
fig1.show()

# Q.3: Tips given according to total bill, number of people at a table, and gender of person paying
fig2 = px.scatter(tips_df, x='total_bill', y='tip', color='sex', facet_row='size', title='Tips vs Total Bill, Grouped by Party Size and Gender')
fig2.show()

# Q.4: Tips given according to total bill, number of people at a table, and time of meal
fig3 = px.scatter(tips_df, x='total_bill', y='tip', color='time', facet_row='size', title='Tips vs Total Bill, Grouped by Party Size and Time')
fig3.show()

# Q.5: Tips given according to day of the week
tips_by_day = tips_df.groupby('day')['tip'].sum().reset_index()
fig4 = px.bar(tips_by_day, x='day', y='tip', title='Total Tips Given by Day')
fig4.show()

# Q.6: Tips given according to gender of person paying the bill
tips_by_gender = tips_df.groupby('sex')['tip'].sum().reset_index()
fig5 = px.bar(tips_by_gender, x='sex', y='tip', title='Total Tips Given by Gender')
fig5.show()

# Q.7: Tips given according to day of the week
fig6 = px.box(tips_df, x='day', y='tip', title='Tips Distribution by Day')
fig6.show()

# Q.8: Tips given by smokers vs non-smokers
tips_by_smoker = tips_df.groupby('smoker')['tip'].sum().reset_index()
fig7 = px.bar(tips_by_smoker, x='smoker', y='tip', title='Total Tips Given by Smoking Status')
fig7.show()

# Q.9: Tips given during lunch vs dinner
tips_by_time = tips_df.groupby('time')['tip'].sum().reset_index()
fig8 = px.bar(tips_by_time, x='time', y='tip', title='Total Tips Given by Meal Time')
fig8.show()

# Data transformation - converting categorical values into numerical values using pd.get_dummies
tips_df = pd.get_dummies(tips_df, columns=['day', 'sex', 'smoker', 'time'], drop_first=True)

# Q.10: Split data into training and test sets, train Linear Regression model
X = tips_df.drop(columns=['tip'])
y = tips_df['tip']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Q.11: Check model prediction
sample_input = X_test.head(1)  # Example input from the test set
predicted_tip = model.predict(sample_input)
print("Predicted tip:", predicted_tip[0])
