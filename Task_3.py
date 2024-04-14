import pandas as pd
import plotly.express as px

# Importing data
df = pd.read_csv(r"C:\Users\Zaryab\OneDrive\Documents\GitHub\ML_T3\Queries.csv")

# Analyzing the top queries by clicks and impressions
def visualize_top_queries(df, metric):
    top_queries = df.sort_values(by=metric, ascending=False).head(10)
    fig = px.bar(top_queries, x='Top queries', y=metric,
                 title=f'Top 10 Queries by {metric}')
    fig.show()

# Analyzing the queries with the highest and lowest CTRs
def visualize_ctr(df, top=True):
    if top:
        queries_ctr = df.sort_values(by='CTR', ascending=False).head(10)
        title = 'Top 10 Queries by CTR'
    else:
        queries_ctr = df.sort_values(by='CTR').head(10)
        title = 'Bottom 10 Queries by CTR'
    fig = px.bar(queries_ctr, x='Top queries', y='CTR', title=title)
    fig.show()

# Checking the correlation between different metrics
def visualize_correlation(df):
    corr_matrix = df.corr()
    fig = px.imshow(corr_matrix, labels=dict(x="Metrics", y="Metrics", color="Correlation"),
                    x=corr_matrix.columns, y=corr_matrix.columns, color_continuous_scale='RdBu_r')
    fig.show()

# Visualizing top queries by clicks and impressions
visualize_top_queries(df, 'Clicks')
visualize_top_queries(df, 'Impressions')

# Visualizing queries with highest and lowest CTRs
visualize_ctr(df, top=True)
visualize_ctr(df, top=False)

# Visualizing correlation between different metrics
visualize_correlation(df[['Clicks', 'Impressions', 'CTR', 'Position']])
