#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import statsmodels.api as sm


# In[6]:


top_crypto_list = ['BTC-GBP', 'ETH-GBP', 'USDT-GBP', 'BNB-GBP', 'SOL-GBP', 'XRP-GBP', 'USDC-GBP', 'ADA-GBP', 'DOGE-GBP', 'SHIB-GBP', 'AVAX-GBP', 'DOT-GBP', 'MATIC-GBP', 'TRX-GBP', 'LINK-GBP', 'UNI7083-GBP', 'BCH-GBP', 'LTC-GBP', 'ICP-GBP', 'FIL-GBP', 'DAI-GBP', 'ETC-GBP', 'ATOM-GBP', 'CRO-GBP', 'STX4847-GBP', 'HBAR-GBP', 'GRT6719-GBP', 'XLM-GBP', 'VET-GBP', 'THETA-GBP']


# In[7]:


def download_crypto_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)


symbol = top_crypto_list
start_date = "2023-03-01"
end_date = "2024-03-01"


data = yf.download(symbol, start=start_date, end=end_date)


# In[8]:


data


# In[9]:


# Save to a working directory
data.to_csv('file1.csv')


# In[10]:


# Extract adjusted close
df = data['Adj Close']
df


# In[11]:


# Transpose to have crypto tickers as rows
df_t = df.T
df_t


# In[12]:


index_column = df_t.index
tickers_df = pd.DataFrame(index_column)
tickers_df


# # PCA

# In[13]:


'''
# Initialize and fit the scaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_numeric)
scaled_data
'''


# In[14]:


# normalize the data
normalized_data = normalize(df_t, axis=1)
normalized_data


# In[15]:


#principal component analysis
pca = PCA(n_components=10) 

# Fit the data and transform it
pca_data = pca.fit_transform(normalized_data)

pca_df = pd.DataFrame(pca_data)

#reattach tickers
pca_n_symbol_df = pd.concat([tickers_df, pca_df], axis=1)
pca_n_symbol_df


# # Clustering

# In[16]:


kmeans = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit_predict(normalized_data)
pca_n_symbol_df['cluster'] = clusters
pca_n_symbol_df


# In[17]:


'''
Agglomerative Clustering
agglomerative = AgglomerativeClustering(n_clusters=4)

# Fit the clustering model and predict clusters
clusters = agglomerative.fit_predict(normalized_data)

# Add clusters to the DataFrame
pca_n_symbol_df['cluster'] = clusters

# Print the DataFrame with cluster assignments
pca_n_symbol_df
'''


# In[18]:


'''
# Example using DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=10)  # Adjust parameters as needed

# Fit the clustering model and predict clusters
clusters = dbscan.fit_predict(normalized_data)

# Add clusters to the DataFrame
pca_n_symbol_df['cluster'] = clusters

# Print the DataFrame with cluster assignments
pca_n_symbol_df
'''


# In[19]:


my_tickers = ['DOGE-GBP', 'SOL-GBP', 'USDC-GBP', 'ETH-GBP' ]


# # Correlation

# In[20]:


def analyze_correlation(selected_ticker, df):
    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Extract correlations for the selected ticker
    selected_correlations = correlation_matrix[selected_ticker].drop(selected_ticker)

    # List the top positively correlated tickers
    top_positively_correlated = selected_correlations.nlargest(4)

    # List the top negatively correlated tickers
    top_negatively_correlated = selected_correlations.nsmallest(4)

    print(f"Top 4 positively correlated tickers to {selected_ticker}:\n{top_positively_correlated}")
    print(f"\nTop 4 negatively correlated tickers to {selected_ticker}:\n{top_negatively_correlated}")

    # Select top positively and negatively correlated tickers
    top_correlated_tickers = top_positively_correlated.index.union(top_negatively_correlated.index)

    # Add the selected ticker to the top correlated tickers
    top_correlated_tickers = top_correlated_tickers.union([selected_ticker])

    # Filter the correlation matrix to include only top correlated tickers
    top_correlation_matrix = correlation_matrix.loc[top_correlated_tickers, top_correlated_tickers]

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(top_correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Top Correlation Heatmap of Cryptocurrency Prices for {selected_ticker}')
    plt.xlabel('Ticker')
    plt.ylabel('Ticker')
    plt.show()


# In[21]:


# Apply the function to each ticker
for ticker in my_tickers:
    analyze_correlation(ticker, df)


# # EDA

# In[22]:


def plot_time_series(df, tickers):
    fig = go.Figure()
    for ticker in tickers:
        fig.add_trace(go.Scatter(x=df.index, y=df[ticker], mode='lines', name=ticker))
    fig.update_layout(title='Time Series Graph of Cryptocurrency Prices', xaxis_title='Date', yaxis_title='Price')
    fig.show()
    
plot_time_series(df, my_tickers)


# In[23]:


"""
def plot_time_series(df, tickers):
plt.figure(figsize=(12, 8))
for i, ticker in enumerate(tickers, 1):
    plt.subplot(len(tickers), 1, i)
    plt.plot(df.index, df[ticker], label=ticker)
    plt.title(f'Time Series Graph of {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
plt.tight_layout()
plt.show()

plot_time_series(df, my_tickers)
"""


# In[24]:


def plot_box_plots(df, tickers, num_rows):
    fig = make_subplots(rows=num_rows, cols=len(tickers)//num_rows, subplot_titles=tickers)
    for i, ticker in enumerate(tickers, 1):
        row = (i - 1) // (len(tickers) // num_rows) + 1
        col = (i - 1) % (len(tickers) // num_rows) + 1
        fig.add_trace(go.Box(y=df[ticker], name=ticker), row=row, col=col)
    fig.update_layout(title='Box Plot of Cryptocurrency Prices', height=500*num_rows, showlegend=False)
    fig.show()
    
plot_box_plots(df, my_tickers, num_rows=2) 


# In[25]:


def plot_histograms(df, tickers, num_rows):
    fig = make_subplots(rows=num_rows, cols=len(tickers)//num_rows, subplot_titles=tickers)
    for i, ticker in enumerate(tickers, 1):
        row = (i - 1) // (len(tickers) // num_rows) + 1
        col = (i - 1) % (len(tickers) // num_rows) + 1
        fig.add_trace(go.Histogram(x=df[ticker], name=ticker, nbinsx=20), row=row, col=col)
    fig.update_layout(title='Histogram of Cryptocurrency Prices', height=500*num_rows, showlegend=False)
    fig.show()
    
plot_histograms(df, my_tickers, num_rows=2)


# In[26]:


def plot_histograms(df, tickers, num_rows):
    fig, axes = plt.subplots(num_rows, len(tickers)//num_rows, figsize=(15, 5*num_rows))
    for i, ticker in enumerate(tickers, 1):
        row = (i - 1) // (len(tickers) // num_rows)
        col = (i - 1) % (len(tickers) // num_rows)
        # Calculate mean and standard deviation
        mean = df[ticker].mean()
        std = df[ticker].std()
        # Plot histogram
        ax = axes[row, col]
        sns.histplot(df[ticker], kde=True, ax=ax)
        ax.set_title(ticker)
        # Plot distribution line
        x_values = np.linspace(df[ticker].min(), df[ticker].max(), 100)
        y_values = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-(x_values - mean)**2 / (2 * std**2))
        ax.plot(x_values, y_values, color='r', label='Distribution')
        ax.legend()
    plt.tight_layout()
    plt.show()
    
plot_histograms(df, my_tickers, num_rows=2)


# In[28]:


def plot_time_series_decomposition(df, ticker):
    # Extract the time series data for the selected ticker
    ts = df[ticker]
    
    # Decompose the time series into trend, seasonal, and residual components
    decomposition = sm.tsa.seasonal_decompose(ts, model='additive')
    
    # Plot the decomposition components
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    ts.plot(ax=axes[0], title='Original Time Series')
    decomposition.trend.plot(ax=axes[1], title='Trend Component')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal Component')
    decomposition.resid.plot(ax=axes[3], title='Residual Component')
    plt.tight_layout()
    plt.show()

# Time series decomposition for a specific ticker
plot_time_series_decomposition(df, 'ETH-GBP')


# In[57]:


# Pair plots
def plot_pair_plots(df, tickers):
    # Select the relevant columns from the DataFrame
    df_subset = df[tickers]
    
    # Create pair plots
    sns.pairplot(df_subset)
    plt.suptitle('Pair Plot of Cryptocurrency Prices', y=1.02)
    plt.show()
    
plot_pair_plots(df, my_tickers)


# In[62]:


# Select the 'Adj Close' columns for each ticker
adj_close_cols = [('Adj Close', ticker) for ticker in my_tickers]

# Create a new DataFrame with only the 'Adj Close' columns
df_adj_close = data[adj_close_cols]

# Plot candlestick charts for each ticker
fig = go.Figure()
for ticker in my_tickers:
    df_ticker = df_adj_close[('Adj Close', ticker)]
    fig.add_trace(go.Candlestick(x=df_ticker.index,
                                 open=df_ticker,
                                 high=df_ticker,
                                 low=df_ticker,
                                 close=df_ticker,
                                 increasing_line_color='green',  # Color for increasing candlesticks
                                 decreasing_line_color='red',   # Color for decreasing candlesticks
                                 line=dict(width=1)))           # Adjust the width of the candlesticks

fig.update_layout(title='Candlestick Charts for Selected Cryptocurrencies',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  xaxis_rangeslider_visible=False)

fig.show()


# In[ ]:




