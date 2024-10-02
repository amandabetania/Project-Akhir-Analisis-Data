import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st

st.write("Nama  : Amanda Betania Maritza")
st.write("email : amandamaritza2004@gmail.com")
st.write("ID Dicoding : amandabetaniamaritza")

# Using Streamlit to describe the application
st.title("Dashboard E-Commerce Analysis")

# Path to dataset folder
dataset_path = './main_data.csv'  # Adjust to your path

# Load data
df = pd.read_csv(dataset_path)

# Data Cleaning
df['product_id'] = df['product_id'].astype(str)
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'], errors='coerce')
default_date = pd.to_datetime('2000-01-01')
df['order_purchase_timestamp'] = df['order_purchase_timestamp'].fillna(default_date)
df.drop_duplicates(inplace=True)

# Display cleaned data
st.write("Data after cleaning:")
st.dataframe(df.head())

# Calculate total sales
df['penjualan'] = df['order_item_id'] * df['price']
df['month'] = df['order_purchase_timestamp'].dt.to_period('M')
penjualan_bulanan = df.groupby(['month']).agg({'penjualan': 'sum'}).reset_index()

# Visualize Total Monthly Sales
st.subheader('Total Penjualan Bulanan')
fig, ax = plt.subplots()
plt.plot(penjualan_bulanan['month'].astype(str), penjualan_bulanan['penjualan'], marker='o')
plt.title('Tren Penjualan Bulanan (Skala Logaritmik)')
plt.xlabel('Bulan')
plt.ylabel('Total Penjualan (IDR)')
plt.xticks(rotation=45)
plt.yscale('log')
plt.grid()
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
st.pyplot(fig)

# Calculate average rating
total_penjualan = df.groupby('order_id').agg({'penjualan': 'sum'}).reset_index()
average_rating = df.groupby('order_id').agg({'review_score': 'mean'}).reset_index()

# Combine data
correlation_data = pd.merge(total_penjualan, average_rating, on='order_id', how='inner')
correlation_data = correlation_data.rename(columns={'penjualan': 'Total Penjualan', 'review_score': 'Rata-rata Rating'})

# Correlation Heatmap
st.subheader('Korelasi antara Total Penjualan dan Rata-rata Rating')
correlation_matrix = correlation_data[['Total Penjualan', 'Rata-rata Rating']].corr()
fig, ax = plt.subplots()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8}, ax=ax)
plt.title('Korelasi antara Total Penjualan dan Rata-rata Rating')
st.pyplot(fig)

# Scatter Plot
st.subheader('Scatter Plot: Total Penjualan vs Rata-rata Rating')
fig, ax = plt.subplots()
sns.scatterplot(data=correlation_data, x='Total Penjualan', y='Rata-rata Rating', ax=ax)
plt.title('Scatter Plot: Total Penjualan vs Rata-rata Rating')
plt.xlabel('Total Penjualan (IDR)')
plt.ylabel('Rata-rata Rating')
plt.grid()
st.pyplot(fig)

# RFM Analysis
snapshot_date = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
rfm_df = df.groupby('customer_id').agg({
    'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
    'order_id': 'count',
    'penjualan': 'sum'
}).rename(columns={'order_purchase_timestamp': 'Recency', 'order_id': 'Frequency', 'penjualan': 'Monetary'})

# Clustering
bins_recency = [0, 30, 60, 90, 180, 365, 1000]
labels_recency = ['Very Recent', 'Recent', 'Moderate', 'Old', 'Very Old', 'Inactive']
rfm_df['Recency Group'] = pd.cut(rfm_df['Recency'], bins=bins_recency, labels=labels_recency)

# Display RFM Analysis Results
st.subheader("Hasil Analisis RFM")
st.dataframe(rfm_df.head())

st.subheader("Hasil Klustering RFM")
st.dataframe(rfm_df[['Recency', 'Frequency', 'Monetary', 'Recency Group']].head())
