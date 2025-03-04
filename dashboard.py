import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Memuat dataset yang digunakan.
day_df = pd.read_csv("data/day_data.csv")
hour_df = pd.read_csv("data/hour_data.csv")

# Konversi ke datetime
day_df["dteday"] = pd.to_datetime(day_df["dteday"])
hour_df["dteday"] = pd.to_datetime(hour_df["dteday"])

# Mengecek rentang tahun yang tersedia
min_date = day_df["dteday"].min()  
max_date = day_df["dteday"].max()  

# Menampilkan informasi di sidebar
st.sidebar.write(f"📅 Data tersedia dari **{min_date.year}** hingga **{max_date.year}**")

# Sidebar untuk filter rentang waktu
st.sidebar.header("Filter Rentang Waktu")
date_range = st.sidebar.date_input(
    "Pilih rentang waktu", 
    [min_date, max_date], 
    min_value=min_date, 
    max_value=max_date
)

# Memaastikan date_range valid sebelum filtering
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = map(pd.to_datetime, date_range)
    filtered_day_df = day_df[(day_df["dteday"] >= start_date) & (day_df["dteday"] <= end_date)]
    filtered_hour_df = hour_df[(hour_df["dteday"] >= start_date) & (hour_df["dteday"] <= end_date)]
else:
    filtered_day_df = day_df
    filtered_hour_df = hour_df

# Judul dashboard
st.title("Bike Sharing Dashboard")

# Visualisasi Distribusi Penyewaan Sepeda Harian
st.subheader("Distribusi Jumlah Penyewaan Sepeda Harian")
fig, ax = plt.subplots(figsize=(12, 6))
sns.histplot(filtered_day_df['cnt'], bins=30, kde=True, ax=ax, color='red')
ax.set_title("Distribusi Jumlah Penyewaan Sepeda Harian")
st.pyplot(fig)

# Visualisasi Hubungan antara Suhu dan Jumlah Penyewaan
st.subheader("Hubungan antara Suhu dan Jumlah Penyewaan Sepeda")
fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(x=filtered_day_df['temp'], y=filtered_day_df['cnt'], alpha=0.3, ax=ax, color='blue')
ax.set_xlabel("Suhu")
ax.set_ylabel("Jumlah Penyewaan")
ax.set_title("Hubungan antara Suhu dan Jumlah Penyewaan Sepeda")
st.pyplot(fig)

# Visualisasi Tren Penyewaan Sepeda sepanjang Hari
st.subheader("Tren Penyewaan Sepeda sepanjang Hari")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x=filtered_hour_df['hr'], y=filtered_hour_df['cnt'], ci=None, ax=ax, color='green')
ax.set_xlabel("Jam")
ax.set_ylabel("Jumlah Penyewaan")
ax.set_title("Tren Penyewaan Sepeda sepanjang Hari")
st.pyplot(fig)

# Customer Segmentation dengan K-Means
st.subheader("Customer Segmentation dengan K-Means")

# Pilih Fitur yang Relevan
features = filtered_day_df[['cnt', 'casual', 'registered']]

# Standarisasi Data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Menentukan Jumlah Cluster dengan Metode Elbow
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)

# Visualisasi Elbow Method
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_range, inertia, marker='o', linestyle='--')
ax.set_xlabel('Jumlah Cluster')
ax.set_ylabel('Inertia')
ax.set_title('Metode Elbow untuk Menentukan K Optimal')
st.pyplot(fig)

# Gunakan K=3 Berdasarkan Grafik Elbow
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
filtered_day_df['Cluster'] = kmeans.fit_predict(features_scaled)

# Visualisasi Hasil Clustering
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=filtered_day_df, x='cnt', y='registered', hue='Cluster', palette='viridis', ax=ax)
ax.set_xlabel('Total Rentals')
ax.set_ylabel('Registered Users')
ax.set_title('Customer Segmentation dengan K-Means')
st.pyplot(fig)

# Menampilkan Statistik dari Setiap Cluster
st.subheader("Statistik dari Setiap Cluster")
st.write(filtered_day_df.groupby('Cluster')[['cnt', 'casual', 'registered']].mean())

st.caption("Dashboard by Streamlit")
