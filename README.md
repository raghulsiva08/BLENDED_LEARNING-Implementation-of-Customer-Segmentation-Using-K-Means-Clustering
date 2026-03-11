# BLENDED LEARNING
# Implementation of Customer Segmentation Using K-Means Clustering

## AIM:
To implement customer segmentation using K-Means clustering on the Mall Customers dataset to group customers based on purchasing habits.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries such as pandas, seaborn, matplotlib, and sklearn.
2.Load the dataset CustomerData.csv and Select important features (Age, Annual Income, Spending Score).
3.Apply StandardScaler to normalize the feature values.
4.Use the Elbow Method to determine the optimal number of clusters.
5.Train the K-Means clustering model with the optimal number of clusters.
6.Assign cluster labels to each data point.
7.Calculate the Silhouette Score to evaluate clustering performance.
8.Visualize the clusters using a scatter plot.
## Program:
~~~
/*
Program to implement customer segmentation using K-Means clustering on the Mall Customers dataset.
Developed by: RAGHUL.S
RegisterNumber: 212225040325
*/
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Load the dataset
data = pd.read_csv('CustomerData.csv')

# Step 2: Explore the data
print(data.head())
print(data.columns)

# Step 3: Select relevant features
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]

# Step 4: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Elbow method to find optimal clusters
inertia_values = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)  # Explicit n_init to suppress warning
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia_values, marker='o', linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Step 6: Train KMeans with chosen clusters
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)  # Explicit n_init
kmeans.fit(X_scaled)

# Step 7: Add cluster labels to data
data['Cluster'] = kmeans.labels_

# Silhouette score
sil_score = silhouette_score(X_scaled, kmeans.labels_)
print(f'Silhouette Score: {sil_score}')

# Step 8: Visualize clusters
print("\nName:RAGHUL.S")
print("Reg No.: 212225040325\n")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data,x='Annual Income (k$)',y='Spending Score (1-100)',hue='Cluster', palette='viridis',s=100,alpha=0.7)
plt.title('Customer Segmentation based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
~~~

## Output:
<img width="820" height="205" alt="Screenshot 2026-03-11 195906" src="https://github.com/user-attachments/assets/98c922c6-b695-4709-ae01-fce6c9f5d58e" />

<img width="1005" height="475" alt="Screenshot 2026-03-11 195919" src="https://github.com/user-attachments/assets/eb6307eb-523c-4e4e-a870-876133e2ff7f" />

<img width="1385" height="750" alt="Screenshot 2026-03-11 195933" src="https://github.com/user-attachments/assets/685b1fa5-8a18-40c8-a1a4-c12c80040e19" />


## Result:
Thus, customer segmentation was successfully implemented using K-Means clustering, grouping customers into distinct segments based on their annual income and spending score. 
