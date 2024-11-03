import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('countries of the world.csv')

# Remove commas from numeric columns and convert them to float
for column in ['Population', 'Area (sq. mi.)', 'GDP ($ per capita)', 'Literacy (%)', 'Phones (per 1000)', 'Arable (%)', 'Birthrate', 'Deathrate']:
    df[column] = df[column].astype(str).str.replace(',', '').astype(float)

# Data Cleaning and Selection of Relevant Features
# Remove any rows with missing values
df = df.dropna()

# Select relevant features
features = ['Population', 'Area (sq. mi.)', 'GDP ($ per capita)', 'Literacy (%)', 'Phones (per 1000)', 'Arable (%)', 'Birthrate', 'Deathrate']

# Filter dataset with selected features
X = df[features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose the optimal k using the Elbow Method
sse = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

# Plotting the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, sse, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Distances (SSE)')
plt.title('Elbow Method to Determine Optimal k')
plt.show()

# Choose k based on the elbow curve (for example, k=4)
k_optimal = 4
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='GDP ($ per capita)', y='Literacy (%)', hue='Cluster', palette='viridis', s=100)
plt.title('Clusters of Countries by GDP per Capita and Literacy Rate')
plt.xlabel('GDP per Capita')
plt.ylabel('Literacy Rate')
plt.legend(title='Cluster')
plt.show()

# Display cluster characteristics
cluster_summary = df.groupby('Cluster')[features].mean()
print(cluster_summary)
