import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import seaborn as sns  

# Paso 1: Cargar los datos
ecommerce_data = pd.read_csv('Archivos\ecommerce_data.csv')

# Paso 2: Preprocesamiento de datos
X = ecommerce_data[['Total de compras', 'Frecuencia de compras', 'Categoría favorita', 'Valor promedio de compra', 'Tiempo en el sitio']]

# Normalización de datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Paso 3: Aplicar K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
y=kmeans.fit(X_scaled)


# Añadir los resultados al dataframe
ecommerce_data['Cluster'] = kmeans.labels_

h=ecommerce_data[['Total de compras','Frecuencia de compras','Valor promedio de compra','Tiempo en el sitio','Cluster','Categoría favorita']]
sns.pairplot(data=h,hue='Cluster',palette='viridis')
plt.show()

# Paso 4: Descripción de resultados
resultados = ecommerce_data.groupby('Cluster').agg({
    'Total de compras': 'mean',
    'Frecuencia de compras': 'mean',
    'Categoría favorita': lambda x: x.mode()[0],
    'Valor promedio de compra': 'mean',
    'Tiempo en el sitio': 'mean',
    'Cluster': 'count'
}).rename(columns={'Cluster': 'Count'})

print("Descripción de Resultados:")
print(resultados)

# Paso 5: Visualización de los centroides
centroids = kmeans.cluster_centers_
centroid_df = pd.DataFrame(centroids, columns=X.columns)
print("\nCentroides de los clusters:")
print(centroid_df)

# Paso 6: Visualización de los clusters
plt.figure(figsize=(10, 6))
for cluster in range(len(centroids)):
    plt.scatter(X_scaled[kmeans.labels_ == cluster][:, 0], X_scaled[kmeans.labels_ == cluster][:, 1], label=f'Cluster {cluster}')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='black', label='Centroides')
plt.xlabel('Total de compras (normalizado)')
plt.ylabel('Frecuencia de compras (normalizado)')
plt.title('Clusters de clientes')
plt.legend()
plt.show()

# Paso 7: Análisis adicional de los clusters y métricas de eficiencia
# Aquí se pueden realizar análisis adicionales según las necesidades específicas
# Métricas como la suma de los cuadrados de las distancias a los centroides (inertia_) pueden indicar la eficiencia del modelo
print("\nInertia (suma de los cuadrados de las distancias a los centroides):", kmeans.inertia_)

inertia_values = []
k_values = range(1, 11)  # Probaremos con k desde 1 hasta 10 clusters

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

# Graficar la inertia en función del número de clusters
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inertia')
plt.title('Inertia vs Número de clusters')
plt.xticks(k_values)
plt.grid(True)
plt.show()

silhouette_scores = []
k_values = range(2, 11)  # Probaremos con k desde 2 hasta 10 clusters

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Graficar el silhouette_score en función del número de clusters
plt.figure(figsize=(10, 6))
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Número de clusters')
plt.xticks(k_values)
plt.grid(True)
plt.show()







