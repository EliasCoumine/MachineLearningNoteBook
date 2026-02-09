from datasets import load_dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

dataset = load_dataset("codesignal/wine-quality")
red_df = pd.DataFrame(dataset["red"])
white_df = pd.DataFrame(dataset["white"])

red_df["color"] = "red"
white_df["color"] = "white"

df = pd.concat([red_df, white_df], ignore_index=True)

X = df.drop(columns=["color"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_values = list(range(1, 11))
inertia = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()
plt.close()

# FIXED: n_clusters (plural) and removed marker='o'
final_kmeans = KMeans(n_clusters=3, random_state=42)
final_kmeans.fit(X_scaled)
cluster = final_kmeans.labels_
df['cluster'] = cluster

print(df.head())

from sklearn.decomposition import PCA

# PCA to reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot clusters in 2D
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster, cmap='viridis', alpha=0.6, s=30)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Wine Quality Clusters (K-Means, k=3)')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, cluster, test_size=0.2, random_state=42)

# Train Random Forest to predict clusters
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
predictions = rf_model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Cluster')
plt.ylabel('Actual Cluster')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()