from datasets import load_dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


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

final_kmeans = KMeans(n_cluster = 3, marker='o' )
final_kmeans.fit(X_scaled)


cluster = final_kmeans.labels_
df['cluster'] = cluster



print(df.head)
 







 

