import pandas as pd
import matplotlib.pyplot as plt

d= pd.read_csv("Mall_Customers.csv")
d.head()

d.info()
d.isnull().sum()

from sklearn.cluster import KMeans
wcss=[]
##wcss- within cluster sum of square

for i in range(1,11):
  kmeans=KMeans(n_clusters=i,init="k-means++")
  kmeans.fit(d.iloc[:,3:])
  wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")

km = KMeans(n_clusters = 5)
km.fit(d.iloc[:,3:])

y_pred = km.predict(d.iloc[:,3:])
d["cluster"] = y_pred
df0 = d[d["cluster"]==0]
df1 = d[d["cluster"]==1]
df2 = d[d["cluster"]==2]
df3 = d[d["cluster"]==3]
df4 = d[d["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="yellow",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="pink",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="purple",label="cluster4")
plt.legend()
plt.title("Customer Segments")