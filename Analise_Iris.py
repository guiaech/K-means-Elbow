#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
iris = pd.read_csv(r"C:\Users\guilh\OneDrive\Desktop\projetos\dataSetIris\iris.csv")
display(iris)


# In[11]:


X = iris.iloc[:, 1:5].values
X


# In[27]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, init = 'random')
kmeans.fit(X)
kmeans.cluster_centers_


# In[28]:


distance = kmeans.fit_transform(X)
distance


# In[29]:


labels = kmeans.labels_
labels


# In[37]:


from sklearn.cluster import KMeans
wcss = []
 
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'random')
    kmeans.fit(X)
    print (i,kmeans.inertia_)
    wcss.append(kmeans.inertia_)  
plt.plot(range(1, 11), wcss)
plt.title('O Metodo Elbow')
plt.xlabel('Numero de Clusters')
plt.ylabel('WSS') #within cluster sum of squares
plt.show()


# In[33]:


data = [
        [ 4.12, 3.4, 1.6, 0.7],
        [ 5.2, 5.8, 5.2, 6.7],
        [ 3.1, 3.5, 3.3, 3.0]
    ]
kmeans.predict(data)


# In[34]:


import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:,1], s = 100, c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'red',label = 'Centroids')
plt.title('Iris Clusters and Centroids')
plt.xlabel('SepalLength')
plt.ylabel('SepalWidth')
plt.legend()

plt.show()


# In[ ]:





# In[ ]:




