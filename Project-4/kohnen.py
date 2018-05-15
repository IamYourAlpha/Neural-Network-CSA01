from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

data = pd.read_csv("./iris.csv", names=["f1", "f2", "f3", "f4", "label"],
                   index_col=None, usecols=None)

data = np.array( data.values)
features = data[:, 0:3]
# Beginning of The k-Means algorithm 

km = KMeans(n_clusters=3, init='k-means++', random_state=0).fit(features)
print km.labels_

