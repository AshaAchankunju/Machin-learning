import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import scipy
import pylab
import scipy.cluster.hierarchy
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt 
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from scipy.spatial.distance import pdist, squareform

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/cars_clus.csv"
response = requests.get(url)

with open("cars_clus.csv", "wb") as f:
    f.write(response.content)

filename = 'cars_clus.csv'

#Read csv
pdf = pd.read_csv(filename)
print ("Shape of dataset: ", pdf.shape)

pdf.head(5)
print ("Shape of dataset before cleaning: ", pdf.size)
pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)
print ("Shape of dataset after cleaning: ", pdf.size)
pdf.head(5)
featureset = pdf[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]

x = featureset.values #returns a numpy array
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)
feature_mtx [0:5]


leng = feature_mtx.shape[0]
D = pdist(feature_mtx, metric='euclidean')   # condensed matrix
Z = hierarchy.linkage(D, method='complete')


max_d = 3
clusters = fcluster(Z, max_d, criterion='distance')
clusters
k = 5
clusters = fcluster(Z, k, criterion='maxclust')
clusters




# # redo it again with scikit-learn package
# fig = pylab.figure(figsize=(18,50))
# def llf(id):
#     return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
    
# dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')
# dist_matrix = euclidean_distances(feature_mtx,feature_mtx) 
# print(dist_matrix)
# Z_using_dist_matrix = hierarchy.linkage(dist_matrix, 'complete')
# fig = pylab.figure(figsize=(18,50))
# def llf(id):
#     return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
    
# dendro = hierarchy.dendrogram(Z_using_dist_matrix,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')
# agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'complete')
# agglom.fit(dist_matrix)

# agglom.labels_
# pdf['cluster_'] = agglom.labels_
# pdf.head()
# import matplotlib.cm as cm
# n_clusters = max(agglom.labels_)+1
# colors = cm.rainbow(np.linspace(0, 1, n_clusters))
# cluster_labels = list(range(0, n_clusters))

# # Create a figure of size 6 inches by 4 inches.
# plt.figure(figsize=(16,14))

# for color, label in zip(colors, cluster_labels):
#     subset = pdf[pdf.cluster_ == label]
#     for i in subset.index:
#             plt.text(subset.horsepow[i], subset.mpg[i],str(subset['model'][i]), rotation=25) 
#     plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10, c=color, label='cluster'+str(label),alpha=0.5)
# #    plt.scatter(subset.horsepow, subset.mpg)
# plt.legend()
# plt.title('Clusters')
# plt.xlabel('horsepow')
# plt.ylabel('mpg')
# pdf.groupby(['cluster_','type'])['cluster_'].count()
# agg_cars = pdf.groupby(['cluster_', 'type'])[['horsepow', 'engine_s', 'mpg', 'price']].mean()

# agg_cars
# plt.figure(figsize=(16,10))
# for color, label in zip(colors, cluster_labels):
#     subset = agg_cars.loc[(label,),]
#     for i in subset.index:
#         plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k')
#     plt.scatter(subset.horsepow, subset.mpg, s=subset.price*10, color=color, label='cluster'+str(label), alpha=0.5)

# plt.legend()
# plt.title('Clusters')
# plt.xlabel('horsepow')
# plt.ylabel('mpg')
