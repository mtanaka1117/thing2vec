import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn
from libs.model import Thing2Vec
import pandas as pd

num_items = 251
embed_size = 8
num_output_tokens=2688

# load model
model = Thing2Vec(num_items, embed_size, num_output_tokens)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.load_model('./output/model/models/model200.pth')
model.eval()


item_indices = torch.arange(num_items).to(device) 
with torch.no_grad():
    embeddings = model.embedding(item_indices).cpu()

normalized_embeddings = F.normalize(embeddings, p=2, dim=0).numpy()

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(normalized_embeddings)

# csvにクラスタを出力
csv_file = './data/thing_train_data/sorted_kishino.csv'
df = pd.read_csv(csv_file)
df['Cluster'] = labels
df.to_csv('cluster.csv', index=False)


pca = PCA(n_components=3)
data_3d = pca.fit_transform(normalized_embeddings)


fig = plt.figure(figsize=(12, 8))
elevations = [20, 50, 80]
azimuths = [30, 120, 210]

for i, (elev, azim) in enumerate(zip(elevations, azimuths)):
    ax = fig.add_subplot(1, len(elevations), i + 1, projection='3d')
    ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=labels, cmap='jet', marker='o')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"View: elev={elev}, azim={azim}")
    

plt.tight_layout()
plt.savefig('pca3d.png')

