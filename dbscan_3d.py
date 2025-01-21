# 必要なライブラリのインポート
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
import torch.nn.functional as F
import torch
import torch.nn as nn
from libs.train_utils import FeatureQuantization
from libs.model import Thing2Vec
import pandas as pd
from adjustText import adjust_text


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


# DBSCANの設定
dbscan = DBSCAN(eps=0.11, min_samples=5)
labels = dbscan.fit_predict(normalized_embeddings)

# PCAで次元削減（8次元 → 3次元）
pca = PCA(n_components=3)
reduced_data = pca.fit_transform(normalized_embeddings)


# 3Dプロット
fig = plt.figure(figsize=(12, 8))

# クラスタごとに色分けしてプロット
unique_labels = set(labels)
colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

elevations = [20, 50, 80]
azimuths = [30, 120, 210]

for i, (elev, azim) in enumerate(zip(elevations, azimuths)):
    ax = fig.add_subplot(1, len(elevations), i + 1, projection='3d')
    for label, color in zip(unique_labels, colors):
        if label == -1:  # ノイズの場合
            color = 'black'
        mask = labels == label
        ax.scatter(reduced_data[mask, 0], reduced_data[mask, 1], reduced_data[mask, 2], 
                    color=color, label=f'Cluster {label}' if label != -1 else "Noise")
    # ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=labels, cmap='jet', marker='o')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"View: elev={elev}, azim={azim}")

# plt.legend(loc='upper center', bbox_to_anchor=(0, 0), ncol=3)
plt.tight_layout()
plt.savefig('dbscan_3d.png')