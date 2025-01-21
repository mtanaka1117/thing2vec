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


pca = PCA(n_components=2)
reduced_data = pca.fit_transform(normalized_embeddings)

plt.figure(figsize=(8, 6))

# クラスタごとに色を分けてプロット
unique_labels = set(labels)
for label in unique_labels:
    # ノイズのラベルは -1 なので除外
    if label == -1:
        color = 'gray'  # ノイズは灰色
    else:
        color = plt.cm.jet(label / len(unique_labels))  # 他のクラスタには異なる色
    plt.scatter(reduced_data[labels == label, 0], reduced_data[labels == label, 1], label=f"Cluster {label}", color=color)

# 凡例
plt.legend()

# グラフの表示
plt.title("DBSCAN Clustering")
plt.xlabel("X1")
plt.ylabel("X2")
plt.savefig('dbscan_2d.png')