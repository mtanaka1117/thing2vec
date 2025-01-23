# 必要なライブラリのインポート
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
import torch.nn.functional as F
import torch
from libs.model import Thing2Vec
import pandas as pd

import argparse


def dbscan_3d(normalized_embeddings, clusters):
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(normalized_embeddings)

    fig = plt.figure(figsize=(12, 8))

    unique_clusters = set(clusters)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_clusters)))

    elevations = [20, 50, 80]
    azimuths = [30, 120, 210]

    for i, (elev, azim) in enumerate(zip(elevations, azimuths)):
        ax = fig.add_subplot(1, len(elevations), i + 1, projection='3d')
        for label, color in zip(unique_clusters, colors):
            if label == -1:  # ノイズの場合
                color = 'black'
            mask = clusters == label
            ax.scatter(reduced_data[mask, 0], reduced_data[mask, 1], reduced_data[mask, 2], 
                        color=color, label=f'Cluster {label}' if label != -1 else "Noise")

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"View: elev={elev}, azim={azim}")

    plt.tight_layout()
    plt.savefig('dbscan_3d.jpg')
    plt.close()


def dbscan_2d(normalized_embeddings, clusters):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(normalized_embeddings)

    plt.figure(figsize=(8, 6))

    unique_clusters = set(clusters)
    for cls in unique_clusters:
        # ノイズのラベル
        if cls == -1:
            color = 'black'
        else:
            color = plt.cm.jet(cls / len(unique_clusters))
        plt.scatter(reduced_data[clusters == cls, 0], reduced_data[clusters == cls, 1], label=f"Cluster {cls}", color=color)
        
        # if cls == 1:
        #     cluster_indices = np.where(clusters == cls)[0]
        #     for idx in cluster_indices:
        #         item_id = df.iloc[idx]['id']
        #         item_label = df.iloc[idx]['label']
        #         plt.text(reduced_data[idx, 0], reduced_data[idx, 1], f'({item_id}-{item_label})', fontsize=8)

    plt.title("DBSCAN Clustering")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.savefig('dbscan_2d.jpg')
    plt.close()


def dbscan_plot(num_items, embed_size, num_output_tokens, eps):

    model = Thing2Vec(num_items, embed_size, num_output_tokens)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_model('./output/model/models/model200.pth')
    model.eval()

    item_indices = torch.arange(num_items).to(device) 
    with torch.no_grad():
        embeddings = model.embedding(item_indices).cpu()
    normalized_embeddings = F.normalize(embeddings, p=2, dim=0).numpy()

    dbscan = DBSCAN(eps, min_samples=5)
    clusters = dbscan.fit_predict(normalized_embeddings)
    csv_file = './data/thing_train_data/sorted_kishino.csv'
    df = pd.read_csv(csv_file)
    df["cluster"] = clusters
    df.to_csv('dbscan_cluster.csv', index=False)


    dbscan_2d(normalized_embeddings, clusters)
    dbscan_3d(normalized_embeddings, clusters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--num_items', type=int)
    parser.add_argument('--embed_size', type=int, default=10)
    parser.add_argument('--num_tokens', type=int, default=67200)
    parser.add_argument('--eps', type=int, default=0.1)
    args = parser.parse_args()

    num_tokens = 24*2*6*4*2*5*5

    dbscan_plot(args.num_items, args.embed_size, num_tokens)