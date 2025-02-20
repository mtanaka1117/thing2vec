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
from adjustText import adjust_text
import argparse

from matplotlib.ticker import MaxNLocator


def dbscan_2d(normalized_embeddings, clusters):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(normalized_embeddings)

    plt.figure(figsize=(8, 5))
    unique_clusters = set(clusters)
    # colors = plt.cm.jet(np.linspace(0, 1, len(unique_clusters)))
    colors = ['palevioletred', 'darkmagenta', 'darkslateblue', 'blue', 'steelblue', 'darkturquoise', 'mediumseagreen', 'green', 'limegreen', 'yellow',
              'goldenrod', 'orange', 'red', 'brown', 'salmon', 'darkred', 'rosybrown']

    unique_clusters = set(clusters)
    for cls, color in zip(unique_clusters, colors):
        if cls == -1: # ノイズ
            color = 'black'
        
        plt.scatter(reduced_data[clusters == cls, 0], reduced_data[clusters == cls, 1], label=f"Cluster {cls}", color=color)

    csv_file = './data/thing_train_data/sorted_kishino.csv'
    df = pd.read_csv(csv_file)
    df["cluster"] = clusters
    df.to_csv('dbscan_cluster.csv', index=False)
    
    labels = df['label']

    texts = []
    for i, label in enumerate(labels):
        x, y = reduced_data[i, 0], reduced_data[i, 1]
        texts.append(plt.text(x, y, label, size=10))

    adjust_text(
        texts,
        expand_text=(5, 5),
        arrowprops=dict(arrowstyle='->', color='red')
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="upper left", bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig('./img/dbscan_2d.jpg')
    plt.close()


def dbscan_3d(normalized_embeddings, clusters, dbscan_centers):
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(normalized_embeddings)

    fig = plt.figure(figsize=(15, 5))

    unique_clusters = set(clusters)
    # colors = plt.cm.jet(np.linspace(0, 1, len(unique_clusters)))
    colors = ['palevioletred', 'darkmagenta', 'darkslateblue', 'blue', 'darkturquoise', 'mediumseagreen', 'green', 'limegreen', 'yellow', 
              'goldenrod', 'orange', 'red', 'brown', 'salmon', 'darkred', 'rosybrown']

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
        
        if len(dbscan_centers) > 0:
            reduced_centers = pca.transform(dbscan_centers)
            ax.scatter(reduced_centers[:,0], reduced_centers[:,1], reduced_centers[:,2],
                       c='red', marker='x', s=20, label='Centroids')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.view_init(elev=elev, azim=azim)

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.zaxis.set_major_locator(MaxNLocator(5))

    plt.tight_layout()
    plt.savefig('./img/dbscan_3d.jpg')
    plt.close()


def dbscan_plot(num_items, embed_size, num_output_tokens, eps, model_path):

    model = Thing2Vec(num_items, embed_size, num_output_tokens)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_model(model_path)
    model.eval()

    item_indices = torch.arange(num_items).to(device)
    with torch.no_grad():
        embeddings = model.embedding(item_indices).cpu()
    normalized_embeddings = F.normalize(embeddings, p=2, dim=0).numpy()

    dbscan = DBSCAN(eps, min_samples=4)
    clusters = dbscan.fit_predict(normalized_embeddings)
    unique_clusters = set(clusters)
    centroids = []

    for label in unique_clusters:
        if label == -1:
            continue
        mask = clusters == label
        cluster_center = np.mean(normalized_embeddings[mask], axis=0)
        centroids.append([label, *cluster_center])

    dbscan_centers = np.array([normalized_embeddings[clusters == label].mean(axis=0) for label in unique_clusters if label != -1])

    df_center = pd.DataFrame(centroids)
    df_center.to_csv('dbscan_center.csv', index=False)
    
    dbscan_2d(normalized_embeddings, clusters)
    dbscan_3d(normalized_embeddings, clusters, dbscan_centers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--num_items', type=int)
    parser.add_argument('--embed_size', type=int, default=10)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--model', type=str, default='./output/model/models/model200.pth')
    args = parser.parse_args()

    # num_tokens = 24*2*6*5*2*5*5
    # num_tokens = 24*2*6*5*2
    num_tokens = 24*2*6*2

    dbscan_plot(args.num_items, args.embed_size, num_tokens, args.eps, args.model)