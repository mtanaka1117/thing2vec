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
    # colormap = plt.cm.get_cmap('tab20', len(unique_clusters))
    markers = ['o', 's', 'D', '^', '*', '+', 'x']
    colormap = ['palevioletred', 'darkmagenta', 'darkslateblue', 'blue', 'steelblue', 'darkturquoise', 'mediumseagreen', 'green', 'limegreen', 'yellow',
            'goldenrod', 'orange', 'red', 'brown', 'salmon', 'darkred', 'rosybrown']

    unique_clusters = set(clusters)
    for i, cls in enumerate(unique_clusters):
        if cls == -1: color = 'black' # ノイズ
        else: color = colormap[i]
        marker = markers[i % len(markers)]
        plt.scatter(reduced_data[clusters == cls, 0], reduced_data[clusters == cls, 1], label=f"Cluster {cls}", color=color, marker=marker)
    
    texts = []
    for i in range(23):
        x, y = reduced_data[i, 0], reduced_data[i, 1]
        texts.append(plt.text(x, y, i, size=10, alpha=0.9))

    adjust_text(
        texts,
        expand_text=(15, 15),
        force_text=(0.5, 0.5),
        force_points=(0.3, 0.3),
        arrowprops=dict(arrowstyle='->', color='red', lw=0.5)
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="upper left", bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig('./img/dbscan_2d_dev.jpg')
    plt.close()


def dbscan_3d(normalized_embeddings, clusters):
    fig = plt.figure(figsize=(10, 5))

    unique_clusters = set(clusters)
    markers = ['o', 's', 'D', '^', '*', '+', 'x']
    colormap = ['palevioletred', 'darkmagenta', 'darkslateblue', 'blue', 'steelblue', 'darkturquoise', 'mediumseagreen', 'green', 'limegreen', 'yellow',
            'goldenrod', 'orange', 'red', 'brown', 'salmon', 'darkred', 'rosybrown']

    elevations = [20, 80]
    azimuths = [30, 210]

    for i, (elev, azim) in enumerate(zip(elevations, azimuths)):
        ax = fig.add_subplot(1, len(elevations), i + 1, projection='3d')
        for i, label in enumerate(unique_clusters):
            if label == -1: color = 'black' # ノイズ
            else: color = colormap[i]
            marker = markers[i % len(markers)]
            mask = clusters == label
            ax.scatter(normalized_embeddings[mask, 0], normalized_embeddings[mask, 1], normalized_embeddings[mask, 2], 
                        color=color, label=f'Cluster {label}' if label != -1 else "Noise", marker=marker)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.view_init(elev=elev, azim=azim)

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.zaxis.set_major_locator(MaxNLocator(5))

    plt.tight_layout()
    plt.savefig('./img/dbscan_3d_dev.jpg')
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
    normalized_embeddings = embeddings.numpy()

    dbscan = DBSCAN(eps, min_samples=4)
    clusters = dbscan.fit_predict(normalized_embeddings)
    
    # df = pd.DataFrame({
    #     "X": normalized_embeddings[:, 0],
    #     "Y": normalized_embeddings[:, 1], 
    #     "Cluster": clusters,  # クラスタ番号
    #     "Label": labels  # 各データポイントのラベル（オプション）
    # })
    
    dbscan_2d(normalized_embeddings, clusters)
    dbscan_3d(normalized_embeddings, clusters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--emb_dim', type=int, help='Cuda number to use', default=3)
    parser.add_argument('-i', '--num_items', type=int, default=23)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--model', default='./output/model_dev/models/model200.pth')
    args = parser.parse_args()
    
    num_tokens = 2*6*2

    dbscan_plot(args.num_items, args.emb_dim, num_tokens, args.eps, args.model)

