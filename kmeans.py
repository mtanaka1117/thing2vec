import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn
from libs.model import Thing2Vec
import pandas as pd
from adjustText import adjust_text
import argparse
from matplotlib.ticker import MaxNLocator

def kmeans_2d(normalized_embeddings, clusters):
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(normalized_embeddings)

    plt.figure(figsize=(8, 5))
    # plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=clusters, cmap='tab20', marker='o')
    
    unique_clusters = set(clusters)
    markers = ['o', 's', 'D', '^', '*', '+', 'x']
    colormap = ['palevioletred', 'darkmagenta', 'darkslateblue', 'blue', 'steelblue', 'darkturquoise', 'mediumseagreen', 'green', 'limegreen', 'yellow',
            'goldenrod', 'orange', 'red', 'brown', 'salmon', 'darkred', 'rosybrown']
    
    for i, cls in enumerate(unique_clusters):
        color = colormap[i]
        marker = markers[i % len(markers)]
        plt.scatter(reduced_vectors[clusters == cls, 0], reduced_vectors[clusters == cls, 1], label=f"Cluster {cls}", color=color, marker=marker)

    csv_file = './data/thing_train_data/sorted_kishino.csv'
    df = pd.read_csv(csv_file)
    df['Cluster'] = clusters
    df.to_csv('kmeans_cluster.csv', index=False)

    labels = df['label']

    texts = []
    for i, label in enumerate(labels):
        x, y = reduced_vectors[i, 0], reduced_vectors[i, 1]
        texts.append(plt.text(x, y, label, size=10))

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
    plt.savefig('./img/kmeans_2d.jpg')
    plt.close()


def kmeans_3d(normalized_embeddings, clusters, kmeans_centers):
    pca = PCA(n_components=3)
    data_3d = pca.fit_transform(normalized_embeddings)
    centers_3d = pca.transform(kmeans_centers)

    fig = plt.figure(figsize=(10, 5))
    elevations = [20, 80]
    azimuths = [30, 210]

    unique_clusters = set(clusters)
    markers = ['o', 's', 'D', '^', '*', '+', 'x']
    colormap = ['palevioletred', 'darkmagenta', 'darkslateblue', 'blue', 'steelblue', 'darkturquoise', 'mediumseagreen', 'green', 'limegreen', 'yellow',
            'goldenrod', 'orange', 'red', 'brown', 'salmon', 'darkred', 'rosybrown']

    for i, (elev, azim) in enumerate(zip(elevations, azimuths)):
        ax = fig.add_subplot(1, len(elevations), i + 1, projection='3d')
        
        for i, label in enumerate(unique_clusters):
            color = colormap[i]
            marker = markers[i % len(markers)]
            mask = clusters == label
            ax.scatter(data_3d[mask, 0], data_3d[mask, 1], data_3d[mask, 2], color=color, label=f'Cluster {label}', marker=marker)
        
        # ax.scatter(centers_3d[:,0], centers_3d[:,1], centers_3d[:,2], color='red', marker='x', s=20, label='Centroids')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.view_init(elev=elev, azim=azim)

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.zaxis.set_major_locator(MaxNLocator(5))
        
    plt.tight_layout()
    plt.savefig('./img/kmeans_3d.jpg')
    plt.close()


def kmeans_plot(num_items, embed_size, num_output_tokens, n_clusters, model_path):

    model = Thing2Vec(num_items, embed_size, num_output_tokens)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_model(model_path)
    model.eval()

    item_indices = torch.arange(num_items).to(device) 
    with torch.no_grad():
        embeddings = model.embedding(item_indices).cpu()

    # normalized_embeddings = F.normalize(embeddings, p=2, dim=0).numpy()
    normalized_embeddings = embeddings.numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(normalized_embeddings)
    kmeans_centers = kmeans.cluster_centers_

    df_center = pd.DataFrame(kmeans_centers)
    df_center.to_csv('kmeans_center.csv', index=False)

    kmeans_2d(normalized_embeddings, clusters)
    kmeans_3d(normalized_embeddings, clusters, kmeans_centers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--num_items', type=int)
    parser.add_argument('--embed_size', type=int, default=10)
    parser.add_argument('--n_clusters', type=int, default=5)
    parser.add_argument('--model', type=str, default='./output/model/models/model200.pth')
    args = parser.parse_args()

    # num_tokens = 24*2*6*5*2*5*5
    num_tokens = 24*2*6*2
    epoch = 200

    kmeans_plot(args.num_items, args.embed_size, num_tokens, args.n_clusters, args.model)
    