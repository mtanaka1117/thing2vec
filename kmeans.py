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


def kmeans_2d(normalized_embeddings, clusters):
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(normalized_embeddings)

    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=clusters, cmap='jet', marker='o')

    # texts = []
    # for i, label in enumerate(clusters):
    #     x, y = reduced_vectors[i, 0], reduced_vectors[i, 1]
    #     texts.append(plt.text(x, y, (i, label), size=10))

    # adjust_text(
    #     texts,
    #     expand_text=(1.2, 1.2),
    #     arrowprops=dict(arrowstyle='->', color='red')
    # )

    plt.savefig('kmeans_2d.jpg')
    plt.close()


def kmeans_3d(normalized_embeddings, clusters):
    pca = PCA(n_components=3)
    data_3d = pca.fit_transform(normalized_embeddings)

    fig = plt.figure(figsize=(12, 8))
    elevations = [20, 50, 80]
    azimuths = [30, 120, 210]

    for i, (elev, azim) in enumerate(zip(elevations, azimuths)):
        ax = fig.add_subplot(1, len(elevations), i + 1, projection='3d')
        ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=clusters, cmap='jet', marker='o')

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"View: elev={elev}, azim={azim}")
        
    plt.tight_layout()
    plt.savefig('kmeans_3d.jpg')
    plt.close()


def kmeans_plot(num_items, embed_size, num_output_tokens, n_clusters):

    model = Thing2Vec(num_items, embed_size, num_output_tokens)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_model('./output/model/models/model200.pth')
    model.eval()

    item_indices = torch.arange(num_items).to(device) 
    with torch.no_grad():
        embeddings = model.embedding(item_indices).cpu()

    normalized_embeddings = F.normalize(embeddings, p=2, dim=0).numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(normalized_embeddings)

    csv_file = './data/thing_train_data/sorted_kishino.csv'
    df = pd.read_csv(csv_file)
    df['Cluster'] = clusters
    df.to_csv('kmeans_cluster.csv', index=False)

    kmeans_2d(normalized_embeddings, clusters)
    kmeans_3d(normalized_embeddings, clusters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--num_items', type=int)
    parser.add_argument('--embed_size', type=int, default=10)
    parser.add_argument('--n_clusters', type=int, default=5)
    args = parser.parse_args()

    num_tokens = 24*2*6*4*2*5*5

    kmeans_plot(args.num_items, args.embed_size, num_tokens, args.n_clusters)
    