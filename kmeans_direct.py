import datetime
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text
import argparse


def label_quantization(label):
    return label

def dt_quantization(dt):
    '''
    dt: hhmm
    時間帯：[
        6~9時: 0, 朝
        9~12時: 1, 昼前
        12~15時: 2, 昼過ぎ
        15~18時: 3, 夕方
        18時~21時: 4, 夜
        21~6時: 5, 深夜
    ]
    '''
    dt = datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
    hour = dt.hour
    if hour>=6 and hour<9:
        return 0
    elif hour>=9 and hour<12:
        return 1
    elif hour>=12 and hour<15:
        return 2
    elif hour>=15 and hour<18:
        return 3
    elif hour>=18 and hour<21:
        return 4
    else:
        return 5

# 曜日
def dow_quantization(dow):
    if dow in ["Saturday", "Sunday"]:
        return 1
    else:
        return 0

# 触れたかどうか
def touch_quantization(is_touch):
    if is_touch:
        return 1
    else:
        return 0
    

def kmeans_2d(normalized_embeddings, clusters):
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(normalized_embeddings)

    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=clusters, cmap='jet', marker='o')

    csv_file = './data/thing_train_data/sorted_kishino.csv'
    df = pd.read_csv(csv_file)
    df['Cluster'] = clusters
    df.to_csv('kmeans_cluster.csv', index=False)

    labels = df['label']

    texts = []
    for i, label in enumerate(labels):
        x, y = reduced_vectors[i, 0], reduced_vectors[i, 1]
        texts.append(plt.text(x, y, (i, label), size=10))

    adjust_text(
        texts,
        expand_text=(1.2, 1.2),
        arrowprops=dict(arrowstyle='->', color='red')
    )

    plt.savefig('kmeans_2d.jpg')
    plt.close()


def kmeans_3d(normalized_embeddings, clusters):
    pca = PCA(n_components=3)
    data_3d = pca.fit_transform(normalized_embeddings)

    fig = plt.figure(figsize=(12, 5))
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




kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(normalized_embeddings)


