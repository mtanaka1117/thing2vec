import datetime
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
import pandas as pd
from adjustText import adjust_text
import argparse

from matplotlib.ticker import MaxNLocator


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


def dbscan_3d(data, clusters):
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(data)

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
            if label == -1:  # ノイズの場合
                color = 'black'
            else: color = colormap[i]
            marker = markers[i % len(markers)]
            mask = clusters == label
            ax.scatter(reduced_data[mask, 0], reduced_data[mask, 1], reduced_data[mask, 2], 
                        color=color, label=f'Cluster {label}' if label != -1 else "Noise", marker=marker)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.view_init(elev=elev, azim=azim)

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.zaxis.set_major_locator(MaxNLocator(5))

    plt.tight_layout()
    plt.savefig('./img/dbscan_direct_3d.jpg')
    plt.close()


def dbscan_2d(data, clusters):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    plt.figure(figsize=(8, 5))
    unique_clusters = set(clusters)
    markers = ['o', 's', 'D', '^', '*', '+', 'x']
    colormap = ['palevioletred', 'darkmagenta', 'darkslateblue', 'blue', 'steelblue', 'darkturquoise', 'mediumseagreen', 'green', 'limegreen', 'yellow',
            'goldenrod', 'orange', 'red', 'brown', 'salmon', 'darkred', 'rosybrown']
    
    for i, cls in enumerate(unique_clusters):
        if cls == -1: # ノイズ
            color = 'black'
        else: color = colormap[i]
        marker = markers[i % len(markers)]
        plt.scatter(reduced_data[clusters == cls, 0], reduced_data[clusters == cls, 1], label=f"Cluster {cls}", color=color, marker=marker)

    csv_file = './data/thing_train_data/sorted_kishino.csv'
    df = pd.read_csv(csv_file)
    df["cluster"] = clusters
    df.to_csv('dbscan_direct_cluster.csv', index=False)
    
    labels = df['label']

    texts = []
    for i, label in enumerate(labels):
        if label != 5 and label != 8 and label != 9:
            x, y = reduced_data[i, 0], reduced_data[i, 1]
            texts.append(plt.text(x, y, label, size=10))
        elif i % 2 == 0 and (label == 5 or label == 8 or label == 9):
            x, y = reduced_data[i, 0], reduced_data[i, 1]
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
    plt.savefig('./img/dbscan_direct_2d.jpg')
    plt.close()



def kmeans_2d(data, clusters):
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(data)

    plt.figure(figsize=(8, 5))
    # plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=clusters, cmap='jet', marker='o')
    
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
    df.to_csv('kmeans_direct_cluster.csv', index=False)

    labels = df['label']

    texts = []
    for i, label in enumerate(labels):
        if label != 5 and label != 8 and label != 9:
            x, y = reduced_vectors[i, 0], reduced_vectors[i, 1]
            texts.append(plt.text(x, y, label, size=10))
        elif i % 2 == 0 and (label == 5 or label == 8 or label == 9):
            x, y = reduced_vectors[i, 0], reduced_vectors[i, 1]
            texts.append(plt.text(x, y, label, size=10))

    adjust_text(
        texts,
        expand_text=(5, 15),
        arrowprops=dict(arrowstyle='->', color='red', lw=0.5)
    )
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="upper left", bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig('./img/kmeans_direct_2d.jpg')
    plt.close()


def kmeans_3d(data, clusters):
    pca = PCA(n_components=3)
    data_3d = pca.fit_transform(data)

    fig = plt.figure(figsize=(10, 5))
    elevations = [20, 80]
    azimuths = [30, 210]

    unique_clusters = set(clusters)
    markers = ['o', 's', 'D', '^', '*', '+', 'x']
    colormap = ['palevioletred', 'darkmagenta', 'darkslateblue', 'blue', 'steelblue', 'darkturquoise', 'mediumseagreen', 'green', 'limegreen', 'yellow',
            'goldenrod', 'orange', 'red', 'brown', 'salmon', 'darkred', 'rosybrown']

    for i, (elev, azim) in enumerate(zip(elevations, azimuths)):
        ax = fig.add_subplot(1, len(elevations), i + 1, projection='3d')
    #     ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=clusters, cmap='jet', marker='o')

        unique_clusters = set(clusters)
        for i, label in enumerate(unique_clusters):
            color = colormap[i]
            marker = markers[i % len(markers)]
            mask = clusters == label
            ax.scatter(data_3d[mask, 0], data_3d[mask, 1], data_3d[mask, 2], color=color, label=f'Cluster {label}', marker=marker)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.view_init(elev=elev, azim=azim)

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.zaxis.set_major_locator(MaxNLocator(5))
        
    plt.tight_layout()
    plt.savefig('./img/kmeans_direct_3d.jpg')
    plt.close()


csv_file = './data/thing_train_data/sorted_kishino.csv'
df = pd.read_csv(csv_file)

eps = 0.2
n_clusters = 17

data = []
for id, label, arrival_time, dow, is_touch in zip(df["id"], df["label"], df["arrival_time"], df["day_of_week"], df["is_touch"]):
    label = label_quantization(label)
    arrival_time = dt_quantization(arrival_time)
    dow = dow_quantization(dow)
    is_touch = touch_quantization(is_touch)

    data.append([label, arrival_time, dow, is_touch])

data = np.array(data)
dbscan = DBSCAN(eps, min_samples=4)
clusters = dbscan.fit_predict(data)

dbscan_2d(data, clusters)
dbscan_3d(data, clusters)

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(data)

kmeans_2d(data, clusters)
kmeans_3d(data, clusters)
