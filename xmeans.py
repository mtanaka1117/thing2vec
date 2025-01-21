# 必要なライブラリのインポート
from pyclustering.cluster.xmeans import xmeans, kmeans_plusplus_initializer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import SIMPLE_SAMPLES
from pyclustering.cluster import cluster_visualizer
from libs.model import Thing2Vec
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

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

# クラスタ数初期化
initial_centers = kmeans_plusplus_initializer(normalized_embeddings, 2).initialize()

# X-Meansアルゴリズムの適用
xmeans_instance = xmeans(normalized_embeddings, initial_centers, kmax=10, ccore=True)
xmeans_instance.process()

# クラスタリング結果の取得
clusters = xmeans_instance.get_clusters()
final_centers = xmeans_instance.get_centers()
final_centers = np.array(final_centers)

# 結果の出力
print(f"最終的なクラスタ数: {len(clusters)}")
print(f"各クラスタのサイズ: {[len(cluster) for cluster in clusters]}")


colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'pink', 'brown', 'grey', 'yellow']  # 最大10クラスタの色
plt.figure(figsize=(8, 6))
for i, cluster in enumerate(clusters):
    cluster_points = normalized_embeddings[cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i % len(colors)], label=f'Cluster {i+1}')

# クラスタの中心をプロット
plt.scatter(final_centers[:, 0], final_centers[:, 1], color='black', marker='*', s=200, label='Centers')

# プロットの装飾
plt.title("X-Means Clustering Result")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid()

# 結果をPNGとして保存
plt.savefig("xmeans_clustering_result.png", dpi=300, bbox_inches='tight')
plt.close()  # 表示せずに保存して終了