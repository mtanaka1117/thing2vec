import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.nn.functional as F
import torch
import torch.nn as nn
from libs.train_utils import FeatureQuantization
from libs.model import Thing2Vec
import pandas as pd
from adjustText import adjust_text


num_items = 64
embed_size = 10
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

# with open('./output.csv', "w") as f:
#     for idx, vector in enumerate(embeddings):
#         vector_str = ",".join(map(str, vector))  # スペース区切りに変換
#         f.write(f"{idx},{vector_str}\n") 


n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(normalized_embeddings)

pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(normalized_embeddings)

plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels, marker='o')
plt.colorbar()


csv_file = './data/thing_train_data/analysis_kishino.csv'
df = pd.read_csv(csv_file)
labels = df["label"].tolist()


texts = []
for i, label in enumerate(labels):
    x, y = reduced_vectors[i, 0], reduced_vectors[i, 1]
    texts.append(plt.text(x, y, (i, label), size=10))


adjust_text(
    texts,
    expand_text=(1.2, 1.2),
    arrowprops=dict(arrowstyle='->', color='red')
)

plt.title("Clustered Embedding Vectors")
plt.savefig('pca2d.png')
