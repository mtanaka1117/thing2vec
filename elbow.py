import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn
from libs.model import Thing2Vec


num_items = 75
embed_size = 5
num_tokens = 24*2*6*2 
model_path = './output/model/models/model200.pth'

model = Thing2Vec(num_items, embed_size, num_tokens)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.load_model(model_path)
model.eval()

item_indices = torch.arange(num_items).to(device) 
with torch.no_grad():
    embeddings = model.embedding(item_indices).cpu()

normalized_embeddings = F.normalize(embeddings, p=2, dim=0).numpy()


sse = []
k_values = range(1, 30)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(normalized_embeddings)
    sse.append(kmeans.inertia_)

# SSEをプロット
plt.figure(figsize=(10, 6))
plt.plot(k_values, sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.title('Elbow Method For Optimal Number of Clusters')
plt.savefig('elbow.jpg')