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

num_items = 173
embed_size = 5
num_output_tokens = 24*2*6*2
model_path = './output/model/models/model200.pth'

model = Thing2Vec(num_items, embed_size, num_output_tokens)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.load_model(model_path)
model.eval()

item_indices = torch.arange(num_items).to(device)
with torch.no_grad():
    embeddings = model.embedding(item_indices).cpu()
normalized_embeddings = embeddings.numpy()

csv_file = './data/thing_train_data/sorted_kishino.csv'
df_label = pd.read_csv(csv_file)
label = df_label['label']
datetime = df_label['arrival_time']
is_touch = df_label['is_touch']

df = pd.DataFrame(normalized_embeddings)
df['label'] = label
df['dt'] = datetime
df['is_touch'] = is_touch
df.to_csv('output.csv', index=False)

