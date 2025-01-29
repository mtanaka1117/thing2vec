import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn
from libs.model import Thing2Vec
import argparse
from sklearn.metrics import silhouette_samples
from matplotlib import cm


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

class_data_list = []
result_class_num = 0

# 2〜データ個数の9割りまでクラスタ数を指定する
loop_num = int(len(normalized_embeddings) * 0.9)
for class_num in range(2, loop_num):
    # クラスタ分類
    km = KMeans(n_clusters=class_num,
            init='k-means++',     
            n_init=10,
            max_iter=300,
            random_state=0)
    y_km = km.fit_predict(normalized_embeddings) 
    
    cluster_labels = np.unique(y_km) 

    # 配列の数
    n_clusters = cluster_labels.shape[0] 

    #シルエット係数を計算
    silhouette_vals = silhouette_samples(normalized_embeddings,y_km,metric='euclidean')
    
    # クラスタ内のデータ数
    sil=[]
    for i,c in enumerate(cluster_labels):
        c_silhouette_vals=silhouette_vals[y_km==c]
        sil.append(len(c_silhouette_vals))
    
    # クラスタ内のデータ数の差がデータ数の2割以下であれば分割できたとみなす
    data_diff = int(len(normalized_embeddings) * 0.2)
    data_diff_flg = max(sil)-min(sil) < data_diff
    # クラスタ内のシルエット係数平均値
    ave_silhouette_vals = np.average(silhouette_vals)
    
    class_data_list.append({'class_num':class_num, 'data_diff':data_diff_flg, 'ave':ave_silhouette_vals})
    
        
max_ave = 0
for class_data in class_data_list:
    if class_data['data_diff'] and (max_ave < class_data['ave']):
        max_ave = class_data['ave']
        result_class_num = class_data['class_num']
        
print(result_class_num)
print(max_ave)