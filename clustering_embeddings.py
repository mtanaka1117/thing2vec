from keplergl import KeplerGl
import json
from tqdm.notebook import tqdm
import numpy as np
import torch
import torch.nn as nn
from libs.train_utils_area2vec import FeatureQuantization
from libs.math_utils import do_kmeans, assign_to_nearest_anchor
from libs.visualize import StackedFeatureGrapher


data_name = "2021"
state_dict = torch.load("../data/japan_city/model/"+data_name+".pth")
with open("../data/japan_city/geojson/"+data_name+".geojson", "r") as f:
    mesh_data = json.load(f)
embeddings = torch.tensor([mesh["properties"]["vector"] for mesh in mesh_data["features"]])

cluster_num = 8
result, element_num_in_cluster = do_kmeans(cluster_num, embeddings, with_center = False, seed = 20210401)

quantization = FeatureQuantization()
_, decoder = state_dict["embedding.weight"], state_dict["decode_linear.weight"]
softmax = nn.Softmax(dim = 1)
stay_weight_matrix = np.load("../data/util_file/stay_weight_matrix.npy")

approximation_stats = softmax(torch.matmul(embeddings, decoder.T))
stay_each_mesh_dow_dt_e =approximation_stats.numpy().dot(stay_weight_matrix).reshape(embeddings.shape[0], 2, 12, 7)
stay_each_mesh_dow_e_dt =np.transpose(stay_each_mesh_dow_dt_e, (0, 1, 3, 2))

visualizer = StackedFeatureGrapher(FeatureQuantization(), day_counts = [22,8])
sorted_indices = visualizer.aggregate_data_from_approximated_features(stay_each_mesh_dow_e_dt, result, sort=True)#sort clusters
result = [-1 if cls == -1 else sorted_indices.index(cls) for cls in result]
visualizer.visualize(visualizer.elapsed_each_time_each_cluster, title="Elapsed")