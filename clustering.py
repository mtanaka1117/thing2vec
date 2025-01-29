from dbscan import dbscan_plot
from kmeans import kmeans_plot
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--emb_dim', type=int, help='Cuda number to use', default=15)
    parser.add_argument('-i', '--num_items', type=int)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--n_clusters', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--model', default='./output/model/models/model200.pth')
    args = parser.parse_args()
    
    # num_tokens = 24*2*6*5*2*5*5 #BBOX
    # num_tokens = 24*2*6*5*2 
    num_tokens = 24*2*6*2      #滞在時間なし
    # num_tokens = 24*2*6*2*5*5   #滞在時間なし＋BBOX

    dbscan_plot(args.num_items, args.emb_dim, num_tokens, args.eps, args.model)
    kmeans_plot(args.num_items, args.emb_dim, num_tokens, args.n_clusters, args.model)
