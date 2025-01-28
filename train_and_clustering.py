from train_thing2vec import train_thing2vec
from dbscan import dbscan_plot
from kmeans import kmeans_plot
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--input_path', type=str, help='Input csv file path', default="./data/thing_train_data/sorted_kishino.csv")
    parser.add_argument('--batch_size', type=int, help='Batchsize in training', default=4)
    parser.add_argument('--learning_rate', type=float, help='Leaning rate in training', default=0.01)
    parser.add_argument('--num_epochs', type=int, help='Number of epochs in training', default=200)
    parser.add_argument('--save_epoch', type=int, help='How many epochs to save the model', default=10)
    parser.add_argument('--cuda', type=int, help='Cuda number to use', default=0) 
    parser.add_argument('--emb_dim', type=int, help='Cuda number to use', default=15)
    parser.add_argument('-i', '--num_items', type=int) # データ数
    parser.add_argument('--eps', type=float, default=0.25)
    parser.add_argument('--n_clusters', type=int, default=5)
    parser.add_argument('--model', type=str, default='./output/model/models/model200.pth')
    args = parser.parse_args()
    train_thing2vec(args.input_path, args.batch_size, args.learning_rate, args.num_epochs, 
                    args.save_epoch, args.cuda, args.emb_dim)
    
    # num_tokens = 24*2*6*5*2*5*5
    num_tokens = 24*2*6*5*2
    # num_tokens = 24*2*6*2

    dbscan_plot(args.num_items, args.emb_dim, num_tokens, args.eps, args.model)
    kmeans_plot(args.num_items, args.emb_dim, num_tokens, args.n_clusters, args.model)

