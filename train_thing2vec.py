import argparse
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from libs.train_utils import train_with_anchoring, train_without_anchoring, Dataset, FeatureQuantization
import torch.nn as nn
from libs.model import Thing2Vec
import torch

def train_thing2vec(input_path, batch_size, learning_rate, num_epochs, save_epoch, cuda, emb_dim):
    item_df = pd.read_csv(input_path)
    quantization = FeatureQuantization()
    dataset = Dataset(quantization)
    dataset.gen_dataset(item_df)
    
    # load anchor embedding
    initial_embedding_weight = torch.rand(dataset.num_items, emb_dim)
    
    # define model
    device = torch.device('cuda:' + str(cuda) if torch.cuda.is_available() else 'cpu')
    model = Thing2Vec(
        num_items=dataset.datasize,
        embed_size=emb_dim,
        num_output_tokens=dataset.num_tokens,
        device=device
    )
    model.initialize_weights(embedding_weight=initial_embedding_weight, freeze_anchor_num=0)
    model = model.to(device)
    
    # train_with_anchoring(
    #     model, 
    #     dataset,  
    #     save_path = "./output/sample_model/", 
    #     batch_size=batch_size, 
    #     learning_rate=learning_rate, 
    #     num_epochs=num_epochs, 
    #     save_epoch=save_epoch, 
    #     weight_type=weight_type, 
    #     alpha=alpha, 
    #     beta=beta
    #     )
    
    # If you don't need anchoring
    train_without_anchoring(
        model,
        dataset,
        save_path = "./output/model/", 
        batch_size=batch_size, 
        learning_rate=learning_rate, 
        num_epochs=num_epochs, 
        save_epoch=save_epoch
        )
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--input_path', type=str, help='Input csv file path', default="./data/thing_train_data/sorted_kishino.csv")
    parser.add_argument('--batch_size', type=int, help='Batchsize in training', default=4)
    parser.add_argument('--learning_rate', type=float, help='Leaning rate in training', default=0.01)
    parser.add_argument('--num_epochs', type=int, help='Number of epochs in training', default=200)
    parser.add_argument('--save_epoch', type=int, help='How many epochs to save the model', default=10)
    # parser.add_argument('--alpha', type=float, help='Initial anchor power', default=0.3)
    # parser.add_argument('--beta', type=float, help='Final anchor power', default=1.0)
    # parser.add_argument('--weight_type', type=str, help='Weight function for anchor power', default="exponential")   
    parser.add_argument('--cuda', type=int, help='Cuda number to use', default=0) 
    parser.add_argument('--emb_dim', type=int, help='Cuda number to use', default=10)
    args = parser.parse_args()
    train_thing2vec(args.input_path, args.batch_size, args.learning_rate, args.num_epochs, 
                    args.save_epoch, args.cuda, args.emb_dim)
