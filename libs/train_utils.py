import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import datetime
import os
import torch.nn as nn
from torch import optim

class FeatureQuantization:
    def __init__(self, label_quant_num=28, dow_quant_num=2, dt_quant_num=6, e_quant_num=4, touch_quant_num=2):
        super().__init__()

        self.label_quant_num = label_quant_num
        self.dow_quant_num = dow_quant_num # day_of_week
        # self.month_quant_num = month_quant_num
        # self.date_quant_num = date_quant_num
        self.dt_quant_num = dt_quant_num # 時間帯
        self.e_quant_num = e_quant_num # 滞在時間
        self.touch_quant_num = touch_quant_num
        
        self.quant_num = label_quant_num*dow_quant_num*dt_quant_num*e_quant_num*touch_quant_num

        self.dow_dic = {"Sunday": 0, "Monday": 1, "Tuesday": 2,
                        "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6}
        self.rev_dow_dic = {v: k for k, v in self.dow_dic.items()}
        
        # upper limits of elapsed time for each token
        self.e_thresholds = [20, 60, 120, 720]


    def label_quantization(self, label):
        return label

    # 滞在時間の量子化
    def e_quantization(self, e):
        e_token = 0
        for e_token, thre in enumerate(self.e_thresholds):
            if e <= thre:
                return e_token
        return self.e_quant_num - 1
    
    # 月日の量子化
    # def month_quantization(self, date):
    #     month = int(date.split("-")[0])
    #     return month
    
    # def date_quantization(self, date):
    #     date = int(date.split("-")[1])
    #     return date
    
    
    # 時刻の量子化
    def dt_quantization(self, dt):
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
        hour = int(dt.split(":")[0])
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
    def dow_quantization(self, dow):
        if self.dow_quant_num == 2:
            if dow in ["Saturday", "Sunday"]:
                return 1
            else:
                return 0
        elif self.dow_quant_num == 8:
            return self.dow_dic[dow] 
    
    
    # 触れたかどうか
    def touch_quantization(self, is_touch):
        if is_touch:
            return 1
        else:
            return 0


    # 量子化
    def quantization(self, label, day_of_week, date_time, elapsed_time, is_touch):
        token = int(0)
        
        token += self.label_quant_num * self.dow_quant_num * self.dt_quant_num * self.e_quant_num * self.touch_quantization(is_touch)
        token += self.dow_quant_num * self.dt_quant_num * self.e_quant_num * self.label_quantization(label)
        token += self.dt_quant_num * self.e_quant_num * self.dow_quantization(day_of_week)
        token += self.e_quant_num*self.dt_quantization(date_time)
        token += self.e_quantization(elapsed_time)
        return token


    def dequantization(self, token):
        elapsed_token = token % self.e_quant_num
        token //= self.e_quant_num

        date_time_token = token % self.dt_quant_num
        token //= self.dt_quant_num
        
        dow_token = token % self.dow_quant_num
        token //= self.dow_quant_num
        
        label_token = token % self.label_quant_num
        token //= self.label_quant_num
        
        touch_token = token % self.touch_quant_num
        
        return label_token, dow_token, date_time_token, elapsed_token, touch_token


class Dataset:
    def __init__(self, quantization=None):
        # Initialize the Dataset with optional quantization and user count data.
        # If no quantization object is provided, create a default FeatureQuantization object.
        if quantization is None:
            quantization = FeatureQuantization(label_quant_num=28, dow_quant_num=2, dt_quant_num=6, e_quant_num=4, touch_quant_num=2)
        self.quantization = quantization  # Object that handles the quantization process.
        self.num_tokens = self.quantization.quant_num  # Number of quantization tokens.
        
    def gen_dataset(self, df, num_items=None):
        # Generate a dataset from a DataFrame, filtering out invalid item IDs and applying quantization.
        self.dataset = []  # Initialize an empty list to store dataset entries.
        for id, label, arrival_time, stay_time, dow, is_touch in zip(df["id"], df["label"], df["arrival_time"], df["stay_time"], df["day_of_week"], df["is_touch"]):
            token = self.quantization.quantization(label, dow, arrival_time, stay_time, is_touch)  # Quantize the features.
            self.dataset.append((id, token))  # Append the item ID and token to the dataset.
        
        self.dataset = torch.tensor(self.dataset)  # Convert the dataset list to a tensor.
        # Determine the number of itemes if not provided.
        if num_items is None:
            self.num_items = int(self.dataset[:, 0].max() + 1)
        else:
            self.num_items = num_items
        self.datasize = len(self.dataset)
        
        
    def gen_anchor_dataset(self, anchor_df):
        if self.quantization == None:
            self.quantization = FeatureQuantization()
        self.anchor_dataset = []
        for m, e, sdt, dow, ih in zip(anchor_df["anchor_id"], anchor_df["stay_time"], anchor_df["arrival_time"], anchor_df["day_of_week"], anchor_df["is_holiday"]):
            token = self.quantization.quantization(dow, ih, sdt, e)  # Quantize the features.
            self.anchor_dataset.append((m, token))  # Append the item ID and token to the dataset.

        self.anchor_dataset = torch.tensor(self.anchor_dataset)  # Convert the dataset list to a tensor.
        self.num_anchors = int(self.anchor_dataset[:, 0].max() + 1)
        self.anchor_datasize = len(self.anchor_dataset)
        
        self.anchor_dataset[:, 0] = self.anchor_dataset[:, 0] + self.num_items
        self.num_items = self.num_items + self.num_anchors
            

# Functions for train Area2Vec
def initialize_save_path(save_path):
    if save_path is None:
        today_date = str(datetime.datetime.today().date())
        save_path = f"../output/{today_date}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path + "models", exist_ok=True)
        os.makedirs(save_path + "fig", exist_ok=True)
        os.makedirs(save_path + "log", exist_ok=True)
    return save_path


def train(model, dataset, save_path=None, batch_size=1024, learning_rate=0.01, num_epochs=100, save_epoch=10):
    save_path = initialize_save_path(save_path)
    writer = SummaryWriter(log_dir=save_path + "log")
    torch.save(model.state_dict(), save_path + "models/model" + str(0) + ".pth")  # initial weight

    dataset = dataset.to(model.device)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
    criterion_category = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        loss_epoch = 0.0
        itr = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            p = model(batch[:, 0])
            loss = criterion_category(p, batch[:, 1])
            loss.backward()
            loss_epoch += float(loss)
            itr += 1
            optimizer.step()
        writer.add_scalar("loss", float(loss_epoch) / itr, epoch)
        if (epoch + 1) % save_epoch == 0:
            torch.save(model.state_dict(), save_path + "models/model" + str(epoch+1) + ".pth")
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss_epoch/itr))


def calculate_stability_weight(epoch, epochs, alpha=0.1, beta=1.0, weight_type="exponential",datasize=None, anchor_datasize=None):
    if weight_type == "linear":
        w = -(beta-alpha)
        b = beta
        return w * epoch/(epochs-1) + b
    elif weight_type == "exponential":
        w = -(np.log(beta)-np.log(alpha))
        b = np.log(beta)
        return np.exp(w* epoch /(epochs - 1) + b)
    elif weight_type == "constant":
        return alpha
    else:
        # same as just mixing　anchor data
        return anchor_datasize / (datasize + anchor_datasize)


def train_with_anchoring(model, dataset, save_path=None, batch_size=1024, learning_rate=0.01, num_epochs=100, save_epoch=10, weight_type="exponential", alpha=0.1, beta=1.0):
    save_path = initialize_save_path(save_path)
    writer = SummaryWriter(log_dir=save_path + "log")
    torch.save(model.state_dict(), os.path.join(save_path, "models/model0.pth"))
    
    dataset.dataset = dataset.dataset.to(model.device)
    dataset.anchor_dataset = dataset.anchor_dataset.to(model.device)
    train_loader = torch.utils.data.DataLoader(dataset.dataset, batch_size=batch_size, shuffle=True)
    batchsize_anchor = int(dataset.anchor_datasize // (dataset.datasize / batch_size))
    if batchsize_anchor > batch_size:
        batchsize_anchor = batch_size
    train_loader_anchor = torch.utils.data.DataLoader(dataset.anchor_dataset, batch_size=batchsize_anchor, shuffle=True)

    criterion_category = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in tqdm(range(num_epochs)):
        loss_epoch, loss_data_epoch, loss_anchor_epoch = 0.0, 0.0, 0.0
        itr = 0
        s = calculate_stability_weight(epoch, num_epochs, alpha, beta, weight_type, datasize = dataset.datasize, anchor_datasize = dataset.anchor_datasize)
    
        for batch, batch_a in zip(train_loader, train_loader_anchor):
            optimizer.zero_grad()
            p = model(batch[:, 0])
            p_a = model(batch_a[:, 0])
            loss_data = criterion_category(p, batch[:, 1])
            loss_anchor = criterion_category(p_a, batch_a[:, 1])
            loss = (1-s) * loss_data + s * loss_anchor
            loss.backward()
            optimizer.step()
            
            loss_epoch += float(loss)
            loss_data_epoch += float(loss_data)
            loss_anchor_epoch += float(loss_anchor)
            itr += 1

        writer.add_scalar("loss", loss_epoch / itr, epoch)
        writer.add_scalar("loss_data", loss_data_epoch / itr, epoch)
        writer.add_scalar("loss_anchor", loss_anchor_epoch / itr, epoch)
        if (epoch + 1) % save_epoch == 0:
            torch.save(model.state_dict(), save_path + "models/model" + str(epoch+1) + ".pth")
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss_epoch/itr))


def train_without_anchoring(model, dataset, batch_size=1024, learning_rate=0.01, num_epochs=100, save_path=None, save_epoch=10):
    save_path = initialize_save_path(save_path)
    writer = SummaryWriter(log_dir=save_path + "log")
    torch.save(model.state_dict(), os.path.join(save_path, "models/model0.pth"))
    
    dataset.dataset = dataset.dataset.to(model.device)
    dataset.anchor_dataset = dataset.anchor_dataset.to(model.device)
    train_loader = torch.utils.data.DataLoader(dataset.dataset, batch_size=batch_size, shuffle=True)

    criterion_category = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in tqdm(range(num_epochs)):
        loss_epoch = 0.0
        itr = 0
        for batch in train_loader:
            optimizer.zero_grad()
            p = model(batch[:, 0])
            loss = criterion_category(p, batch[:, 1])
            loss.backward()
            optimizer.step()
            loss_epoch += float(loss)
            itr += 1
        writer.add_scalar("loss", float(loss_epoch) / itr, epoch)
        if (epoch + 1) % save_epoch == 0:
            torch.save(model.state_dict(), save_path + "models/model" + str(epoch+1) + ".pth")
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss_epoch/itr))