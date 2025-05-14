from dtaidistance import dtw_ndim
import similaritymeasures
import data.CMAPSSDataset as CMAPSSDataset
from torch.utils.data import DataLoader,TensorDataset,random_split
import numpy as np
import pandas as pd
from args import args

def cal_dist_dtw(original_data, pred_data):
    len_data = len(pred_data)
    dist = 0
    print('------------calculate DTW distance------------')
    if len(original_data) == len(pred_data):
        for i in range(len_data):
            dist += dtw_ndim.distance(original_data[i], pred_data[i])
            # dist += similaritymeasures.dtw(original_data[i], pred_data[i])[0]
        dist /= len_data
    else:
        split_data = np.random.shuffle(original_data)[0:len_data]
        for i in range(len_data):
            dist += dtw_ndim.distance(split_data[i], pred_data[i])
        dist /= len_data
    return dist

def cal_dist_fd(original_data, pred_data):
    len_data = len(pred_data)
    dist = 0
    print('------------calculate Frechet distance------------')
    if len(original_data) == len(pred_data):
        for i in range(len_data):
            dist += similaritymeasures.frechet_dist(original_data[i], pred_data[i])
        dist /= len_data
    else:
        split_data = np.random.shuffle(original_data)[0:len_data]
        for i in range(len_data):
            dist += similaritymeasures.frechet_dist(split_data[i], pred_data[i])
        dist /= len_data
    return dist

if __name__ == '__main__':
    args.dataset = 'FD003'
    loaded_data = np.load('./weights/syn_data/'+ args.dataset +'.npz')
    pred_data = np.transpose(loaded_data['data'], (0,2,1)) 
    datasets = CMAPSSDataset.CMAPSSDataset(fd_number=args.dataset, sequence_length=48,deleted_engine=[1000])
    train_data = datasets.get_train_data()
    train_data = datasets.get_feature_slice(train_data).numpy()
    
    # pred_data = np.random.rand(*train_data.shape)
    random_indices = np.random.choice(train_data.shape[0], size=len(train_data) //10 , replace=False)
    
    print(train_data.shape)
    print(pred_data.shape)
    dtw_dist = cal_dist_dtw(train_data[random_indices], pred_data[random_indices])
    fd_dist =cal_dist_fd(train_data[random_indices], pred_data[random_indices])
    print(dtw_dist,fd_dist)