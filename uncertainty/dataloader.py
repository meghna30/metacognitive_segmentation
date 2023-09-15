import glob
import os
import cv2
import numpy as np
import torch
import torchvision.datasets as datasets
import torch.utils.data as data

from torch.utils.data import Dataset

import torchvision.transforms as transforms
import torch.nn.functional as F

import pdb

class CustomDataset(Dataset):
    def __init__(self, data_path, split, transform, no_channels = 4):
        self.data_path = data_path
        self.data_list = glob.glob(self.data_path + "*")
        self.transform = transform
        self.no_channels = no_channels
        self.split = split ## id or ood
        self.channels = ['entropy', 'pred', 'var_ratio', 'prob_margin']

    def __len__(self):
        return len(self.data_list)


    def loader(self, path):
        data = np.load(path)
        label_id = data['label']
        self.image_hw = data['entropy'].shape
        # image = np.zeros((label_id.shape[0], label_id.shape[1], self.no_channels,))
        image = np.zeros((self.image_hw[0], self.image_hw[1], self.no_channels))
        image[:,:,0] = data['entropy']
        image[:,:,1] = data['pred']
        #image[:,:,1] = data['var_ratio']



        if self.no_channels > 4:
            image[:,:,1:] = np.transpose(data['scores'],(1,2,0))
            # image[:,:,1:] = np.transpose(data['feats'],(1,2,0))

        if self.split == 'id':
            target = data['pred'] == label_id
        elif self.split == 'ood':
            target = np.where(label_id == 2, 0, 1) #lostandfound 
            target = np.where(label_id == 1, 0 , 1) #fishyscapes 
            
        else:
            print("incorrect split")
        mask = np.where(label_id == 255, 0, 1).astype(bool)
        ## flipping target
        target = 1 - target

        return image, target, mask, label_id, data['pred'],data['entropy']

    def __getitem__(self, idx):
        path = self.data_list[idx]

        img, target, mask, label_id_true, label_id_pred, ent = self.loader(path)
        if img is None:
            print(path)
        if self.transform is not None:
            img = self.transform(img)
        target = torch.tensor(target)

        return img, target, mask, label_id_true, label_id_pred, ent

if __name__ == "__main__":


    trainloader = data.DataLoader(trainset, batch_size = 32, shuffle = False)
    # testloader  = data.Dataloader(testset,  batch_siz = 32, shuffle = False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
 
