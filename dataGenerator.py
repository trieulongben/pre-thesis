import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torch



""" Generate dataset based on rank from 0 to len(unique(datalabel))"""
class plantdisease(Dataset):
    def __init__(self, train_df='../code/data/raw/train.csv',test_df='../code/data/raw/test.csv'
                 , img_dir='../code/data/oneFolder'
                 , transform=None, 
                 target_transform=None,resize=False,isTest=False,rank=0):
        #if isTest==True:
        #    self.img_labels=pd.read_csv(test_df)
        #elif isTest==False:
        #    self.img_labels=pd.read_csv(train_df)
        self.img_labels=pd.read_csv('../code/data/raw/annotation.csv')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.resize=resize
        self.rank=rank
        self.img_labels=self.make_newCSV()
    def __len__(self):
        return len(self.img_labels)
    def make_newCSV(self):
        old_csv=self.img_labels
        newCSV=old_csv[old_csv.iloc[:,2]==(self.rank)]
        return newCSV
    def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
            image = (read_image(img_path).float())

            label = self.img_labels.iloc[idx, 2]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label