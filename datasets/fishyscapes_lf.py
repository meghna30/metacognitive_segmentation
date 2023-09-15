
import glob
from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.utils.data as data
import torchvision.transforms as transforms
import pdb

class FishyScapesLF(Dataset):

    def __init__(self, root, transform):
        "load all the file names"
        self.transform = transform
        self.root = root
        self.images = []
        self.targets = []

        filenames = os.listdir(os.path.join(self.root, 'original'))

        for image_file in filenames:
           
            self.images.append(os.path.join(self.root, 'original',image_file))
            self.targets.append(os.path.join(self.root, 'labels', image_file))


    def __len__(self):
        return(len(self.images))

    def __getitem__(self, idx):
        """Return raw image and trainIds as PIL image or torch.Tensor"""
        image = Image.open(self.images[idx]).convert('RGB')
        target = Image.open(self.targets[idx]).convert('L')

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target


if __name__=="__main__":

    transform = transforms.ToTensor()

    dataset = FishyScapes('data/fishyscapes/', transform = transform)
    trainloader = data.DataLoader(dataset, batch_size = 4, shuffle = False)

    for images, labels in trainloader:
        print("okay")
