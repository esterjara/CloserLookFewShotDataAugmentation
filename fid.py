#!/usr/bin/env python3
"""
FID: Fréchet inception distance
Metric used to assess the quality of images created by a generative model
"""
__author__ = "Ester Jara Lorente"
__since__ = "2022/05/25"

import tensorflow as tf
import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
import math
import tqdm

from data.dataset import SetDataset, EpisodicBatchSampler
from data.datamgr import DataManager, TransformLoader

labels = ["n01532829", "n01558993", "n01704323", "n01749939", "n01770081", "n01843383", "n01910747", "n02074367", "n02089867", "n02091831", "n02101006", "n02105505", "n02108089", "n02108551", "n02108915", "n02111277", "n02113712", "n02120079", "n02165456", "n02457408", "n02606052", "n02687172", "n02747177", "n02795169", "n02823428", "n02966193", "n03017168", "n03047690", "n03062245", "n03207743", "n03220513", "n03337140", "n03347037", "n03400231", "n03476684", "n03527444", "n03676483", "n03838899", "n03854065", "n03888605", "n03908618", "n03924679", "n03998194", "n04067472", "n04243546", "n04251144", "n04258138", "n04275548", "n04296562", "n04389033", "n04435653", "n04443257", "n04509417", "n04515003", "n04596742", "n04604644", "n04612504", "n06794110", "n07584110", "n07697537", "n07747607", "n09246464", "n13054560", "n13133613", "n01855672", "n02091244", "n02114548", "n02138441", "n02174001", "n02950826", "n02971356", "n02981792", "n03075370", "n03417042", "n03535780", "n03584254", "n03770439", "n03773504", "n03980874", "n09256479", "n01930112", "n01981276", "n02099601", "n02110063", "n02110341", "n02116738", "n02129165", "n02219486", "n02443484", "n02871525", "n03127925", "n03146219", "n03272010", "n03544143", "n03775546", "n04146614", "n04149813", "n04418357", "n04522168", "n07613480"]
classes = {j:i for i,j in enumerate(labels)}
identity = lambda x:x


class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self,i):
        image_path = os.path.join(self.meta['image_names'][i])
        label = classes[image_path.split('/')[-2]]
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, label

    def __len__(self):
        return len(self.meta['image_names'])

class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):        
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(data_file, transform)
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 12, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class NovelDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self,i):
        image_path = os.path.join(list(self.meta.keys())[i])
        label = classes[image_path.split('/')[-2].split('_')[0]]
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.meta.keys())

class NovelDataManager(DataManager):
    def __init__(self, image_size, batch_size):        
        super(NovelDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = NovelDataset(data_file, transform)
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 12, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

inception_model = tf.keras.applications.InceptionV3(include_top=False, 
                              weights="imagenet", 
                              pooling='avg')
BATCH_SIZE = 32
def compute_embeddings(dataloader, count):
    image_embeddings = []

    for _ in range(count):
        img, label = next(iter(dataloader))
        img = img.squeeze()
        img = img.reshape((32,84,84,3))
        img = img.numpy()
        images = tf.convert_to_tensor(img)
        print(images.shape)
        embeddings = inception_model.predict(images)

        image_embeddings.extend(embeddings)

    return np.array(image_embeddings)

count = math.ceil(10000/BATCH_SIZE)

# data loader for real images
loadfile = '/mnt/home/CloserLookFewShot/novel.json'  
image_size = 84
datamgr         = SimpleDataManager(image_size, batch_size = 32)         # batch_size=64
trainloader     = datamgr.get_data_loader(loadfile, aug = False)

# data loader for generated images
loadfile = '/mnt/home/CloserLookFewShot/generated_images.json'  
image_size = 84
datamgr         = NovelDataManager(image_size, batch_size = 32)         # batch_size=64
genloader     = datamgr.get_data_loader(loadfile, aug = False)
print(len(genloader))

print('real')
# compute embeddings for real images
real_image_embeddings = compute_embeddings(trainloader, count)

print('generated')
# compute embeddings for generated images
generated_image_embeddings = compute_embeddings(genloader, count)

def calculate_fid(real_embeddings, generated_embeddings):
     # calculate mean and covariance statistics
     mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
     mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings,  rowvar=False)
     # calculate sum squared difference between means
     ssdiff = np.sum((mu1 - mu2)**2.0)
     # calculate sqrt of product between cov
     covmean = linalg.sqrtm(sigma1.dot(sigma2))
     # check and correct imaginary numbers from sqrt
     if np.iscomplexobj(covmean):
       covmean = covmean.real
     # calculate score
     fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
     return fid

fid = calculate_fid(real_image_embeddings, generated_image_embeddings)

print(fid)