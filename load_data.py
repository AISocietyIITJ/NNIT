from torch.utils.data import Dataset, DataLoader


def load_data(directory_path,patch_size,image_size):
  '''
  Takes: the path of a directory with images, patch_size,image_size
  Returns :a Torch DataLoader object with images in format
          which can be directly passed to a VIT style encoder
  '''

#!pip install einops
#!pip install opendatasets
#!pip install pandas
import opendatasets as od
import pandas
od.download("https://www.kaggle.com/datasets/landlord/handwriting-recognition")

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

training_data_path="/content/handwriting-recognition/train_v2"
training_data=[]
import os
for label in os.listdir(training_data_path):#os.listdir(training_data_path) gives list of list of 0 images,1 images etc =[0,1,2,3,4,5,6,7,8,9,'A','B',....,'Z']
  for j in os.listdir(os.path.join(training_data_path,label)):
    training_data.append([os.path.join(training_data_path,label,j),label])

LABELS={'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25}

# resize to imagenet size
transform = Compose([Resize((224, 224)), ToTensor()])

from torch.utils.data import Dataset,DataLoader
class Train_Dataset(Dataset):
  def __init__(self,img_list):
    self.img_list=img_list
  def __getitem__(self,index):
    path=self.img_list[index][0]
    label=self.img_list[index][1]
    one_hot= np.eye(26)[LABELS[label]]
    img=Image.open(path)
    img=transform(img)
    x=transform(img)
    x = x.unsqueeze(0)
    return [x,one_hot]     #return[img,label]
  def __len__(self):
    return len(self.img_list)

train_dataloader=DataLoader(Train_Dataset(training_data))


testing_data_path="/content/handwriting-recognition/test_v2"
testing_data=[]
import os
for label in os.listdir(testing_data_path):#os.listdir(testing_data_path) gives list of list of 0 images,1 images etc =[0,1,2,3,4,5,6,7,8,9,'A','B',....,'Z']
  for j in os.listdir(os.path.join(testing_data_path,label)):
    testing_data.append([os.path.join(testing_data_path,label,j),label])

class Test_Dataset(Dataset):
  def __init__(self,img_list):
    self.img_list=img_list
  def __getitem__(self,index):
    path=self.img_list[index][0]
    label=self.img_list[index][1]
    one_hot= np.eye(26)[LABELS[label]]
    img=Image.open(path)
    img=transform(img)
    x=transform(img)
    x = x.unsqueeze(0)
    return [x,one_hot]     #return[img,label]
  def __len__(self):
    return len(self.img_list)

test_dataloader=DataLoader(Test_Dataset(testing_data))


