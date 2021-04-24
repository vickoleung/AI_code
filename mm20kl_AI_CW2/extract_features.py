"""
COMP5623M Coursework on Image Caption Generation


Forward pass through Flickr8k image data to extract and save features from
pretrained CNN.

"""


import torch
import numpy as np

import torch.nn as nn
from torchvision import transforms

from models import EncoderCNN
from datasets import Flickr8k_Images
from utils import *
from config import *
import string



lines = read_lines(TOKEN_FILE_TRAIN)
# see what is in lines
# print(lines[:2])

#########################################################################
#
#       QUESTION 1.1 Text preparation
# 
#########################################################################

image_ids, cleaned_captions = parse_lines(lines)
# to check the results after writing the cleaning function
# print(image_ids[:2])
# print(cleaned_captions[:2])

vocab = build_vocab(cleaned_captions)
# to check the results
print("Number of words in vocab:", vocab.idx)

# sample each image once
image_ids = image_ids[::5]


# crop size matches the input dimensions expected by the pre-trained ResNet
data_transform = transforms.Compose([ 
    transforms.Resize(224), 
    transforms.CenterCrop(224), 
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),   # using ImageNet norms
                         (0.229, 0.224, 0.225))])

dataset_train = Flickr8k_Images(
    image_ids=image_ids,
    transform=data_transform,
)

train_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=64,
    shuffle=False,
    num_workers=0,
)

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EncoderCNN().to(device)



#########################################################################
#
#        QUESTION 1.2 Extracting image features
# 
#########################################################################
features = []


# TODO loop through all image data, extracting features and saving them
# no gradients needed
with torch.no_grad():
    for data in train_loader:
        inputs = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        features.append(outputs)

for i in range(len(features)):
    features[i] = features[i].squeeze()

# Concatenate all features vertically   
features = torch.cat(features, dim=0)

# to check your results, features should be dimensions [len(train_set), 2048]
# convert features to a PyTorch Tensor before saving
print(features.shape)


# save features
torch.save(features, "features.pt")


