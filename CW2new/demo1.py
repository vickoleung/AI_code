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

build_vocab(cleaned_captions)