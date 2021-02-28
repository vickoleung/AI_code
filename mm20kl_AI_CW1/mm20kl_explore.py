"""

Template code for COMP5623M CW1 Question 2

python explore.py --image_path XX --use_pre_trained True


"""

import argparse
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np



# Set up training arguments and parse
parser = argparse.ArgumentParser(description='Explore pre-trained AlexNet')


parser.add_argument(
    '--image_path', type=str,
    help='Full path to the input image to load.')
parser.add_argument(
    '--use_pre_trained', type=bool, default=True,
    help='Load pre-trained weights?')


args = parser.parse_args()

# Device configuration - defaults to CPU unless GPU is available on device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


print("=======================================")
print("                PARAMS               ")
print("=======================================")
for arg in vars(args):
    print(F"{arg:>20} {getattr(args, arg)}")


#########################################################################
#
#        QUESTION 2.1.2 code here
# 
#########################################################################


# Read in image located at args.image_path

image_path = "fish.jpg"
Image.open(image_path)


# Normalisations expected by pre-trained net, to apply in the image transform
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]




# Loads the model and downloads pre-trained weights if not already downloaded
model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

# To see the AlexNet architecture
print(model)


model.eval()


# Pass image through a single forward pass of the network

input_image = Image.open(image_path)

preprocess = transforms.Compose([
    #transforms.Resize(256),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
    ])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


outputs = model(input_batch)

print(outputs)

# layer indices of each conv layer in AlexNet
conv_layer_indices = [0, 3, 6, 8, 10]


def extract_filter(conv_layer_idx, model):

    
    the_filter = model.features[conv_layer_idx].weight

    return the_filter


def extract_feature_maps(model, input):
    feature_map = []
    index = np.array([0, 3, 6, 8, 10])
    
    def hook(module, input, output):
        feature_map.append(output.clone().detach())
        return None
    
    for i in index:
        # register hook for each relu layer
        handle = model.features[i+1].register_forward_hook(hook)
     
    output = model(input)
        
    return feature_map





# Normalize feature maps and filters
def normalize(inputs):
    mins = inputs.min(); maxs = inputs.max();
    inputs = (inputs - mins) / (maxs - mins)
    return inputs   

ft0 = extract_filter(0, model).cpu().detach().clone().numpy()
ft0 = normalize(ft0)
ft6 = extract_filter(6, model).cpu().detach().clone().numpy()
ft6 = normalize(ft6)
ft10 = extract_filter(10, model).cpu().detach().clone().numpy()
ft10 = normalize(ft10)  

fea_map = extract_feature_maps(model, input_batch)
for i in range(len(fea_map)):
    fea_map[i] = fea_map[i].cpu().detach().clone().numpy()
    fea_map[i] = normalize(fea_map[i])    


# Plot filters and feature maps

def plot_filter(ft, square):
    
    plt.figure(figsize = (50,50))
    
    for i, filter in enumerate(ft):
        plt.subplot(square, square, i+1)
        plt.imshow(filter[0,:,:], cmap = "gray")
        plt.axis('off')
        plt.savefig('filter.png')
    plt.show()
    
    return None

def plot_feature_map(idx, fea_map, square):
    
    plt.figure(figsize = (50,50))
    
    for i, fm in enumerate(fea_map[idx][0]):
        plt.subplot(square, square, i+1)
        plt.imshow(fm, cmap = "gray")
        plt.axis('off')
        plt.savefig('feature_maps.png')
    plt.show()
    
    return None

plot_filter(ft0, 8)
plot_feature_map(0, fea_map, 8)

plot_filter(ft6, 20)
plot_feature_map(2, fea_map, 20)

plot_filter(ft10, 16)
plot_feature_map(4, fea_map, 16)



        





