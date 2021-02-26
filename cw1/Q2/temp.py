"""

Template code for COMP5623M CW1 Question 2

python explore.py --image_path XX --use_pre_trained True


"""

import argparse
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt



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

image_path = "image1.JPEG"
input_image = Image.open(image_path)


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
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
    ])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_batch = input_batch.to(device)
model = model.to(device)

outputs = model(input_batch)

#print(outputs)

# layer indices of each conv layer in AlexNet
conv_layer_indices = [0, 3, 6, 8, 10]


def extract_filter(conv_layer_idx, model):

    
    the_filter = model.features[conv_layer_idx].weight

    return the_filter


def extract_feature_maps(input, model):

    all_feature_maps = []
    
    conv0 = model.features[0].forward(input)
    relu0 = model.features[1].forward(conv0)
    all_feature_maps.append(relu0)
    
    conv3 = model.features[3].forward(relu0)
    relu3 = model.features[4].forward(conv3)
    all_feature_maps.append(relu3)
    
    conv6 = model.features[6].forward(relu3)
    relu6 = model.features[7].forward(conv6)
    all_feature_maps.append(relu6)
    
    conv8 = model.features[8].forward(relu6)
    relu8 = model.features[9].forward(conv8)
    all_feature_maps.append(relu8)
    
    conv10 = model.features[10].forward(relu8)
    relu10 = model.features[11].forward(conv10)
    all_feature_maps.append(relu10)

    return all_feature_maps
    

def normalize(inputs):
    inputs = inputs - inputs.min()
    inputs = inputs / inputs.max()
    return inputs

fm0, fm3, fm6, fm8, fm10 = extract_feature_maps(input_batch, model)

fm = fm0.cpu().detach().clone().numpy()

plt.figure(figsize = (20, 17))
for i, fm in enumerate(fm[0]):
    plt.subplot(8,8,i+1)
    plt.imshow(fm)
    plt.savefig('filter.png')
plt.show()

     

filter0 = extract_filter(0, model)
filter0 = filter0.cpu().detach().clone().numpy()

plt.figure(figsize = (30, 30))
fig, axarr = plt.subplots(20,20)

for i in range(filter0.shape[0]):
    for j in range(filter0.shape[1]):
        axarr[i][j].imshow(filter0[i,j,:,:])


