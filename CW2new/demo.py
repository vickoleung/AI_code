"""
COMP5623M Coursework on Image Caption Generation


python decoder.py


"""

import torch
import numpy as np

import torch.nn as nn
from torch.nn.modules.loss import TripletMarginWithDistanceLoss
from torch.nn.modules.pooling import AvgPool1d
from torch.utils import data
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity


from datasets import Flickr8k_Images, Flickr8k_Features
from models import DecoderRNN, EncoderCNN
from utils import *
from config import *


# if false, train model; otherwise try loading model from checkpoint and evaluate
#EVAL = False
EVAL = True


# reconstruct the captions and vocab, just as in extract_features.py
lines = read_lines(TOKEN_FILE_TRAIN)
image_ids, cleaned_captions = parse_lines(lines)
vocab = build_vocab(cleaned_captions)


# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# initialize the models and set the learning parameters
decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS).to(device)


if not EVAL:

    # load the features saved from extract_features.py
    print(len(lines))
    features = torch.load('features.pt', map_location=device)
    print("Loaded features", features.shape)

    features = features.repeat_interleave(5, 0)
    print("Duplicated features", features.shape)

    dataset_train = Flickr8k_Features(
        image_ids=image_ids,
        captions=cleaned_captions,
        vocab=vocab,
        features=features,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64, # change as needed
        shuffle=True,
        num_workers=0, # may need to set to 0
        collate_fn=caption_collate_fn, # explicitly overwrite the collate_fn
    )


    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)

    print(len(image_ids))
    print(len(cleaned_captions))
    print(features.shape)


#########################################################################
#
#        QUESTION 1.3 Training DecoderRNN
# 
#########################################################################

    # TODO write training loop on decoder here


    # for each batch, prepare the targets using this torch.nn.utils.rnn function
    # targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
    
    for i in range(NUM_EPOCHS):
        for data in train_loader:
            images, targets, lengths = data

            images = images.to(device); targets = targets.to(device)

            optimizer.zero_grad()

            outputs = decoder(images, targets, lengths)

            targets = pack_padded_sequence(targets, lengths, batch_first=True)[0]

            loss = criterion(outputs, targets)

            loss.backward()

            optimizer.step()

    # save model after training
    decoder_ckpt = torch.save(decoder, "decoder.ckpt")



# if we already trained, and EVAL == True, reload saved model
else:

    data_transform = transforms.Compose([ 
        transforms.Resize(224),     
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),   # using ImageNet norms
                             (0.229, 0.224, 0.225))])


    test_lines = read_lines(TOKEN_FILE_TEST)
    test_image_ids, test_cleaned_captions = parse_lines(test_lines)


    # load models
    encoder = EncoderCNN().to(device)
    decoder = torch.load("decoder.ckpt").to(device)
    encoder.eval()
    decoder.eval() # generate caption, eval mode to not influence batchnorm



#########################################################################
#
#        QUESTION 2.1 Generating predictions on test data
# 
#########################################################################


    # TODO define decode_caption() function in utils.py
    # predicted_caption = decode_caption(word_ids, vocab)
    
    dataset_test = Flickr8k_Images(
        image_ids=test_image_ids,
        transform=data_transform,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    
    sample_ids = []

    for data in test_loader:
        image = data
        image = image.to(device)
        image_feature = encoder(image)
        sample_id = decoder.sample(image_feature.squeeze(-1).squeeze(-1))
        sample_ids.append(sample_id)
        
    predicted_captions = decode_caption(None, sample_ids, vocab)


#########################################################################
#
#        QUESTION 2.2-3 Caption evaluation via text similarity 
# 
#########################################################################


    # Feel free to add helper functions to utils.py as needed,
    # documenting what they do in the code and in your report

    #reference_cap_idx = list(np.arange(0, 5016, 5))

    bleu_score, bleu_all_scores = Evaluation_bleu(test_cleaned_captions, predicted_captions)
    cos_score, cos_all_scores = COS_SIMILARITY(predicted_captions, test_cleaned_captions, vocab)
    
        
    


