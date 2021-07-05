"""
COMP5623M Coursework on Image Caption Generation


python decoder.py


"""

from math import cos
import torch
import numpy as np

import torch.nn as nn
from torch.nn.modules.loss import TripletMarginWithDistanceLoss
from torch.utils import data
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image


from datasets import Flickr8k_Images, Flickr8k_Features
from models import DecoderRNN, EncoderCNN
from utils import *
from config import *

# if false, train model; otherwise try loading model from checkpoint and evaluate
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
    print("test begin")

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

    # Generate sample ids
    for data in test_loader:
        image = data
        image = image.to(device)
        image_feature = encoder(image)
        sample_id = decoder.sample(image_feature.squeeze(-1).squeeze(-1))
        sample_ids.append(sample_id)

    
    predicted_captions = decode_caption(None, sample_ids, vocab)
    predicted_captions_sampled = predicted_captions[::5]

#########################################################################
#
#        QUESTION 2.2-3 Caption evaluation via text similarity 
# 
#########################################################################


    # Feel free to add helper functions to utils.py as needed,
    # documenting what they do in the code and in your report


    # Compute BLEU score
    avg_bleu_score, all_bleu_scores = Evaluation_bleu(test_cleaned_captions, predicted_captions_sampled)
    
    # Compute cosinie similarity score
    def Cos_similarity(reference, predict, vocab, decoder):
        # Generate word id 
        ref = [vocab(word) for word in reference.split(" ")]
        pred = [vocab(word) for word in predict.split(" ")]

        # Embedding refernce caption and predict caption
        embedding_ref = decoder.embed(torch.tensor(ref).to(device).long()).cpu().detach().clone().numpy()
        embedding_pred = decoder.embed(torch.tensor(pred).to(device).long()).cpu().detach().clone().numpy()

        # Compute average vector 
        ref_vector = np.sum(embedding_ref, axis = 0).reshape(1,-1)/embedding_ref.shape[0]
        pred_vector = np.sum(embedding_pred, axis = 0).reshape(1,-1)/embedding_pred.shape[0]
    
        cos_score = cosine_similarity(ref_vector, pred_vector)[0][0]
        return cos_score

    cos_scores = []
    for i in range(len(predicted_captions)):
        reference = test_cleaned_captions[i]
        predict_cap = predicted_captions[i]
        score = Cos_similarity(reference, predict_cap ,vocab, decoder)
        cos_scores.append(score)
    average_cos_score = sum(cos_scores) / len(cos_scores) # Average cosine score
    

    # Rescale cosine similarity score
    cos_array = np.array(cos_scores)
    rescaled_cos_array = cos_array - np.min(cos_array) / ( np.max(cos_array) - np.min(cos_array) )
    rescaled_average_score = np.mean(rescaled_cos_array)

    
    
    
