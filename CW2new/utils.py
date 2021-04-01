import torch
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from nltk.translate.bleu_score import sentence_bleu

from vocabulary import Vocabulary
from config import *
import string


def read_lines(filepath):
    """ Open the ground truth captions into memory, line by line. 
    Args:
        filepath (str): the complete path to the tokens txt file
    """
    file = open(filepath, 'r')
    lines = []

    while True: 
        # Get next line from file 
        line = file.readline() 
        if not line: 
            break
        lines.append(line.strip())
    file.close()
    return lines


def parse_lines(lines):
    """
    Parses token file captions into image_ids and captions.
    Args:
        lines (str list): str lines from token file
    Return:
        image_ids (int list): list of image ids, with duplicates
        cleaned_captions (list of lists of str): lists of words
    """
    image_ids = []
    cleaned_captions = []

    for line in lines:
        ids, text = line.split('\t')
        ids = ids[:-6]
        text = text.strip(string.punctuation).lower()
        text = text.strip(' ')
        image_ids.append(ids)
        cleaned_captions.append(text)

    # QUESTION 1.1


    return image_ids, cleaned_captions


def build_vocab(cleaned_captions):
    """ 
    Parses training set token file captions and builds a Vocabulary object
    Args:
        cleaned_captions (str list): cleaned list of human captions to build vocab with

    Returns:
        vocab (Vocabulary): Vocabulary object
    """

    # QUESTION 1.1
    # TODO collect words
    words = dict()
    for sentence in cleaned_captions:
        for word in sentence.split():
            words[word] = words.get(word, 0) + 1
    
    words_1 = dict(filter(lambda item: item[1]>3, words.items()))


    # create a vocab instance
    vocab = Vocabulary()

    # add the token words
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # TODO add the rest of the words from the cleaned captions here
    # vocab.add_word('word')
    for word in words_1:
        vocab.add_word(word)

    return vocab



def decode_caption(ref_captions, sampled_ids, vocab):
    """ 
    Args:
        ref_captions (str list): ground truth capti ons
        sampled_ids (int list): list of word IDs from decoder
        vocab (Vocabulary): vocab for conversion
    Return:
        predicted_caption (str): predicted string sentence
    """
    '''
    predicted_caption = " "

    list_sample_id = sampled_ids[0].cpu().numpy()

    sentence = []

    for idx in list_sample_id:
        if idx == 2:
            break
        word = vocab.idx2word[idx]
        sentence.append(word.strip('<>'))
    '''

    # QUESTION 2.1
    res = []
    for sentence in sampled_ids:
        sentences = []
        for idx in sentence:
            idx = list(idx.cpu().numpy())
            for id in idx:
                if id == 2:
                    break
                word = vocab.idx2word[id]
                sentences.append(word.strip('<>'))
        res.append(" ".join(sentences[1:]))

    predicted_caption = res[::5]


    return predicted_caption



    #return predicted_caption.join(sentence[1:])


"""
We need to overwrite the default PyTorch collate_fn() because our 
ground truth captions are sequential data of varying lengths. The default
collate_fn() does not support merging the captions with padding.

You can read more about it here:
https://pytorch.org/docs/stable/data.html#dataloader-collate-fn. 
"""
def caption_collate_fn(data):
    """ Creates mini-batch tensors from the list of tuples (image, caption).
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 224, 224).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 224, 224).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length from longest to shortest.
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # merge images (from tuple of 3D tensor to 4D tensor).
    # if using features, 2D tensor to 3D tensor. (batch_size, 256)
    images = torch.stack(images, 0) 

    # merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def Evaluation_bleu(ref_captions, predicted_captions):
    scores = []
    for i in range(1, len(predicted_captions)):
        score = sentence_bleu(ref_captions[5*(i-1): 5*i],
                              predicted_captions[i-1])
        scores.append(score)
    
    average_score = sum(scores) / len(scores)

    return average_score, scores