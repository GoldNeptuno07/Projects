#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 20:41:11 2024

@author: angelcruz

Scrip to implement functions get and preprocess text data.
"""

import requests
import io
import os
import collections
import tarfile
from nltk.corpus import stopwords
import re
import numpy as np

# =============================================================================
# Function to load data.
# =============================================================================
def load_movies_data():
    # Define paths
    save_folder_name = 'movies_data'
    pos_file = os.path.join(save_folder_name, 'rt-polarity.pos')
    neg_file = os.path.join(save_folder_name, 'rt-polarity.neg')
    
    if os.path.exists(pos_file) and os.path.exists(neg_file):
        ## Get the data from path
        pos_data = []
        with open(pos_file, 'r') as pos_file_handler:
            for row in pos_file_handler:
                pos_data.append(row)
        neg_data = []
        with open(neg_file, 'r') as neg_file_handler:
            for row in neg_file_handler:
                neg_data.append(row)
    else:
        # Download data from url
        url = "https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz"
        req = requests.get(url)
        # Performe Request
        if req.ok:
            stream_data = io.BytesIO(req.content)
            tmp = io.BytesIO() 
            while True:
                s = stream_data.read(16384)
                if not s:
                    break
                tmp.write(s)
            stream_data.close()
            tmp.seek(0)
        else:
            raise ConnectionError(f"Something went wrong. Code: {req.code}")
        # Extract tar File
        tar_file = tarfile.open(fileobj= tmp, mode= "r:gz")
        pos = tar_file.extractfile('rt-polaritydata/rt-polarity.pos')
        neg = tar_file.extractfile('rt-polaritydata/rt-polarity.neg')
        # Get positive reviews
        pos_data = []
        for line in pos:
            pos_data.append(line.decode("ISO-8859-1").encode('ascii', errors= 'ignore').decode())
        # Get negative reviews
        neg_data = []
        for line in neg:
            neg_data.append(line.decode('ISO-8859-1').encode('ascii', errors= 'ignore').decode())
        tar_file.close()
        # Save data
        os.makedirs(save_folder_name, exist_ok= True)
        with open(pos_file, 'w') as pos_file_handler:
            pos_file_handler.write(''.join(pos_data))
        with open(neg_file, 'w') as neg_file_handler:
            neg_file_handler.write(''.join(neg_data))
    texts = pos_data + neg_data
    target = [1]*len(pos_data) + [0]*len(neg_data)
    return (texts, target)


# =============================================================================
#  Function to clean the text
# =============================================================================
def normalize_text(texts, stop):
    texts = [x.lower() for x in texts] # To lower case
    texts = [re.findall(pattern= "[a-z]+", string= x) for x in texts] # Remove Numbers and Punctuation marks
    texts = [' '.join([x for x in row if x not in stop]) for row in texts ]
    return texts


# =============================================================================
#  Function to return the dictionary of words
# =============================================================================
def build_dictionary(sentences, vocabulary_size):
    words = []
    for sentence in sentences:
        words.extend(sentence.split())
    count= [['RARE', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    word_dict = {}
    for word, word_count in count:
        word_dict[word] = len(word_dict)
    return word_dict

# =============================================================================
# Function to turn text to numbers
# =============================================================================
def text_to_numbers(sentences, word_dict):
    data = []
    for sentence in sentences:
        sentence_data = []
        for word in sentence.split():
            if word in word_dict:
                word_idx = word_dict[word]
            else:
                word_idx = 0
            sentence_data.append(word_idx)
        data.append(sentence_data)
    return data


