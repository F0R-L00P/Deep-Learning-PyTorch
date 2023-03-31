import torch
import torch.nn as nn
from torch.nn.utils import clip_grad

import os
import string
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
############################################################
############################################################
############################################################
# define directory and data path
path = os.getcwd()
data_path = r'\dataset\down_the_rabbit_hole.txt'
full_path = path+data_path
############################################################
############################################################
# define class to capture
# word and its crresponding index
# ability to use index to find word
class Dictionary:
    # start with class variables
    def __init__(self):
        # the index will be incremented with 
        # every word thats added to the dict
        self.index = 0
        self.word2index = {}
        self.index2word = {}
        
    # adding a method to class
    def add_word(self, word):
        # check if word is in dict
        # if not add it
        if word not in self.word2index:
            # This line assigns the value of indx
            # starting at zero
            self.word2index[word] = self.index 
            # assign a lookup for the index to be the word
            self.index2word[self.index] = word
            # increment the index for the next word
            self.index += 1
            
    def __len__(self):
        # get number of vocab
        return len(self.word2index)


class TextPreprocessing:
    def __init__(self):
        self.dictionary = Dictionary()
        self.stop_words = set(stopwords.words('english'))

    def get_clean_text(self, document_path, batch_size=20):
        punctuation = set(string.punctuation)
        cleaned_text = []
        with open(document_path, 'r') as txt_doc:
            for line in txt_doc:
                if line.strip():
                    line = line.translate(str.maketrans("", "", string.punctuation))
                    line = line.lower()
                    words = line.strip().split()
                    words = [word for word in words if word not in self.stop_words]
                    cleaned_text += words

                    for word in ['<start>'] + words + ['<eos>']:
                        self.dictionary.add_word(word)
                else:
                    cleaned_text.append('<new_para>')
                    self.dictionary.add_word('<new_para>')
        return self.dictionary

    def generate_tensor(self, path):
        index_list = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    # Preprocess the text: remove punctuation and convert to lowercase
                    line = line.translate(str.maketrans("", "", string.punctuation))
                    line = line.lower()
                    words = line.strip().split()

                    # Remove stop words
                    words = [word for word in words if word not in self.stop_words]

                    # Iterate over the preprocessed words
                    for word in words:
                        # Add the integer index of the word to the list, if it's in the dictionary
                        if word in self.dictionary.word2index:
                            index_list.append(self.dictionary.word2index[word])

                # Add the '<eos>' token at the end of each line
                index_list.append(self.dictionary.word2index['<eos>'])

        # Create a 1-D tensor from the list of integer indices
        rep_tensor = torch.LongTensor(index_list)
        return rep_tensor

    def get_data(self, path, batch_size=20):
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words: 
                    self.dictionary.add_word(word)  
            
        # call the generate_tensor method to create the 1-D tensor
        rep_tensor = self.generate_tensor(path)

        #Find out how many batches we need            
        num_batches = rep_tensor.shape[0] // batch_size     
        #Remove the remainder (Filter out the ones that don't fit)
        rep_tensor = rep_tensor[:num_batches*batch_size]
        # return (batch_size,num_batches)
        rep_tensor = rep_tensor.view(batch_size, -1)
        return rep_tensor

#########################################################################    
#########################################################################
#########################################################################
# Instantiate TextPreprocessing class
text_preprocessor = TextPreprocessing()
# Preprocess the text file and build the dictionary
dictionary = text_preprocessor.get_clean_text(full_path)
# Generate tensor representation of the text
tensor_representation = text_preprocessor.generate_tensor(full_path)

print("Dictionary:")
print(dictionary.word2index)

print("\nTensor representation:")
print(tensor_representation)
#########################################################################
#########################################################################
#########################################################################










import collections
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer

# Define special tokens
PAD_TOKEN = '<PAD>'
START_TOKEN = '<START>'
END_TOKEN = '<END>'
UNK_TOKEN = '<UNK>'
COMMA_TOKEN = '<COMMA>'
PERIOD_TOKEN = '<PERIOD>'
EXCLAMATION_TOKEN = '<EXCLAMATION>'
AT_TOKEN = '<AT>'
HASH_TOKEN = '<HASH>'
DOLLAR_TOKEN = '<DOLLAR>'
PERCENT_TOKEN = '<PERCENT>'
CARET_TOKEN = '<CARET>'
AMPERSAND_TOKEN = '<AMPERSAND>'

# Define tokenizer
tokenizer = get_tokenizer('basic_english')

# Define dictionary mapping special characters to special tokens
special_tokens = {
    ',': COMMA_TOKEN,
    '.': PERIOD_TOKEN,
    '!': EXCLAMATION_TOKEN,
    '@': AT_TOKEN,
    '#': HASH_TOKEN,
    '$': DOLLAR_TOKEN,
    '%': PERCENT_TOKEN,
    '^': CARET_TOKEN,
    '&': AMPERSAND_TOKEN
}

# Define preprocessing function to add special tokens
def add_special_tokens(text):
    tokens = tokenizer(text)
    for i in range(len(tokens)):
        if tokens[i] in special_tokens:
            tokens[i] = special_tokens[tokens[i]]
    return [START_TOKEN] + tokens + [END_TOKEN]

# Define vocabulary
tokens = []
with open(full_path) as f:
    for line in f:
        tokens += add_special_tokens(line.strip())

tokens