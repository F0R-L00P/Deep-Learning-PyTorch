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
        # Create a set of all punctuation characters to filter them out later
        punctuation = set(string.punctuation)        
        # Initialize an empty list to store the cleaned text
        cleaned_text = []
        # Open the document file for reading
        with open(document_path, 'r') as txt_doc:
            # Iterate through each line in the document
            for line in txt_doc:
                # Check if the line is not empty or contains only whitespaces
                if line.strip():
                    # Remove punctuation characters from the line
                    line = line.translate(str.maketrans("", "", string.punctuation))
                    # Convert the line to lowercase
                    line = line.lower()
                    # Split the line into words
                    words = line.strip().split()                    
                    # Remove stop words from the list of words
                    words = [word for word in words if word not in self.stop_words]                    
                    # Add the cleaned words to the cleaned_text list
                    cleaned_text += words
                    # Add the '<start>', words, and '<eos>' tokens to the dictionary
                    for word in ['<start>'] + words + ['<eos>']:
                        self.dictionary.add_word(word)
                else:
                    # If the line is empty, add a '<new_para>' token to the cleaned_text list
                    cleaned_text.append('<new_para>')                    
                    # Add the '<new_para>' token to the dictionary
                    self.dictionary.add_word('<new_para>')
        
        # Return the updated dictionary object
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
len(dictionary.word2index)