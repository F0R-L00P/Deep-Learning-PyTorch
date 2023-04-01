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
            
class TextPreprocessing:
    def __init__(self):
        # Initialize the dictionary object
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size=20):
        # Read the text file and count the total number of tokens (words)
        # while adding words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)        
        # Create a 1-D tensor to store the indices of the words in the file
        rep_tensor = torch.LongTensor(tokens)
        index = 0        
        # Read the text file again and populate the tensor with word indices
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    rep_tensor[index] = self.dictionary.word2index[word]
                    index += 1        
        # Calculate the number of batches based on the batch size
        num_batches = rep_tensor.shape[0] // batch_size  
        # Truncate the tensor to remove any extra tokens that don't fit in the last batch
        rep_tensor = rep_tensor[:num_batches * batch_size]        
        # Reshape the tensor to have dimensions (batch_size, num_batches)
        rep_tensor = rep_tensor.view(batch_size, -1)        
        # Return the reshaped tensor
        return rep_tensor


#########################################################################    
#########################################################################
#########################################################################
# Create a TextPreprocessing object
text_preprocessor = TextPreprocessing()

# Use the get_data method to read the sample text file and create a tensor
batch_size = 1
tensor_data = text_preprocessor.get_data(full_path, batch_size)

# Print the generated tensor
print(tensor_data)

# print vocab dictionary
print("Dictionary:")
print("Vocabulary size:", len(text_preprocessor.dictionary.word2index))
print(text_preprocessor.dictionary.word2index)

print("\nTensor representation:")
print(tensor_data.shape)

#########################################################################
#########################################################################
#########################################################################
# initalize class
corpus = TextPreprocessing()

# get tensor representation of the .txt file
# to find the batch size, to process the sequence
# assuming the sequence is 1000, and time step is 50
# the batching of the sentence will be 1000 // 50 = 20
# therefore the sequence will be processed 20 characters at a time
tensor_data = corpus.get_data(full_path, batch_size)

# tensor_data is the tensor that contains the index of all the words. 
# Each row contains 662 words by at batch of 20
print(tensor_data.shape)

# obtain vocab size 
vocab_size = len(corpus.dictionary)
print(vocab_size)

#########################################################################
#########################################################################
#########################################################################
class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1):
        super(TextGenerator, self).__init__()
        # Define the embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Define the LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        # Define the linear layer
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        # Pass the input through the embedding layer
        x = self.embedding(x)
        # Pass the embedded input through the LSTM layer
        x, hidden = self.lstm(x, hidden)
        # Reshape the output to be fed into the linear layer
        x = x.contiguous().view(-1, x.shape[2])
        # Pass the LSTM output through the linear layer
        x = self.linear(x)
        return x, hidden

# setting parameters
embed_size = 128    #Input features to the LSTM
hidden_size = 1024  #Number of LSTM units
num_layers = 1
num_epochs = 20
batch_size = 20
timesteps = 30 # look at 30 previouse words to predict the next word
learning_rate = 0.002

# initalize class
lstm_model = TextGenerator()
