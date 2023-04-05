import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

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
batch_size = 20
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
vocab_size = len(corpus.dictionary.word2index)
print(vocab_size)

#########################################################################
#########################################################################
#########################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
def detach(states):
    """
If we have a tensor z,'z.detach()' returns a tensor that shares the same storage
as 'z', but with the computation history forgotten. It doesn't know anything
about how it was computed. In other words, we have broken the tensor z away from its past history
Here, we want to perform truncated Backpropagation
TBPTT splits the 1,000-long sequence into 50 sequences (say) each of length 20 and treats each sequence of length 20 as 
a separate training case. This is a sensible approach that can work well in practice, but it is blind to temporal 
dependencies that span more than 20 timesteps.
    """
    return [state.detach() for state in states] 

# setting variable parametes
# Sample variables
vocab_size = vocab_size # Assuming there are 100 unique words in the text
embed_size = 128        # Embedding size
hidden_size = 1024      # Hidden state size
num_layers = 1          # Number of LSTM layers
batch_size = 20         # Batch size
num_epochs = 10         # number of epochs
timesteps = 30          # Sequence length  
learning_rate=0.002


# initalize class
model = TextGenerator(vocab_size, embed_size, hidden_size, vocab_size, num_layers)
model = model.to(device)

# setup loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

patience = 5
num_epochs_without_improvement = 3
min_val_loss = float('inf')

for epoch in tqdm(range(num_epochs)):
    epoch_loss = 0

    for i in range(0, tensor_data.size(1) - timesteps, timesteps):
        # Get mini-batch inputs and targets
        inputs = tensor_data[:, i:i+timesteps].to(device)
        targets = tensor_data[:, i:i+timesteps].to(device)

        # Set initial hidden and cell states
        input_batch_size = inputs.size(0)
        states = (torch.zeros(num_layers, input_batch_size, hidden_size).to(device),
                  torch.zeros(num_layers, input_batch_size, hidden_size).to(device))

        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs, states = model(inputs, states)
        # Detach states to prevent backpropagation through entire sequence history
        states = detach(states)
        # Calculate the loss
        loss = criterion(outputs, targets.reshape(-1))
        epoch_loss += loss.item()
        # Backward pass
        loss.backward()
        # Clip gradients to prevent exploding gradients
        clip_grad_norm_(model.parameters(), 0.5)
        # Update the weights
        optimizer.step()

        step = (i + 1) // timesteps
        if step % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{step}], Loss: {loss.item()}")

    # Calculate the average loss for the current epoch
    avg_epoch_loss = epoch_loss / (i + 1)

    # Check if the validation loss has improved
    if avg_epoch_loss < min_val_loss:
        min_val_loss = avg_epoch_loss
        num_epochs_without_improvement = 0
    else:
        num_epochs_without_improvement += 1

    # Check if early stopping should be applied
    if num_epochs_without_improvement >= patience:
        print("Early stopping applied.")
        break


# Test the model
with torch.no_grad():
    with open('results.txt', 'w') as f:
        # Set initial hidden and cell states
        state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                 torch.zeros(num_layers, 1, hidden_size).to(device))
        # Select one word id randomly and convert it to shape (1,1)
        inputs = torch.randint(0, vocab_size, (1,)).long().unsqueeze(1).to(device)

        for i in range(500):
            output, _ = model(inputs, state)
            print(output.shape)
            # Sample a word id from the exponential of the output 
            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()
            print(word_id)
            # Replace the input with sampled word id for the next time step
            inputs.fill_(word_id)

            # Write the results to file
            word = corpus.dictionary.index2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)
            
            if (i+1) % 100 == 0:
                print('Sampled [{}/{}] words and save to {}'.format(i+1, 500, 'results.txt'))
