import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler

# loading data using pandas
directory = r'C:\Users\file_path'
df = pd.read_csv(directory + r'\diabetes.csv')
df.head()

df.info()
df.describe().transpose()

# get target label and convert to numpy arrays
X = df.iloc[:,:-1].values
y_string = df.iloc[:,-1]

# matrix label shape
print(f'X:{X.shape}',f'\ny:{y_string.shape}')

# convert y_string to binary values 0/1
y = y_string.apply(lambda x: 0 if (x == 'negative') else 1)
# convert to array
y = np.array(y, dtype='float64')

# NOTE:
# must normalize the data as bigger values will dominate lower values
# normalize features between -1 & 1
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
X_scaled

# convert np.array to pytorch
X_scaled = torch.tensor(X_scaled)
y = torch.tensor(y).unsqueeze(1)

print(f'X:{X_scaled.shape}',f'\ny:{y.shape}')

# lets build custome pytorch dataset
# inheriting from pytorch Dataset class
class Dataset(Dataset):

    def __init__(self, x, y):
        self.X = X
        self.y = y
        
    def __getitem__(self,index):
        # Get one item from the dataset
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.X)

# define dataset
data = Dataset(X_scaled, y)
len(data)

# call data loader to setup load/batch/shuffle
train_loader = DataLoader(data, batch_size=32, shuffle=True)

# Let's have a look at the data loader
print("There is {} batches in the dataset".format(len(train_loader)))
for (x,y) in train_loader:
    print("For one iteration (batch), there is:")
    print("Data:    {}".format(x.shape))
    print("Labels:  {}".format(y.shape))
    break

# build torch model with input and output features
# 7 -> 5 -> 4 -> 3 -> 1
class Model(nn.Module):
    def __init__(self, input_features, output_feature):
        super(Model, self).__init__()
        # every class needs attributs and functionalities
        # attribute -> number of layers
        # functionality -> forward prop
        self.layer1 = nn.Linear(in_features=input_features, out_features=5)
        self.layer2 = nn.Linear(in_features=5, out_features=4)
        self.layer3 = nn.Linear(in_features=4, out_features=3)
        self.layer4 = nn.Linear(in_features=3, out_features=output_feature)
