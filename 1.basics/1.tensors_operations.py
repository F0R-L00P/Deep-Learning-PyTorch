import torch
import numpy as np

# generate 1-dimensional tensor
t1 = torch.Tensor([2, 2, 1])
print(t1)

# generate 2-dimensional tensor (mtrix)
t2 = torch.Tensor([[2, 2, 1], [3, 3, 1], [4, 4, 1], [5, 5, 1]])
print(t2)

# size dim
print(t1.size(), t2.size())

# getting number of rows of the defined 2D tensor
print(t2.shape[0])

# creating float tensor
ft = torch.FloatTensor([[2, 2, 1], [3, 3, 1], [4, 4, 1], [5, 5, 1]])
print(ft)
# or 
ft = torch.tensor([2, 2, 1], dtype=torch.float)
print(ft)

# increasing floating points
dt = torch.DoubleTensor([[2, 2, 1], [3, 3, 1], [4, 4, 1], [5, 5, 1]])
print(dt)

# checking mean and standard deviation
print(ft.mean())
print(dt.std())

# reshaping
# can complete using view, when -1m shape can be inferred
# i.e. 4x3 if use (-1, 1) it will inffer as 12x1
t2.view(-1,1) #-> change to 1 column
t2.view(-1, 4) #-> change to 4 columns
print(t2.shape)

# check dimension
t2.dim()

# creating matrix with random numbers between 0-1
t_random = torch.rand(3, 3)

# creating matrix with random values drawn from a normal distribution
# with mean 0, and variance 1
tr_normal = torch.randn(3, 3)
print(tr_normal.dim())
print(tr_normal.size())
print(tr_normal.dtype)

# creating a mtrix of random integers 
# with values between 7 and exclusive of 10
mat = torch.randint(7, 10, (3, 3))
print(mat)

# getting number of elements in an array
print(torch.numel(mat))

# define a matrix of zeros with long dtype
lmat = torch.zeros(4, 4, dtype=torch.long)
print(lmat)
print(lmat.dtype)

# define a matrix of ones
omat = torch.ones(3, 3, dtype=torch.long)
print(omat)
print(omat.dtype)

# adding tensors
t_add = torch.add(t1, t2)
# or
t1.add(t2)

# NOTE: can convert between numpy and torch
num1 = np.ones(6)
num_tor = torch.from_numpy(num1)

#######################
# moving tensors to GPU
#######################
t1 = t1.cuda()

# SWITCHING between CPU & GPU
CUDA = torch.cuda.is_available()
print(CUDA)
if CUDA:
    my_result = my_result.cuda()
    print(result)


# tesor concatenation
t1 = torch.randn(2, 5)
print(t1)
t2 = torch.randn(3, 5)
print(t2)
# lets concat rows along zero (0) dimension
cat_tens1 = torch.cat([t1, t2])
print(cat_tens1)

# concat along the columns
t1 = torch.randn(2, 5)
t2 = torch.randn(2, 5)

cat_tens2 = torch.cat([t1, t2], 1)

##Adding dimensions
my_tensor = torch.tensor([1, 1, 2, 2, 3])
print(my_tensor)

# 1 row 5 columns
another_tensor = torch.unsqueeze(my_tensor, 0)

# make 3D tensor with 2 channels, 3 rows 4 columns
ten3 = torch.rand(2, 3, 4)