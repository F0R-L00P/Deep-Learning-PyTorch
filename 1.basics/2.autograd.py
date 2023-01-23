import torch

# to keep track of how tensor object is created
# set parameter --> requires_grad = True
X = torch.tensor([2., 2., 1.], requires_grad=True)
y = torch.tensor([3., 3., 1.], requires_grad=True)

res = X + y
print(res)

# res object tracks it was created from X and y tensors
print(res.grad_fn)

# can assign object with calculations
res_total = res.sum()
print(res_total)
print(res_total.grad_fn)

# lets do back-prop on res_total
# finding gradient of res_total with respect to X
res_total.backward()
print(X.grad)

# tensors have by default --> require_grad = False
X = torch.tensor([2., 2., 1.])
y = torch.tensor([3., 3., 1.])

print(X.requires_grad)
print(y.requires_grad)

# So you can't backprop through res
res = X + y
print(res.grad_fn)

# Another way to set the requires_grad = True is
X.requires_grad_()
y.requires_grad_()

z = X + y
print(z.grad_fn)

# z.detach() returns a tensor that shares the same storage as ``z``, 
# but with the computation history forgotten. 
# It doesn't know anything about how it was computed.
# In other words, we have broken the Tensor away from its past history
z_forgotten = z.detach()
print(z_forgotten.grad_fn)

# You can also stop autograd from tracking history on Tensors. 
# This concept is useful when applying Transfer Learning 
print(X.requires_grad)
print((X+10).requires_grad)

with torch.no_grad():
    print((X+10).requires_grad)
