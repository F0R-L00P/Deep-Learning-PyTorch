# assume the model is defined and architecture is provided
model = CNN().cuda()
# setup target optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# define checkpoint parameter
checkpoint = {'epoch': 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': 0.2}
# save model from checkpoint
torch.save(checkpoint, 'model.pth.tar')
# obtain checkpoint parameters and layer weights
checkpoint = torch.load('model.pth.tar')

# load model from chackpoint
model.load_state_dict(checkpoint['model_state_dict'])
# load optimizer from checkpoint
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# obtain last checkpoint epoch
epoch = checkpoint['epoch']
# obtain final loss measure
loss = checkpoint['loss']

# If testing lock model parameters and continue
model.eval()
# If resume training, simply run from checkpoint
model.train()