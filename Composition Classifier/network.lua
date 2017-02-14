require 'nn'
require 'cunn'
require 'cudnn'

net = torch.load('resnet-34.t7')
net:remove()
net:add(nn.Linear(512,512))
net:add(nn.ReLU(true))
net:add(nn.Linear(512,1))
net:add(nn.Sigmoid())
net:cuda()
return net