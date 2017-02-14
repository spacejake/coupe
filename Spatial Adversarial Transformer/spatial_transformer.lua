require 'xlua'
require 'optim'
require 'cunn'
require 'nn'
require 'cudnn'
require 'image'
require 'stn'

spanet = nn.Sequential()
 
concat = nn.ConcatTable()

-- first branch is there to transpose inputs to BHWD, for the bilinear sampler
tranet=nn.Sequential()
tranet:add(nn.Identity())
-- tranet:add(nn.Transpose({2,3},{3,4}))
tranet:add(nn.Transpose({2,4}))

-- second branch is the localization network


-- locnet = nn.Sequential()

-- locnet:add(nn.SpatialMaxPooling(2,2,2,2))
-- locnet:add(nn.SpatialConvolution(3,96,7,7))
-- locnet:add(nn.ReLU(true))

-- locnet:add(nn.SpatialMaxPooling(3,3,2,2))
-- locnet:add(nn.SpatialConvolution(96,256,5,5))
-- locnet:add(nn.ReLU(true))

-- locnet:add(nn.SpatialMaxPooling(3,3,2,2))
-- locnet:add(nn.SpatialConvolution(256,512,3,3))
-- locnet:add(nn.ReLU(true))

-- locnet:add(nn.SpatialMaxPooling(3,3,2,2))
-- locnet:add(nn.SpatialConvolution(512,512,3,3))
-- locnet:add(nn.ReLU(true))

-- locnet:add(nn.View(512*8*8))
-- locnet:add(nn.Linear(512*8*8,4096))
-- locnet:add(nn.ReLU(true))
-- locnet:add(nn.Dropout(0.5))

-- prototxt = './VGG-S_Deploy.prototxt'
-- binary = 'VGG_CNN_S.caffemodel'
-- locnet = loadcaffe.load(prototxt, binary)
-- locnet:remove()
-- locnet:remove()

-- prototxt = './VGG-Deploy.prototxt'
-- binary = 'VGG_CNN_M.caffemodel'
-- locnet = loadcaffe.load(prototxt, binary)
-- locnet:remove()
-- locnet:remove()
-- locnet:remove()
-- locnet:add(nn.Dropout(0.7))

resnet = torch.load('/media/eunbin/Data2/Loc_labeled_model/lamda(0.05)/d-epoch-6.net')
resnet:remove()
resnet:add(nn.Linear(512,512)) -- add
resnet:add(nn.LeakyReLU()) -- add
resnet:add(nn.Dropout(0.5)) -- add


--locnet:add(nn.Dropout(0.5))

-- we initialize the output layer so it gives the identity transform

outLayer = nn.Linear(512,3)
outLayer.weight:fill(0)
bias = torch.FloatTensor(3):fill(0)
bias[1]=1
--bias[5]=1
outLayer.bias:copy(bias)
resnet:add(outLayer)

locnet = nn.Sequential()
locnet:add(resnet)

-- locnet:add(nn.SplitTable(2))

-- penaltyNet = nn.ParallelTable()
-- -- penaltyNet:add(nn.Sequential():add(nn.Abs()):add(nn.Sqrt()):add(nn.MulConstant(-0.1)):add(nn.Exp()):add(nn.Unsqueeze(2, 1)))
-- penaltyNet:add(nn.Sequential():add(nn.MulConstant(-1.2/2)):add(nn.Tanh()):add(nn.Abs()):add(nn.Unsqueeze(2, 1)))
-- penaltyNet:add(nn.Sequential():add(nn.MulConstant(-1.2)):add(nn.Tanh()):add(nn.Unsqueeze(2, 1)))
-- penaltyNet:add(nn.Sequential():add(nn.MulConstant(-1.2)):add(nn.Tanh()):add(nn.Unsqueeze(2, 1)))
-- -- penaltyNet:add(nn.Sequential():add(nn.Abs()):add(nn.Sqrt()):add(nn.MulConstant(-0.01)):add(nn.Exp()):add(nn.MulConstant(2)):add(nn.AddConstant(-1)):add(nn.Unsqueeze(2, 1)))
-- -- penaltyNet:add(nn.Sequential():add(nn.Abs()):add(nn.Sqrt()):add(nn.MulConstant(-0.01)):add(nn.Exp()):add(nn.MulConstant(2)):add(nn.AddConstant(-1)):add(nn.Unsqueeze(2, 1)))
-- locnet:add(penaltyNet)
-- locnet:add(nn.JoinTable(2))

-- TxNet_mother = nn.Sequential()
-- TxNet = nn.ConcatTable()
-- TxNet:add(nn.Sequential())
-- TxNet:get(1):add(nn.SelectTable(1)):add(nn.MulConstant(-1)):add(nn.AddConstant(1)) -- 1-s
-- TxNet:add(nn.SelectTable(2))
-- TxNet_mother:add(TxNet)
-- TxNet_mother:add(nn.CMulTable())
-- TxNet_mother:add(nn.Unsqueeze(2, 1))
-- TyNet_mother = nn.Sequential()
-- TyNet = nn.ConcatTable()
-- TyNet:add(nn.Sequential())
-- TyNet:get(1):add(nn.SelectTable(1)):add(nn.MulConstant(-1)):add(nn.AddConstant(1)) -- 1-s
-- TyNet:add(nn.SelectTable(3))
-- TyNet_mother:add(TyNet)
-- TyNet_mother:add(nn.CMulTable())
-- TyNet_mother:add(nn.Unsqueeze(2, 1))
-- normalizeNet = nn.ConcatTable()
-- normalizeNet:add(nn:Sequential():add(nn.SelectTable(1)):add(nn.Unsqueeze(2, 1))) -- select scale
-- normalizeNet:add(TxNet_mother) -- (1-s)*tx
-- normalizeNet:add(TyNet_mother) -- (1-s)*ty
-- locnet:add(nn.SplitTable(2))
-- locnet:add(normalizeNet)
-- locnet:add(nn.JoinTable(2))

locnet:add(nn.AffineTransformMatrixGenerator(false, true, true))

-- there we generate the grids
locnet:add(nn.AffineGridGeneratorBHWD(224,224))


-- we need a table input for the bilinear sampler, so we use concattable
concat:add(tranet)
concat:add(locnet)

spanet:add(concat)
spanet:add(nn.BilinearSamplerBHWD())

-- and we transpose back to standard BDHW format for subsequent processing by nn modules
-- spanet:add(nn.Transpose({3,4},{2,3}))
spanet:add(nn.Transpose({2,4}))
