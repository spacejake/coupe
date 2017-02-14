require 'nn'
require 'cunn'
--require 'cudnn'
-- load torchnet:
local tnt = require 'torchnet'

-- use GPU or not:
local cmd = torch.CmdLine()
cmd:option('-usegpu', false, 'use gpu for training')
local config = cmd:parse(arg)
print(string.format('running on %s', config.usegpu and 'GPU' or 'CPU'))
--- Returns transform function used for training

local getIterator = dofile('myIterator.lua')
--local getIterator = dofile('batchIterator.lua')

-- set up logistic regressor:
local net = dofile('network.lua')

-- local criterion = nn.CrossEntropyCriterion()
local criterion = nn.BCECriterion()

-- set up training engine:
local engine = tnt.SGDEngine()
local meter  = tnt.AverageValueMeter()
-- local clerr  = tnt.ClassErrorMeter{topk = {1}}
local logtext = require 'torchnet.log.view.text'
local logstatus = require 'torchnet.log.view.status'
log = tnt.Log{
   keys = {"loss", "accuracy"},
   onFlush = {
      -- write out all keys in "log" file
      logtext{filename='log.txt', keys={"loss", "accuracy"}, format={"%10.5f", "%3.2f"}},
      -- write out loss in a standalone file
      logtext{filename='loss.txt', keys={"loss"}},
      -- print on screen too
      logtext{keys={"loss", "accuracy"}},
   },
   onSet = {
      -- add status to log
      logstatus{filename='log.txt'},
      -- print status to screen
      logstatus{},
   }
}


theta = 1.0
engine.hooks.onBackward = function(state)
   param, gradParam = state.network:getParameters()
   gradParam:clamp(-theta/state.lr, theta/state.lr)
end

engine.hooks.onStart = function(state)
if state.training then
   log:status("Train Start")
else
   log:status("Test Start")
end
end

engine.hooks.onEnd = function(state)
if state.training then
   log:status("Train End")
else
   log:status("Test End")
end
end

engine.hooks.onStartEpoch = function(state)
print(string.format('Current learning rate: %f', state.lr))
meter:reset()
   --clerr:reset()
end

engine.hooks.onForwardCriterion = function(state)
print(state.criterion.output)
meter:add(state.criterion.output)
-- clerr:add(state.network.output, state.sample.target)
   if state.training then
      -- print(string.format('avg. loss: %2.4f; avg. accuracy: %2.4f', meter:value(), 100-clerr:value{k = 1}))
      print(string.format('avg. loss: %2.4f;', meter:value()))
      log:set{
        loss = meter:value(),
      }
   end
end

lastLoss = 99999
engine.hooks.onEndEpoch = function(state)
if lastLoss <= meter:value() then
   print('Learning rate decays.')
   state.lr = state.lr * 0.1
end
lastLoss = meter:value()
local filename = '/media/eunbin/Data2/Dis_model/ROT/baseline-epoch-' .. tostring(state.epoch) .. '.net'
os.execute('mkdir -p ' .. sys.dirname(filename))
print('<engine> saving network to '..filename)
   --net_save = state.network:clone()
   --torch.save(filename, net_save:clearState())
   torch.save(filename, state.network:clearState())
   log:flush()
end

-- set up GPU training:
if config.usegpu then

   -- copy model to GPU:
   require 'cunn'
   require 'cudnn'
   --net       = net:cuda()
   criterion = criterion:cuda()
   --cudnn.benchmark = false
   --cudnn.fastest = true
   --cudnn.convert(net, cudnn)

   -- copy sample to GPU buffer:
   local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
   engine.hooks.onSample = function(state)
   igpu:resize(state.sample.input:size() ):copy(state.sample.input)
   tgpu:resize(state.sample.target:size()):copy(state.sample.target)
   state.sample.input  = igpu
   state.sample.target = tgpu
   end  -- alternatively, this logic can be implemented via a TransformDataset
end

-- train the model:
engine:train{
   network   = net,
   iterator  = getIterator('train'),
   criterion = criterion,
   lr        = 0.01,
   maxepoch  = 20,
}

-- measure test loss and error:
meter:reset()
--clerr:reset()
engine:test{
   network   = net,
   iterator  = getIterator('test'),
   criterion = criterion,
}

print(string.format('test loss: %2.4f; test error: %2.4f', meter:value(), clerr:value{k = 1}))