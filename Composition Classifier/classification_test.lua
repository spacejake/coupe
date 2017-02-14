require 'cunn'
require 'cudnn'
require 'nn'
require 'image'
require 'lfs'
require 'cutorch'
require 'optim'
-- cutorch.setDevice(1)
-- sofar = os.clock()

cmd = torch.CmdLine()
cmd:option('-net', '/media/eunbin/Data2/Dis_model/ggiruk/baseline-epoch-9.net', 'trained model')
cmd:option('-dir', '/home/eunbin/TorchNet/test/test_bad2', 'target directory')
opt = cmd:parse(arg or {})
num_bad = 0
num_good = 0
print(opt.net)
print(opt.dir)
model = torch.load(opt.net):cuda() 
model:evaluate()

-- require 'hdf5'
-- print('Loading test data...')
-- testFile = hdf5.open('discriminator_test.h5', 'r')
-- testData = testFile:all()
-- testFile:close()

-- dataset = testData
-- data = dataset.data
-- label = dataset.label
-- dataset.size = data:size()[1]
-- batchSize = 32
-- classes = {'o', 'x'}
-- confusion = optim.ConfusionMatrix(classes)

-- print('<trainer> on testing Set:')
-- for t = 1,dataset.size,batchSize do
-- 	xlua.progress(t, dataset.size)
-- 	-- prepare input batch
-- 	local inputs = data:sub(t,math.min(t+batchSize-1, dataset.size))
-- 	inputs = inputs:cuda()
-- 	local targets = label:sub(t,math.min(t+batchSize-1, dataset.size))+1
-- 	--local targets_tl = targets:narrow(4,1,1)
-- 	--local targets_br = targets:narrow(4,2,1)
-- 	--local targets_both = {targets_tl, targets_br}
-- 	targets = targets:cuda()

-- 	local outputs = model:forward(inputs)
-- 	for i = 1,batchSize do
-- 		confusion:add(outputs[i]:view(-1), targets[i]:view(-1)[1])
-- 		print(outputs[i]:view(-1), targets[i]:view(-1)[1])
-- 		--confusion_br:add(outputs[2][i]:view(-1), targets_both[2][i]:view(-1)[1])
-- 	end
-- 	print(outputs)
-- end
-- print(confusion)
-- --print(confusion_br)
-- --testLogger:add{['% mean class accuracy (test set, top left)'] = confusion_tl.totalValid * 100,
-- --			   ['% mean class accuracy (test set, bottom right)'] = confusion_br.totalValid * 100}
-- confusion:zero()



for file in lfs.dir(opt.dir) do
if lfs.attributes(opt.dir..'/'..file, "mode") == "file" then
img = image.load(opt.dir..'/'..file, 3, 'float')
--transposed_img = img:transpose(2,3)
--imggg = image.vflip(transposed_img)
channel = img:size()[1]
height = img:size()[2]
width = img:size()[3]
output_size = channel * height * width

input = image.scale(img, '224x224', 'bicubic')
batch = input:view(1, table.unpack(input:size():totable()))
--input = input:view(1, 3, 224, 224)
--print(input)
--input = input:cuda()



output = model:forward(batch:cuda())
if output[1][1] > 0.5 then 
	num_good = num_good + 1
	result = 1
else
	num_bad = num_bad + 1
	result = 0
end
--print(file..' : '.. result)



-- prob, idx = torch.max(output, 2)
-- if idx[1][1] == 1 then
-- 	num_good = num_good + 1
-- else 
-- 	num_bad = num_bad + 1
-- end
	
 --print(file..' : '.. idx[1][1])

print(file..' : '.. result)

--print(output)
--print(idx)

end
end

print('number of good : '.. num_good)
print('number of bad : '.. num_bad)



--image.save('result.png', transposed_img)
--image.save('resultgg.png', imggg)

print('done.')
