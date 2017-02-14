------------------------------------------------------------
--- This code is based on the eyescream code released at
--- https://github.com/facebook/eyescream
--- If you find it usefull consider citing
--- http://arxiv.org/abs/1506.05751
------------------------------------------------------------

require 'hdf5'
require 'nngraph'
require 'cudnn'
require 'torch'
require 'nn'
require 'cunn'
require 'optim'
require 'image'
require 'pl'
require 'paths'
require 'lfs'
require 'ROTCriterion.lua'
require 'ROTCriterion_boundary.lua'
require 'ROTSizeCriterion_boundary'

ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
adversarial = require 'Adversarial'


----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
-s,--save          (default "logs_256_no_avg")      subdirectory to save logs
--saveFreq         (default 1)          save every saveFreq epochs
-n,--network       (default "")          reload pretrained network
-p,--plot                                plot while training
-r,--learningRate  (default 0.001)        learning rate
--learningRate_save(default 0.001)        learning rate for recovery
-b,--batchSize     (default 52)         batch size
-m,--momentum      (default 0)           momentum, for SGD only
--momentum_save    (default 0)           momentum, for SGD only for recovery
--coefL1           (default 0)           L1 penalty on the weights
--coefL2           (default 0)           L2 penalty on the weights
-t,--threads       (default 4)           number of threads
-g,--gpu           (default 0)           gpu to run on (default cpu)
-d,--noiseDim      (default 512)         dimensionality of noise vector
--K                (default 1)           number of iterations to optimize D for
-w, --window       (default 3)           windsow id of sample image
--scale            (default 256)          scale of images to train on
--epoch            (default 0)           epock offset
]]


if opt.gpu < 0 or opt.gpu > 3 then opt.gpu = false end

opt.learningRate = 0.00001
--opt.epoch = 40
--opt.momentum = 0.2
opt.batchSize = 32
opt.scale = 224
opt.geometry = {3, opt.scale, opt.scale}

print(opt)

init_learningRate = opt.learningRate

-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

if opt.gpu then
    cutorch.setDevice(opt.gpu + 1)
    print('<gpu> using device ' .. opt.gpu)
    cudnn.benchmark = true
    cudnn.fastest = true
    print('fastest = true')
    torch.setdefaulttensortype('torch.CudaTensor')
else
    torch.setdefaulttensortype('torch.FloatTensor')
end

local input_sz = opt.geometry[1] * opt.geometry[2] * opt.geometry[3]

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

if opt.network == '' then
    ----------------------------------------------------------------------
    -- define D network to train
    -- model_D = nn.Sequential()
    -- model_D:add(cudnn.SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2))
    -- model_D:add(cudnn.SpatialMaxPooling(2,2)) -- 128 X 128
    -- model_D:add(cudnn.ReLU(true))
    -- model_D:add(nn.SpatialDropout(0.2))
    -- model_D:add(cudnn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2))
    -- model_D:add(cudnn.SpatialMaxPooling(2,2)) -- 64 X 64
    -- model_D:add(cudnn.ReLU(true))
    -- model_D:add(nn.SpatialDropout(0.2))
    -- model_D:add(cudnn.SpatialConvolution(64, 96, 5, 5, 1, 1, 2, 2))
    -- model_D:add(cudnn.ReLU(true))
    -- model_D:add(cudnn.SpatialMaxPooling(2,2)) -- 32 X 32
    -- model_D:add(nn.SpatialDropout(0.2))
    -- model_D:add(cudnn.SpatialConvolution(96, 96, 5, 5, 1, 1, 2, 2))
    -- model_D:add(cudnn.ReLU(true))
    -- model_D:add(cudnn.SpatialMaxPooling(2,2)) -- 16 X 16
    -- model_D:add(nn.SpatialDropout(0.2))
    -- model_D:add(cudnn.SpatialConvolution(96, 96, 5, 5, 1, 1, 2, 2))
    -- model_D:add(cudnn.ReLU(true))
    -- model_D:add(cudnn.SpatialMaxPooling(2,2)) -- 8 X 8
    -- model_D:add(nn.SpatialDropout(0.2))
    -- model_D:add(nn.Reshape(8*8*96))
    -- model_D:add(nn.Linear(8*8*96, 1024))
    -- model_D:add(cudnn.ReLU(true))
    -- model_D:add(nn.Dropout())
    -- model_D:add(nn.Linear(1024,1))
    -- model_D:add(nn.Sigmoid())
 

    -- model = torch.load('resnet-34.t7') 
    -- linear = nn.Linear(512, 1)
    -- linear.bias:zero()
    -- model:remove()
    -- model:add(linear)
    -- model:add(nn.Sigmoid())
    -- model_D = model:cuda()
    -- cudnn.convert(model_D, cudnn) 
    model = torch.load('/media/eunbin/Data2/Dis_model/ROT/baseline-epoch-6.net')
    model:remove()
    model:remove()
    model:remove() -- add
    model:add(nn.LeakyReLU()) -- add
    -- model:add(nn.Dropout(0.5))
    model:add(nn.Linear(512,1))
    model:add(nn.Sigmoid())
    --model:add(nn.Tanh())
    model_D = model:cuda()
    cudnn.convert(model_D, cudnn) 

    -- x_input = nn.Identity()()
    -- --[[
    -- lg = cudnn.SpatialConvolution(3, 256, 3, 3, 1, 1, 1, 1)(x_input)
    -- lg = cudnn.ReLU(true)(lg)
    -- lg = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(lg)
    -- lg = cudnn.ReLU(true)(lg)
    -- lg = cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, 1, 1)(lg)
    -- lg = cudnn.ReLU(true)(lg)
    -- lg = cudnn.SpatialConvolution(128, 64, 3, 3, 1, 1, 1, 1)(lg)
    -- lg = cudnn.ReLU(true)(lg)
    -- lg = cudnn.SpatialConvolution(64, 32, 1, 1)(lg)
    -- lg = cudnn.ReLU(true)(lg)
    -- lg = cudnn.SpatialConvolution(32, 3, 3, 3, 1, 1, 1, 1)(lg)
    -- --]]
    -- lg = cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1)(x_input)
    -- lg = cudnn.ReLU(true)(lg)
    -- lg = cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)(lg)
    -- lg = cudnn.ReLU(true)(lg)
    -- lg = cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)(lg)
    -- lg = cudnn.ReLU(true)(lg)
    -- lg = cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)(lg)
    -- lg = cudnn.ReLU(true)(lg)
    -- lg = cudnn.SpatialConvolution(64, 32, 1, 1)(lg)
    -- lg = cudnn.ReLU(true)(lg)
    -- lg = cudnn.SpatialConvolution(32, 3, 3, 3, 1, 1, 1, 1)(lg)
    -- model_G = nn.gModule({x_input}, {lg})

    paths.dofile('spatial_transformer.lua')

    model_G = spanet:cuda()
    cudnn.convert(model_G, cudnn)

else
    print('<trainer> reloading previously trained network: ' .. opt.network)
    tmp = torch.load(opt.network)
    epoch = tmp.opt.epoch
    opt.momentum = tmp.opt.momentum_save
    init_learningRate = tmp.opt.learningRate_save
    model_D = tmp.D
    model_G = tmp.G

    -- log results to files
    for k, valtable in pairs(tmp.train_symbols) do
        for i, val in pairs(valtable) do
            trainLogger:add{[k]=val}
        end
    end
    for k, valtable in pairs(tmp.test_symbols) do
        for i, val in pairs(valtable) do
            testLogger:add{[k]=val}
        end
    end

end

-- loss function: negative log-likelihood
criterion = nn.BCECriterion()
criterion_labeled = nn.ROTSizeCriterion_boundary()
--criterion = nn.CrossEntropyCriterion()


-- retrieve parameters and gradients
parameters_D,gradParameters_D = model_D:getParameters()
parameters_G,gradParameters_G = model_G:getParameters()

-- print networks
print('Discriminator network:')
print(model_D)
print('Generator network:')
print(model_G)

-- this matrix records the current confusion across classes
classes = {'0','1'}
confusion = optim.ConfusionMatrix(classes)


if opt.gpu then
    print('Copy model to gpu')
    model_D:cuda()
    model_G:cuda()
end

-- Training parameters
sgdState_D = {
    learningRate = init_learningRate,
    momentum = opt.momentum,
    optimize=true,
    numUpdates = 0
}

sgdState_G = {
    learningRate = init_learningRate * 0.1,
    momentum = opt.momentum,
    optimize=true,
    numUpdates=0
}

-- Get examples to plot
function getSamples(dataset, N)
    local N = N or 25
    local inputs = dataset[{{1, N}}]
    inputs = inputs:cuda()

    -- Generate samples
    local samples = model_G:forward(inputs)
    local to_plot = {}
    for i=1,N do
        to_plot[#to_plot+1] = samples[i]:float()
    end

    return to_plot
end


function get_filepaths(folder)
    file_paths = {}
    i = 1;
    path = folder

    for file in lfs.dir(path) do
        if(file == ".") then
            for l in lfs.dir(path.."/"..file) do
                if string.match(l, "h5") then
                    file_paths[i] = path.."/"..l
                    i = i + 1;
                end
            end
        end
    end

    return file_paths
end

-- good_train_filepaths = get_filepaths("/media/eunbin/Data2/ROT_Data/train_data/good")
-- bad_train_filepaths = get_filepaths("/media/eunbin/Data2/ROT_Data/train_data/bad")
 good_test_filepaths = get_filepaths("/media/eunbin/Data2/good&badData/Adversarial_train_data/good_test")
 bad_test_filepaths = get_filepaths("/media/eunbin/Data2/good&badData/Adversarial_train_data/bad_test")


-- night_train_filepaths = get_filepaths("/home/jylee/ssd_1T/junyonglee/datasets/hdf5_256_noavg/train_night")
-- original_train_filepaths = get_filepaths("/home/jylee/ssd_1T/junyonglee/datasets/hdf5_256_noavg/train_original")
-- night_test_filepaths = get_filepaths("/home/jylee/ssd_1T/junyonglee/datasets/hdf5_256_noavg/test_night")
-- original_test_filepaths = get_filepaths("/home/jylee/ssd_1T/junyonglee/datasets/hdf5_256_noavg/test_original")

ntrain = 1280
ntest = 1280



function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

function lines_from(file)
  if not file_exists(file) then return {} end
  lines = {}
  for line in io.lines(file) do 
    lines[#lines + 1] = line
  end
  return lines
end

local train_file_good = 'train_hdf5_good.txt'
local train_list_good = lines_from(train_file_good)

local train_file_bad = 'train_hdf5_bad.txt'
local train_list_bad = lines_from(train_file_bad)

local train_file_labeled = 'train_hdf5_labeled_three.txt'
local train_list_labeled = lines_from(train_file_labeled)

-- for f = 1,#trainlist do
--         -- print("File ", f, "Done.")
--         collectgarbage()
--         trainFile = hdf5.open(trainlist[f], 'r')
--         local dataset = trainFile:all()
--         trainFile:close()

--         data = dataset.data:transpose(3,4)
--         label = dataset.label
--         dataset.size = data:size()[1]

--         for t = 1,dataset.size,batchSize do
--             --print(f, t, sample_count)
--             -- prepare input batch
--             local inputs = data:sub(t,math.min(t+batchSize-1, dataset.size))
--             inputs = inputs:cuda()
--             local targets = label:sub(t,math.min(t+batchSize-1, dataset.size))
--             -- targets = targets:cuda()

--             collectgarbage()





print("starting training/testing")
-- training loop
while true do

    torch.setdefaulttensortype('torch.FloatTensor')

    trainLogger:style{['% mean class accuracy (train set)'] = '-'}
    testLogger:style{['% mean class accuracy (test set)'] = '-'}
    trainLogger:plot()
    testLogger:plot()

    -- train
    for train_index = 1, #train_list_labeled do
    --for train_index = 1, 1 do
        --------------gathering dataset------------
        print("reading good composition set hdf5:"..train_index.."/"..#train_list_good)
        local good_set = nil
        local good_train_data = nil
        collectgarbage()
        local good_set = hdf5.open(train_list_good[train_index], 'r')
        local good_train_data = good_set:read('data'):all()
        good_set:close()
        good_train_data = good_train_data:transpose(3,4)


        print("reading bad composition set hdf5:"..train_index.."/"..#train_list_bad)
        local bad_set = nil
        local bad_train_data = nil
        collectgarbage()
        local bad_set = hdf5.open(train_list_bad[train_index], 'r')
        local bad_train_data = bad_set:read('data'):all()
        bad_set:close()
        bad_train_data = bad_train_data:transpose(3,4)


        print("reading labeled composition set hdf5:"..train_index.."/"..#train_list_labeled)
        local labeled_set = nil
        local labeled_train = nil
        collectgarbage()
        local labeled_set = hdf5.open(train_list_labeled[train_index], 'r')
        local labeled_train = labeled_set:all()
        labeled_set:close()
        labeled_train_data = labeled_train.data:transpose(3,4)
        labeled_train_label = labeled_train.label


        trainData_good = good_train_data[{{1, ntrain}}]
        trainData_bad = bad_train_data[{{1, ntrain}}]
        trainData_labeled = labeled_train_data[{{1, ntrain}}]
        trainLabel_labeled = labeled_train_label[{{1, ntrain}}]
        -------------------------------------------

        if opt.gpu then
            torch.setdefaulttensortype('torch.CudaTensor')
        else
            torch.setdefaulttensortype('torch.FloatTensor')
        end

        print("training: "..train_index.."/"..#train_list_labeled)
        --adversarial.train(trainData_original, trainData_night)
        adversarial.train(trainData_bad, trainData_good, trainData_labeled, trainLabel_labeled)

    end

    model_D:clearState();
    model_G:clearState();

    -- print confusion matrix
    print(confusion)
    trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
    confusion:zero()


    -- test
    for test_index = 1, #good_test_filepaths do
    --for test_index = 1, 1 do
        print("reading good composition test set hdf5:"..test_index.."/"..#good_test_filepaths)
        local good_set = nil
        local good_test_data = nil
        collectgarbage()
        local good_set = hdf5.open(good_test_filepaths[test_index], 'r')
        local good_test_data = good_set:read('data'):all()
        -- night_test_data:mul(2):add(-1)
        good_set:close()

        print("reading bad composition test set hdf5:"..test_index.."/"..#good_test_filepaths)
        local bad_set = nil
        local bad_test_data = nil
        collectgarbage()
        local bad_set = hdf5.open(bad_test_filepaths[test_index], 'r')
        local bad_test_data = bad_set:read('data'):all()
        -- original_test_data:mul(2):add(-1)
        bad_set:close()

        valData_good = good_test_data[{{1, ntest}}]
        valData_bad = bad_test_data[{{1, ntest}}]

        local to_plot = getSamples(valData_bad, 25)
        torch.setdefaulttensortype('torch.FloatTensor')
        local formatted = image.toDisplayTensor({input=to_plot, nrow=5})
        formatted:float()
        image.save(opt.save .."/lfw_example_v1_"..(epoch or 0)..'.png', formatted)

        if opt.gpu then
            torch.setdefaulttensortype('torch.CudaTensor')
        else
            torch.setdefaulttensortype('torch.FloatTensor')
        end

        print("testing: "..test_index.."/"..#good_test_filepaths)
        --adversarial.test(valData_original, valData_night)
        adversarial.test(valData_bad, valData_good)
    end
    model_D:clearState();
    model_G:clearState();

  

    -- print confusion matrix
    print(confusion)
    testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
    confusion:zero()

    sgdState_D.momentum = math.min(sgdState_D.momentum + 0.0008, 0.7)
    sgdState_D.learningRate = math.max(opt.learningRate*0.99^epoch, 0.000001)
    sgdState_G.momentum = math.min(sgdState_G.momentum + 0.0008, 0.7)
    sgdState_G.learningRate = math.max(opt.learningRate*0.99^epoch, 0.000001)

    -- save/log current net
    if epoch % 1 == 0 then
        local filename = '/media/eunbin/Data2/Loc_labeled_model_revised/ROT_Size_semi_model/adversarial-epoch-' .. tostring(epoch) .. '.net'
        os.execute('mkdir -p ' .. sys.dirname(filename))
        --if paths.filep(filename) then
        --os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
        --end
        print('<trainer> saving network to '..filename)
        -- Getting rid of unnecessary things and freeing the memory
        opt.epoch = epoch
        opt.momentum_save = sgdState_D.momentum
        opt.learningRate_save = sgdState_D.learningRate
        torch.save(filename, {D = model_D, G = model_G, opt = opt, train_symbols = trainLogger.symbols, test_symbols = testLogger.symbols})
    end

    epoch = epoch + 1

end
