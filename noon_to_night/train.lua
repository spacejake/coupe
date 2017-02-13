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

ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
adversarial = require 'adversarial'


----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
-s,--save          (default "logs_256_night_noon")      subdirectory to save logs
--saveFreq         (default 1)          save every saveFreq epochs
-n,--network       (default "")          reload pretrained network
-p,--plot                                plot while training
-r,--learningRate  (default 0.001)        learning rate
--learningRate_save(default 0.001)        learning rate for recovery
-b,--batchSize     (default 20)         batch size
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
    torch.setdefaulttensortype('torch.CudaTensor')
else
    torch.setdefaulttensortype('torch.FloatTensor')
end

opt.geometry = {3, opt.scale, opt.scale}

local input_sz = opt.geometry[1] * opt.geometry[2] * opt.geometry[3]

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

if opt.network == '' then
    ----------------------------------------------------------------------
    -- define D network to train
    model_D = nn.Sequential()
    model_D:add(cudnn.SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2))
    model_D:add(cudnn.SpatialMaxPooling(2,2)) -- 128 X 128
    model_D:add(cudnn.ReLU(true))
    model_D:add(nn.SpatialDropout(0.2))
    model_D:add(cudnn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2))
    model_D:add(cudnn.SpatialMaxPooling(2,2)) -- 64 X 64
    model_D:add(cudnn.ReLU(true))
    model_D:add(nn.SpatialDropout(0.2))
    model_D:add(cudnn.SpatialConvolution(64, 96, 5, 5, 1, 1, 2, 2))
    model_D:add(cudnn.ReLU(true))
    model_D:add(cudnn.SpatialMaxPooling(2,2)) -- 32 X 32
    model_D:add(nn.SpatialDropout(0.2))
    model_D:add(cudnn.SpatialConvolution(96, 96, 5, 5, 1, 1, 2, 2))
    model_D:add(cudnn.ReLU(true))
    model_D:add(cudnn.SpatialMaxPooling(2,2)) -- 16 X 16
    model_D:add(nn.SpatialDropout(0.2))
    model_D:add(cudnn.SpatialConvolution(96, 96, 5, 5, 1, 1, 2, 2))
    model_D:add(cudnn.ReLU(true))
    model_D:add(cudnn.SpatialMaxPooling(2,2)) -- 8 X 8
    model_D:add(nn.SpatialDropout(0.2))
    model_D:add(nn.Reshape(8*8*96))
    model_D:add(nn.Linear(8*8*96, 1024))
    model_D:add(cudnn.ReLU(true))
    model_D:add(nn.Dropout())
    model_D:add(nn.Linear(1024,1))
    model_D:add(nn.Sigmoid())

    x_input = nn.Identity()()
    lg = cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1)(x_input)
    lg = cudnn.ReLU(true)(lg)
    lg = cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)(lg)
    lg = cudnn.ReLU(true)(lg)
    lg = cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)(lg)
    lg = cudnn.ReLU(true)(lg)
    lg = cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)(lg)
    lg = cudnn.ReLU(true)(lg)
    lg = cudnn.SpatialConvolution(64, 32, 1, 1)(lg)
    lg = cudnn.ReLU(true)(lg)
    lg = cudnn.SpatialConvolution(32, 3, 3, 3, 1, 1, 1, 1)(lg)
    model_G = nn.gModule({x_input}, {lg})


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
    learningRate = init_learningRate,
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
    path = "/home/junyonglee/hdf5_256/"..folder

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

night_train_filepaths = get_filepaths("train_night")
original_train_filepaths = get_filepaths("train_original")
night_test_filepaths = get_filepaths("test_night")
original_test_filepaths = get_filepaths("test_original")

ntrain = 5000
ntest = 5000

print("starting training/testing")
-- training loop
while true do

    torch.setdefaulttensortype('torch.FloatTensor')

    trainLogger:style{['% mean class accuracy (train set)'] = '-'}
    testLogger:style{['% mean class accuracy (test set)'] = '-'}
    trainLogger:plot()
    testLogger:plot()

    -- train
    for train_index = 1, #night_train_filepaths do
    -- for train_index = 1, 1 do
        --------------gathering dataset------------
        print("reading night set hdf5:"..train_index.."/"..#night_train_filepaths)
        local night_set = nil
        local night_train_data = nil
        collectgarbage()
        local night_set = hdf5.open(night_train_filepaths[train_index], 'r')
        local night_train_data = night_set:read('data'):all()
        -- night_train_data:mul(2):add(-1)
        night_set:close()

        print("reading original set hdf5:"..train_index.."/"..#night_train_filepaths)
        local original_set = nil
        local original_train_data = nil
        collectgarbage()
        local original_set = hdf5.open(original_train_filepaths[train_index], 'r')
        local original_train_data = original_set:read('data'):all()
        -- original_train_data:mul(2):add(-1)
        original_set:close()

        trainData_night = night_train_data[{{1, ntrain}}]
        trainData_original = original_train_data[{{1, ntrain}}]
        -------------------------------------------

        if opt.gpu then
            torch.setdefaulttensortype('torch.CudaTensor')
        else
            torch.setdefaulttensortype('torch.FloatTensor')
        end

        print("training: "..train_index.."/"..#night_train_filepaths)
        adversarial.train(trainData_original, trainData_night)

    end

    model_D:clearState();
    model_G:clearState();

    -- print confusion matrix
    print(confusion)
    trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
    confusion:zero()


    -- test
    for test_index = 1, #night_test_filepaths do
    -- for test_index = 1, 1 do
        print("reading night test set hdf5:"..test_index.."/"..#night_test_filepaths)
        local night_set = nil
        local night_test_data = nil
        collectgarbage()
        local night_set = hdf5.open(night_test_filepaths[test_index], 'r')
        local night_test_data = night_set:read('data'):all()
        -- night_test_data:mul(2):add(-1)
        night_set:close()

        print("reading original test set hdf5:"..test_index.."/"..#night_test_filepaths)
        local original_set = nil
        local original_test_data = nil
        collectgarbage()
        local original_set = hdf5.open(original_test_filepaths[test_index], 'r')
        local original_test_data = original_set:read('data'):all()
        -- original_test_data:mul(2):add(-1)
        original_set:close()

        valData_night = night_test_data[{{1, ntest}}]
        valData_original = original_test_data[{{1, ntest}}]

        local to_plot = getSamples(valData_original, 25)
        torch.setdefaulttensortype('torch.FloatTensor')
        local formatted = image.toDisplayTensor({input=to_plot, nrow=5})
        formatted:float()
        image.save(opt.save .."/lfw_example_v1_"..(epoch or 0)..'.png', formatted)

        if opt.gpu then
            torch.setdefaulttensortype('torch.CudaTensor')
        else
            torch.setdefaulttensortype('torch.FloatTensor')
        end

        print("testing: "..test_index.."/"..#night_test_filepaths)
        -- adversarial.test(valData_original, valData_night)
        adversarial.test(valData_night, valData_original)
    end
    model_D:clearState();
    model_G:clearState();

    epoch = epoch + 1

    -- print confusion matrix
    print(confusion)
    testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
    confusion:zero()

    sgdState_D.momentum = math.min(sgdState_D.momentum + 0.0008, 0.7)
    sgdState_D.learningRate = math.max(opt.learningRate*0.99^epoch, 0.000001)
    sgdState_G.momentum = math.min(sgdState_G.momentum + 0.0008, 0.7)
    sgdState_G.learningRate = math.max(opt.learningRate*0.99^epoch, 0.000001)

    -- save/log current net
    if epoch % opt.saveFreq == 0 then
        local filename = paths.concat(opt.save, 'adversarial'..(epoch-1)..'.net')
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



end
