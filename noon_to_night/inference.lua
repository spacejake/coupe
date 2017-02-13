require 'cudnn'
require 'nn'
require 'image'
require 'nngraph'
require 'lfs'


network = 'logs_256_avg/adversarial52.net'
function get_filepaths(folder)
    file_paths = {}
    i = 1;
    path = folder

    for file in lfs.dir(path) do
        if(file == '.') then
            for l in lfs.dir(path..'/'..file) do
                if string.match(l, 'png') or string.match(l, 'jpg') then
                    file_paths[i] = l
                    i = i + 1;
                end
            end
        end
    end

    return file_paths
end

cmd = torch.CmdLine()
cmd:option('-net', network, 'trained model')
--cmd:option('-net', 'logs_patch_256_cluster/adversarial192.net', 'trained model')
opt = cmd:parse(arg or {})

print('<trainer> loading previously trained network: ' .. opt.net)
tmp = torch.load(opt.net)
model_D = tmp.D
model_G = tmp.G

--file_paths = get_filepaths('datasets/test')
file_paths = get_filepaths('copyrightfree')
for i = 1,#file_paths,1 do

    --img = image.load('datasets/test/'..file_paths[i], 3, 'float')
    img = image.load('copyrightfree/'..file_paths[i], 3, 'float')
    img = image.scale(img, 1080)
    print(file_paths[i])

    input = img:view(1, img:size()[1], img:size()[2], img:size()[3])
    --input:mul(2):add(-1)
    input = input:cuda()

    output = model_G:forward(input)
    --output:add(1):mul(0.5)
    image.save('copyrightfree/result/'..file_paths[i], output[1])
    --image.save('sample/'..file_paths[i], output[1])
end

print('done.')

--itorch.image({y_crop, y_low_crop, output})
--itorch.image(y_low)
