require 'cudnn'
require 'cunn'
require 'nn'
require 'image'
require 'lfs'
require 'cutorch'
require 'optim'
require 'stn'
-- cutorch.setDevice(1)
-- sofar = os.clock()

cmd = torch.CmdLine()
cmd:option('-net', '/media/eunbin/Data2/Loc_labeled_model_revised/ROT_semi_model/adversarial-epoch-9.net', 'trained model')
cmd:option('-dir', '/home/eunbin/torch-model/Ad_add_labeled_composition/star', 'target directory')
opt = cmd:parse(arg or {})
print(opt.net)
print(opt.dir)
net = torch.load(opt.net)
model = net.G
--model:remove(4)
model:evaluate()

i = 0
for file in lfs.dir(opt.dir) do
	if lfs.attributes(opt.dir..'/'..file, "mode") == "file" then
		img = image.load(opt.dir..'/'..file, 3, 'float')
		--transposed_img = img_raw:transpose(2,3)
		--img = image.vflip(transposed_img)
		channel = img:size()[1]
		height = img:size()[2]
		width = img:size()[3]
		output_size = channel * height * width

		input = image.scale(img, '224x224', 'bicubic')
		input = input:view(1, 3, 224, 224)
		if i == 0 then
			input_batch = input
		else
			input_batch = torch.cat(input_batch, input, 1)
		end
		i = i + 1	
	end
end
input_batch = input_batch:cuda()
output_batch = model:forward(input_batch)

for f = 1,(#input_batch)[1] do
	print('input '..f)
	--print(model:get(1):get(2):get(18).output[f])	
	print(model:get(1):get(2):get(1):get(14).output[f])
	input_img = input_batch[f]:float()
	--input_img = image.vflip(input_img)
	--input_img = input_img:transpose(2,3)	
	image.save('/home/eunbin/torch-model/Ad_add_labeled_composition/crop_results/image_'..f.. '_input.jpg', input_img)

	--print(output_batch[f]:float())

	output_img = output_batch[f]:float()
	--output_img = image.vflip(output_img)
	--output_img = output_img:transpose(2,3)
	image.save('/home/eunbin/torch-model/Ad_add_labeled_composition/crop_results/image_'..f..'_res.jpg', output_img)	

end

-- image.save('input.png', y_low_crop[1])
-- image.save('result.png', output[1])

print('done.')
