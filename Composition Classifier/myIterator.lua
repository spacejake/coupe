local tnt = require 'torchnet'
local t = dofile ('utils/transforms.lua')
local path_to_data = {}
path_to_data["train"] = {}
path_to_data["val"] = {}
path_to_data["train"][1] = '/media/eunbin/Data2/ROT_Data/bad/'
path_to_data["train"][2] = '/media/eunbin/Data2/ROT_Data/good/'
path_to_data["val"][1] = '/media/eunbin/Data2/good&badData/test_bad2/'
path_to_data["val"][2] = '/media/eunbin/Data2/good&badData/test_good2/'

local dataset = {}
dataset["train"] = {}
dataset["val"] = {}
for l, path in ipairs(path_to_data["train"]) do
   local input_list = paths.dir(path)
   table.sort(input_list)
   table.remove(input_list, 1)
   table.remove(input_list, 1)
   dataset["train"][l] = input_list
end
for l, path in ipairs(path_to_data["val"]) do
   local input_list = paths.dir(path)
   table.sort(input_list)
   table.remove(input_list, 1)
   table.remove(input_list, 1)
   dataset["val"][l] = input_list
end

local myIterator = function(mode, nEpoch)
   return tnt.ParallelDatasetIterator{
      nthread = 1,
      init = function()
         require 'torchnet'
         require 'image'
      end,
      closure = function()
         classes= {}
         for l,class in ipairs(dataset[mode]) do
            local list = tnt.ListDataset{
               list = class,
               load = function(im)
                  --i = image.load(paths.concat(path_to_data[mode][l],im)):float()
                  --print(i.size(i))
                  return {
                     input = image.load(paths.concat(path_to_data[mode][l],im)):float(),
                     target = torch.DoubleTensor{l-1},
                  }
               end,
            }:transform{ -- imagetransformations
               input =
                  mode == 'train' and
                     tnt.transform.compose{
                        t.Fix(),
                        --t.RandomSizedCrop(224),
                        t.ColorJitter({
                           brightness = 0.4,
                           contrast = 0.4,
                           saturation = 0.4,
                        }),
                        -- t.Lighting(0.1, pca.eigval, pca.eigvec),
                        -- t.ColorNormalize(dataset.meanstd),
                        t.HorizontalFlip(0.5),
                        t.Resize(224, 224),
                     }
                  or mode == 'val' and
                     tnt.transform.compose{
                        t.Fix(),
                        -- --t.Scale(256),
                        -- t.ColorNormalize(dataset.meanstd),
                        --t.CenterCrop(224),
                        t.Resize(224, 224),
                     }
            }
            classes[#classes+1] = mode == 'train' and list:shuffle() or list:shuffle()
         end
         return tnt.ConcatDataset{datasets = classes}:shuffle():batch(16,'skip-last')
      end,
   }
end

return myIterator