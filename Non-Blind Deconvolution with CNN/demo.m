clear all, close all;
addpath('caffe/matlab'); %require matcaffe path
weights = 'model/model_a0.caffemodel';
model = 'model/net.prototxt';

%% load an image and a blur kernel
img = imread('kodim11.png');
img = im2double(img);
ker = im2double(imread('k8.png'));
ker = ker(:,:,1) / (sum(sum(ker(:,:,1))));

gt = img;
[h, w, ~] = size(gt);
xest = gpuArray(zeros(h, w, 3));

%% make an synthetic blurred image
img = imfilter(img, ker, 'circular','conv');
noise_var =0.0001;
img = imnoise(img, 'gaussian', 0, noise_var); %0.0005

%% run deconv_cnn
caffe.set_mode_gpu();
net = caffe.Net(model, weights, 'test');
nsr = -1;
result_img = deconv_cnn(img,ker,net,nsr); % if nsr < 0, it uses estimated nsr
caffe.reset_all();

psnr(double(result_img), gt);

%% post processing
tic
alpha = 0.1;
% xest = zeros(h, w, 3);
xest = postprocessing(img, ker, result_img,alpha);
toc

psnr(double(xest),gt);
imwrite(xest,'out.png');