% Mohsen Dorraki
% Deep learning 
% 21/02/2020
% https://au.mathworks.com/help/deeplearning/examples/create-simple-deep-learning-network-for-classification.html

clc
clear
close all

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %                          Data Prepration
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Reading the Control images
% Addr_Ct = 'F:\Desktop\Ideas\Bone\Code\4.Deeplearning\Data\Cnt';
% filePattern_Ct = fullfile(Addr_Ct, '*.bmp');
% bmpFiles = dir(filePattern_Ct);
% 
% 
% for k = 1:length(bmpFiles)
%   baseFileName = bmpFiles(k).name;
%   fullFileName = fullfile(Addr_Ct, baseFileName);
%   imageArray_Ct{k} = imread(fullFileName);
% end
% 
% % Reading the OA images
% Addr_OA = 'F:\Desktop\Ideas\Bone\Code\4.Deeplearning\Data\OA';
% filePattern_OA = fullfile(Addr_OA, '*.bmp');
% bmpFiles_OA = dir(filePattern_OA);
% 
% for k = 1:length(bmpFiles_OA)
%   baseFileName = bmpFiles_OA(k).name;
%   fullFileName_OA = fullfile(Addr_OA, baseFileName);
%   imageArray_OA{k} = imread(fullFileName_OA);
% end
% 
% 
% % Making BW (Control)
% for k = 1: length(bmpFiles)% Number of files found
%     BWs{k}  = im2bw(imageArray_Ct{k},0.3);
%     BWs2{k} = imresize(BWs{k},[200 200]);
%     newAddr_Ct='F:\Desktop\Ideas\Bone\Code\4.Deeplearning\Data\BW\Control';
%     fullFileName = fullfile(newAddr_Ct, bmpFiles(k).name);
%     imwrite(BWs2{k}, fullFileName);
% end
% 
% % Making BW (OA)
% for k = 1: length(bmpFiles_OA)% Number of files found
%     BWs_OA{k} = im2bw(imageArray_OA{k},0.3);
%     BWs_OA2{k} = imresize(BWs_OA{k},[200 200]);
%     newAddr_OA='F:\Desktop\Ideas\Bone\Code\4.Deeplearning\Data\BW\OA';
%     fullFileName_OA = fullfile(newAddr_OA, bmpFiles_OA(k).name);
%     imwrite(BWs_OA2{k}, fullFileName_OA);
% end






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                 Deep learning 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Making path, where the files are located
digitDatasetPath = fullfile('F:\Desktop\Ideas\Bone\Code\4.Deeplearning\Data\BW\Training');

% Saving the Normal dataset
imds = imageDatastore(digitDatasetPath,'IncludeSubfolders',true, ...
    'LabelSource','foldernames','FileExtensions',{'.bmp'});

% Calculate the number of images in each category
labelCount = countEachLabel(imds)

% Saving the dataset
imds = imageDatastore(digitDatasetPath,'IncludeSubfolders',true, ...
    'LabelSource','foldernames','FileExtensions',{'.bmp'});

% Calculate the number of images in each category
labelCount = countEachLabel(imds)


img = readimage(imds,1);
size(img)

% Specify Training and Validation Sets   
numTrainFiles = 120;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');


%Define the convolutional neural network architecture.
layers = [
    imageInputLayer([200 200 1],'Name','Image_Input')
    
    convolution2dLayer(3,8,'Padding','same','Name','conv_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu_1')
    
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool_1')
    
    %%%%% Layer_2
    convolution2dLayer(3,16,'Padding','same','Name','conv_2')
    batchNormalizationLayer('Name','BN_2')
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool_2')
    
    %%%% Layer_3
    convolution2dLayer(3,32,'Padding','same','Name','conv_3')
    batchNormalizationLayer('Name','BN_3')
    reluLayer('Name','relu_3')
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool_3')

    %%%% Layer_4
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Refined by Mohsen
    
    convolution2dLayer(3,64,'Padding','same','Name','conv_4')
    batchNormalizationLayer('Name','BN_4')
    reluLayer('Name','relu_4')
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    fullyConnectedLayer(2,'Name','fullyConnected')
    softmaxLayer('Name','softmaxLayer')
    classificationLayer('Name','classificationLayer')];

% Ploting layers
lgraph = layerGraph(layers);
figure (1)
plot(lgraph);


% Specify Training Options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',50, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');


% Train Network Using Training Data
net = trainNetwork(imdsTrain,layers,options);

%Classify Validation Images and Compute Accuracy
[YPred,scores] = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

% Plotting the test images 
figure (2);
title('Displaying some of the images in the test data, Top: Normal, Buttom: OA')
perm = datasample([0:40],4,'Replace',true);
for i = 1:4
    subplot(2,4,i);
    imshow(imdsValidation.Files{perm(i)});
end
hold on
perm = datasample([41:80],4,'Replace',true);
for i = 1:4
    subplot(2,4,i+4);
    imshow(imdsValidation.Files{perm(i)});
end
hold off

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%% Checking single images using our network
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% I  = imread('F:\Desktop\Ideas\Bone\Code\4.Deeplearning\Data\BW\Training\OA\16 (20).bmp');
% BW = im2bw(I,0.3);
% X  = imresize(BW,[200 200]);
% classify(net,X)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Create new imd for test data
% 
% TestPath = fullfile('F:\Desktop\Ideas\Bone\Code\4.Deeplearning\Data\BW\Test');
% 
% % Saving the Normal dataset
% imds_Test = imageDatastore(TestPath,'IncludeSubfolders',true, ...
%     'LabelSource','foldernames','FileExtensions',{'.bmp'});
% 
% % Calculate the number of images in each category
% labelCount = countEachLabel(imds_Test)
% 
% % Saving the dataset
% imds_Test = imageDatastore(TestPath,'IncludeSubfolders',true, ...
%     'LabelSource','foldernames','FileExtensions',{'.bmp'});
% 
% %Classify Validation Images and Compute Accuracy
% YPred_Test = classify(net,imds_Test);
% YValidation_Test = imds_Test.Labels;
% 
% accuracy_Test = sum(YPred_Test == YValidation_Test)/numel(YValidation_Test)



C = confusionmat(YValidation,YPred) %Confusion matrix
figure(3)
confusionchart(C)



