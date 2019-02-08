%% For Barbara

imF = imread('../input/barbara.png');
imM = imread('../input/negative_barbara.png');
imM = MoveImage(imM);
img_out = RegisterImage(imF, imM);
figure
imshow(img_out);

%% For Flash

imF1 = imread('../input/flash1.jpg');
imM1 = imread('../input/noflash1.jpg');
imF = rgb2gray(imF1); % convert to greyscale
imM = rgb2gray(imM1); % convert to greyscale
imM = MoveImage(imM);
img_out = RegisterImage(imF, imM);
figure
imshow(img_out);