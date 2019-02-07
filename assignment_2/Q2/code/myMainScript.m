%% Rigit Transform between 2 sets of 3D Points

%% Load Data
img = imread('../input/wembley.jpeg');
figure, imshow(img)
impixelinfo

% Bottom Right: (1024, 813)
% Top Right: (1142, 519)
% Top Left: (374, 436)
% Right Dee: (1061, 721), (845, 682), (962, 537), (1126, 559)
% Left Dee: (67, 544), (178, 564), (410, 471), (314, 457)
field = [[0 0]; [-18 0]; [-18 44]; [0 44]];
rdee = [[1061 721]; [845 682]; [962 537]; [1126 559]];
H = homography(field,rdee);  
% disp(H);
BR = [1024 813];
TR = [1142 519];
TL = [374 436];

br = H*[BR 1]';
br = br/br(3);
tr = H*[TR 1]';
tr = tr/tr(3);
tl = H*[TL 1]';
tl = tl/tl(3);

length = sqrt(sum((tr-tl).^2));
width = sqrt(sum((tr-br).^2));

disp(length);
disp(width);

%% Your code here