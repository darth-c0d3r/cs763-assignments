%% MyMainScript

tic;
%% Your code here

% Firstly, the second image is set as base image
% We then find homography using RANSAC between feature points of the first and second
% image, and the third and second images (if num_images = 3).
% Once these are obtained image warping is done on a plane image by inverse
% mapping from I1 and I3 (possibly), and RGB values are averaged in points of overlap,
% leaving all other points as zero to construct the final image.

% Uncomment required task below:
% task = 'hill'; num_images = 3; thresh = 0.2;
task = 'ledge'; num_images = 3; thresh = 2;
% task = 'pier'; num_images = 3; thresh = 1;
% task = 'monument'; num_images = 2; thresh = 2;
% task = 'ground'; num_images = 3; thresh = 5;
% task = 'room'; num_images = 3; thresh = 5;

rgb1 = im2double(imread(strcat('../input/',task,'/1.JPG')));
rgb2 = im2double(imread(strcat('../input/',task,'/2.JPG')));

img1 = rgb2gray(rgb1);
img2 = rgb2gray(rgb2);

[h,w] = size(img1);
output_img = zeros(3*h,3*w,3);

p1 = detectSURFFeatures(img1);
p2 = detectSURFFeatures(img2);

[f1,v1] = extractFeatures(img1,p1);
[f2,v2] = extractFeatures(img2,p2);

indexPairs1 = matchFeatures(f1,f2);
mp1_1 = v1(indexPairs1(:,1));
mp2_1 = v2(indexPairs1(:,2));

H1 = ransacHomography(mp2_1.Location,mp1_1.Location,thresh);

if num_images == 3
    rgb3 = im2double(imread(strcat('../input/',task,'/3.JPG')));
    img3 = rgb2gray(rgb3);
    
    p3 = detectSURFFeatures(img3);
    [f3,v3] = extractFeatures(img3,p3);
    indexPairs2 = matchFeatures(f2,f3);

    mp2_2 = v2(indexPairs2(:,1));
    mp3_2 = v3(indexPairs2(:,2));
    H2 = ransacHomography(mp2_2.Location,mp3_2.Location,thresh);
end

for i = 1:size(output_img,1)
    for j = 1:size(output_img,2)
        res2 = [i-h j-w 1]';
        res1 = H1\res2;
        res1 = res1/res1(3);    
        cnt = 0;
        if res1(1)>=2 && res1(2)>=2 && res1(1)<=h-2 && res1(2)<=w-2
            output_img(i,j,:) = output_img(i,j,:)+rgb1(ceil(res1(1)),ceil(res1(2)),:);
            cnt = cnt + 1;
        end    
        if res2(1)>=2 && res2(2)>=2 && res2(1)<=h-2 && res2(2)<=w-2
            output_img(i,j,:) = output_img(i,j,:)+rgb2(ceil(res2(1)),ceil(res2(2)),:);
            cnt = cnt + 1;
        end
        
        if num_images == 3
            res3 = H2\res2;
            res3 = res3/res3(3);
            if res3(1)>=2 && res3(2)>=2 && res3(1)<=h-2 && res3(2)<=w-2
                output_img(i,j,:) = output_img(i,j,:)+rgb3(ceil(res3(1)),ceil(res3(2)),:);
                cnt = cnt + 1;
            end
        end
        
        if cnt > 0
            output_img(i,j,:) = output_img(i,j,:)/cnt;
        end    
    end
end    

figure, imshow(output_img);
imwrite(output_img,strcat('../output/',task,'.jpg'));

toc;


