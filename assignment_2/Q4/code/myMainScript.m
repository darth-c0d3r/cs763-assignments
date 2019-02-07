%% MyMainScript

tic;
%% Your code here

% Parameters ('thresh' value):
% Ledge => thresh = 2;
% Hill => thresh = 0.2;
% Monument
% Pier => 

num_images = 3;

rgb1 = im2double(imread('../input/hill/1.JPG'));
rgb2 = im2double(imread('../input/hill/2.JPG'));
rgb3 = im2double(imread('../input/hill/3.JPG'));

thresh = 1;

img1 = rgb2gray(rgb1);
img2 = rgb2gray(rgb2);
img3 = rgb2gray(rgb3);

p1 = detectSURFFeatures(img1);
p2 = detectSURFFeatures(img2);
p3 = detectSURFFeatures(img3);

[f1,v1] = extractFeatures(img1,p1);
[f2,v2] = extractFeatures(img2,p2);
[f3,v3] = extractFeatures(img3,p3);

indexPairs1 = matchFeatures(f1,f2);
indexPairs2 = matchFeatures(f2,f3);
indexPairs3 = matchFeatures(f1,f3);

mp1_1 = v1(indexPairs1(:,1));
mp2_1 = v2(indexPairs1(:,2));

mp2_2 = v2(indexPairs2(:,1));
mp3_2 = v3(indexPairs2(:,2));

mp1_3 = v1(indexPairs3(:,1));
mp3_3 = v3(indexPairs3(:,2));

H1 = ransacHomography(mp2_1.Location,mp1_1.Location,thresh);
H2 = ransacHomography(mp2_2.Location,mp3_2.Location,thresh);

[h,w] = size(img1);
output_img = zeros(3*h,3*w,3);

for i = 1:3*h
    for j = 1:3*w
        res2 = [i-h j-w 1]';
        res1 = H1\res2;
        res1 = res1/res1(3);
        res3 = H2\res2;
        res3 = res3/res3(3);
        cnt = 0;
        if res1(1)>=1 && res1(2)>=1 && res1(1)<=h && res1(2)<=w
            output_img(i,j,:) = output_img(i,j,:)+rgb1(ceil(res1(1)),ceil(res1(2)),:);
            cnt = cnt + 1;
        end    
        if res2(1)>=1 && res2(2)>=1 && res2(1)<=h && res2(2)<=w
            output_img(i,j,:) = output_img(i,j,:)+rgb2(ceil(res2(1)),ceil(res2(2)),:);
            cnt = cnt + 1;
        end    
        if res3(1)>=1 && res3(2)>=1 && res3(1)<=h && res3(2)<=w
            output_img(i,j,:) = output_img(i,j,:)+rgb3(ceil(res3(1)),ceil(res3(2)),:);
            cnt = cnt + 1;
        end
        if cnt > 0
            output_img(i,j,:) = output_img(i,j,:)/cnt;
        end    
    end
end    

figure, imshow(output_img);



% figure; showMatchedFeatures(img1,img2,mp1_1,mp2_1);
% legend('matched points 1','matched points 2');
% 
% figure; showMatchedFeatures(img2,img3,mp2_2,mp3_2);
% legend('matched points 2','matched points 3');
% 
% figure; showMatchedFeatures(img1,img3,mp1_3,mp3_3);
% legend('matched points 1','matched points 3');

toc;


