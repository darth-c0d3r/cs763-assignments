function [ H ] = homography( p1, p2 )
%HOMOGRAPHY Summary of this function goes here
%   Detailed explanation goes here
    sz = size(p1,1);
    M = zeros(2*sz,9);
    for i = 1:sz
        M(2*i-1,1:3) = -[p2(i,:) 1];
        M(2*i-1,7:9) = p1(i,1)*[p2(i,:) 1];
        M(2*i,4:6) = -[p2(i,:) 1];
        M(2*i,7:9) = p1(i,2)*[p2(i,:) 1];
    end
    
    [~, ~, V] = svd(M);
    P = V(:,end);
    H = zeros(3);
    H(1,:) = P(1:3);
    H(2,:) = P(4:6);
    H(3,:) = P(7:9);
end
