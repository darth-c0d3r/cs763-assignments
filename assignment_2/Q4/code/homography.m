function [ H ] = homography( p1, p2 )
% HOMOGRAPHY Computes 3X3 Homography matrix given set of points
%   4 points are enough to compute homography. First, we reshape it into a vector, and then apply DLT.
%	It turns out that the last column on applying SVD to the constructed matrix yields the desired result which minimizes error, which is acquired using the code below.
% [p1_i 1] = H*[p2_i 1]
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
