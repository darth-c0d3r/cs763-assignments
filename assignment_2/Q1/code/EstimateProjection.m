function P = EstimateProjection(x, X, T, U)
%   x = n x 3
%   X = n x 4
%   T = 
%   n = number of sample points
    n = size(x,1);
    M = zeros(2*n, 12); % 12 x 12
    
%   Homogenize x and X
    x = x ./ x(:,3);
    X = X ./ X(:,4);
    
    for i = 1:n
        M(2*i-1,:) = [-X(i,1:3) -1 0 0 0 0 x(i,1)*X(i, 1:3) x(i,1)];
        M(2*1  ,:) = [0 0 0 0 -X(i,1:3) -1 x(i,2)*X(i, 1:3) x(i,2)];   
    end
    [~,~,V] = svd(M);
    p = V(:,12);
    P = [p(1:4,:)'; p(5:8,:)'; p(9:12,:)'];
    P = (T\P) * U; % equivalent to inv(T) * P * U
end