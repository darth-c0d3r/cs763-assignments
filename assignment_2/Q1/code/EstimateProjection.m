function P = EstimateProjection(x, X, T, U)
%   x = 3 x n [n = 6]
%   X = 4 x n [n = 6]
%   T = 3 x 3
%   U = 4 x 4
%   n = number of sample points
    n = size(x,2);
    M = zeros(2*n, 12); % 12 x 12
    
%   Homogenize x and X
    x = x ./ x(3,:);
    X = X ./ X(4,:);
    
    for i = 1:n
%         M(2*i-1,:) = [-X(1:3, i)' -1 0 0 0 0 x(1,i)*X(1:3,i)' x(1,i)];
%         M(2*1  ,:) = [0 0 0 0 -X(1:3,i)' -1 x(2,i)*X(1:3, i)' x(2,i)];   
          M(2*i-1,:) = [X(1,i), 0, -X(1,i)*x(1,i), X(2,i), 0, -X(2,i)*x(1,i), X(3,i), 0, -X(3,i)*x(1,i), 1, 0, -x(1,i)];
          M(2*i,:) =   [0, X(1,i), -X(1,i)*x(2,i), 0, X(2,i), -X(2,i)*x(2,i), 0, X(3,i), -X(3,i)*x(2,i), 0, 1 -x(2,i)];
    end
    [~,~,V] = svd(M);
    p = V(:,12);
    P = reshape(p, 3,4);
%     P = [p(1:4,:)'; p(5:8,:)'; p(9:12,:)'];
    P = (T\P) * U; % equivalent to inv(T) * P * U
end