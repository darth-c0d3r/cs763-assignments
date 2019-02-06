function R = normalizeCoordinates(X)
    d = size(X,1); % either 3 or 4
    n = size(X,2); % number of points
    
    t = -sum(X,2) / n;
    T = [eye(d-1) t(1:d-1) ; zeros(1,d-1) 1]; % translation
    X = T*X;
    
    D = 1/sqrt(n-1) * sum(sqrt(sum(X(1:d-1,:).*X(1:d-1,:))),2);
    S = [eye(d-1)/D zeros(d-1,1) ; zeros(1,d-1) 1]; % scaling
    R = S*T;
end