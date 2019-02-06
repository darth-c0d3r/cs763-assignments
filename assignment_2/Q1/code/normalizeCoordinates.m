function R = normalizeCoordinates(X)
    d = size(X,1); % either 3 or 4
    n = size(X,2); % number of points
    
    tx = -sum(X,2) / n;
    Tx = [eye(d-1) tx(1:d-1) ; zeros(1,d-1) 1]; % translation
    X = Tx*X;
    
    D = sqrt(d-1)/mean(sqrt(sum(X(1:d-1,:).^2,1)),2);
    S = [eye(d-1)*D zeros(d-1,1) ; zeros(1,d-1) 1]; % scaling
    R = S*Tx;
end