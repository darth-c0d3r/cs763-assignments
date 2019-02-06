function [K,R,X0] = DecomposeProjection(P)
    
    H = P(:,1:3);
    h = P(:,4);
    
    X0 = -inv(H)*h;
    [R,K] = qr(inv(H));
    R = R';
    K = inv(K);
    K = K ./ K(3,3);
end