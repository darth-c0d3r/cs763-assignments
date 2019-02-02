function out = RegisterImage(ImF, ImM)
    thetas = linspace(-60,60,121);
    tx_s = linspace(-12,12,25);
    
    JE_s = zeros(size(thetas,1), size(tx_s,1));
    
    for i = 1:121
        for j = 1:25
            theta = thetas(i);
            tx = tx_s(j);
            Im_ = imrotate(ImM, theta);
            Im_ = imtranslate(Im_, [tx,0]);
            [M, N] = size(Im_);
            [m, n] = size(ImM);
            Im_ = imcrop(Im_, [ceil((N-n+1)/2), ceil((M-m+1)/2), n-1, m-1]);
            JE_s(i,j) = JointEntropy(ImF, Im_);
        end
    end
    
    [~,I] = min(JE_s(:));
    [i,j] = ind2sub(size(JE_s), I);
    
    out = imrotate(ImM, thetas(i)); 
    out = imtranslate(out, [tx_s(j), 0]);
    [M, N] = size(ImF);
    [m, n] = size(out);
    out = imcrop(out, [ceil((N-n+1)/2), ceil((M-m+1)/2), n-1, m-1]);
    
%     PLOT JE_s
    
    [X,Y] = meshgrid(tx_s, thetas);
    surf(X,Y,JE_s);

    
end