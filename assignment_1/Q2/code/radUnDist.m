function imOut = radUnDist(imIn, k1, k2, nSteps)
    % Your code here
    [m, n] = size(imIn);
    [x, y]=meshgrid(1:n, 1:m);
    cx = m/2;
    cy = n/2;
    
    x = x - cx;
    y = y - cy;
    x = x/cx;
    y = y/cy;

    x_d = x;
    y_d = y;
    for iter = 1: nSteps
        r = sqrt(x.^2 + y.^2);
        dr = k1*r + k2*r.^2;
        % explanation on how this equation was derived is given in the PDF
        x = x_d - x.*dr;
        y = y_d - y.*dr;
    end
    
    x = x*cx + cx;
    y = y*cy + cy;
    
    imOut = interp2(imIn, x, y, 'cubic');
end