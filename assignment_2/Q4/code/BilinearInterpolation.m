function [val] = BilinearInterpolation(img, p)
    x = p(1);
    y = p(2);
    fx = floor(x);
    fy = floor(y);
    val = img(fx,fy,:)*(fx+1-x)*(fy+1-y)+img(fx+1,fy,:)*(x-fx)*(fy+1-y)+img(fx,fy+1,:)*(fx+1-x)*(y-fy)+img(fx+1,fy+1)*(x-fx)*(y-fy);
end