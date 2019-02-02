function out = MoveImage(in)
    out = imrotate(in, 23.5);
    out = imtranslate(out, [-3,0]);
    
    [M, N] = size(out);
    [m, n] = size(in);
    out = imcrop(out, [ceil((N-n+1)/2), ceil((M-m+1)/2), n-1, m-1]);
    % Caution : floor ? ceil
    
    noise = uint8(8 * rand(size(out)));
    out = out +  noise;
    out(out < 0)   = 0;
    out(out > 255) = 255;
end