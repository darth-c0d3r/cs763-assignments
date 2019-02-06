%% Creating 3D Dataset [X]
% Dimensions of X : n x 4 [n = 6]
X = [4 1 0 1 ;
     2 2 0 1 ;
     0 2 5 1 ;
     0 4 3 1 ;
     3 0 2 1 ;
     2 0 6 1 ;]';
 
 %% Creating 2D Dataset [x]
 % Dimensions of X : n x 3 [n = 6]
 
%  uncomment to obtain 2D image coordinates
%  img = imread('../input/checkerbox_3D_marked.png');
%  imshow(img);
%  impixelinfo;
 
 x = [573 541 1 ;
      448 363 1 ;
      160 414 1 ;
      238 241 1 ;
      474 537 1 ;
      272 636 1 ;]';
  
  T = normalizeCoordinates(x);
  U = normalizeCoordinates(X);
  
  x_ = T*x;
  X_ = U*X;
  
  P = EstimateProjection(x_, X_, T, U);
  [K,R,X0] = DecomposeProjection(P);
  xx = P*X;
  xx = xx ./ xx(3,:);
  
  diff = (xx-x_).^2;
  RMSE = sqrt(mean(diff(:)));
  str = sprintf("RMSE = %f\n", RMSE);
  disp(str);

  
  