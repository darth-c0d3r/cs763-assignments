%% Creating 3D Dataset [X]
% Dimensions of X : n x 4 [n = 6]
X = [4 1 0 1 ;
     2 2 0 1 ;
     0 2 5 1 ;
     0 4 3 1 ;
     3 0 2 1 ;
     2 0 6 1 ;
     1 1 0 1 ;
     1 3 0 1 ;
     4 3 0 1 ;
     0 2 2 1 ;
     0 2 7 1 ;
     0 3 6 1 ;
     1 0 4 1 ;
     2 0 8 1 ;
     4 0 6 1 ;
     ]';
 
 X_test = [
     1 0 0 1;
     1 5 0 1;
     2 1 0 1;
     2 3 0 1;
     3 3 0 1;
     3 4 0 1;
     4 5 0 1;
     4 4 0 1;
     1 0 1 1;
     1 0 7 1;
     2 0 3 1;
     2 0 5 1;
     3 0 6 1;
     3 0 8 1;
     4 0 4 1;
     4 0 1 1;
     0 1 1 1;
     0 3 1 1;
     0 1 2 1;
     0 5 2 1;
     0 5 4 1;
     0 1 4 1;
     0 0 6 1;
     0 5 6 1;
     ]';
 
 %% Creating 2D Dataset [x]
 % Dimensions of X : n x 3 [n = 6]
 
%  uncomment to obtain 2D image coordinates
%  img = imread('../input/checkerbox_3D_marked.png');
%  imshow(img);
%  impixelinfo;
 
 x = [573 439 1 ;
      448 363 1 ;
      160 414 1 ;
      238 241 1 ;
      474 537 1 ;
      272 636 1 ;
      389 418 1 ;
      389 295 1 ;
      585 307 1 ;
      276 372 1 ;
      043 450 1 ;
      099 341 1 ;
      276 555 1 ;
      156 729 1 ;
      454 676 1 ;
      ]';
  
  T = normalizeCoordinates(x);
  U = normalizeCoordinates(X);
  
  x_ = T*x;
  X_ = U*X;
  
  P = EstimateProjection(x_, X_, T, U);
  [K,R,X0] = DecomposeProjection(P);
%   disp(P);
%   disp(K);
%   disp(R);
%   disp(X0);
  xx = P*X;
  xx = xx ./ xx(3,:);
  
  diff = (xx-x).^2;
  RMSE = sqrt(mean(diff(:)));
  str = sprintf("RMSE = %f\n", RMSE);
  disp(str);
  
  x_test = P*X_test;
  x_test = x_test ./ x_test(3,:);

  imshow('../input/checkerbox_3D_original.jpg');
  hold on;
  plot(x(1,:), x(2,:),'r.','MarkerSize',10);
  hold on;
  plot(xx(1,:),xx(2,:),'yo','MarkerSize',5);
  hold on;
  plot(x_test(1,:),x_test(2,:),'co','MarkerSize',5);
  
  