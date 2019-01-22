function rot_matrix = angles2rot(angles_list)
    %% Your code here
    % angles_list: [theta1, theta2, theta3] about the x,y and z axes,
    % respectively.
    rot_matrix = zeros(15,3,3);
    for idx = 1:size(angles_list,1)
        rots  = angles_list(idx,:);
        x_rot = [1 0 0 ; 0 cos(rots(1)) -sin(rots(1)) ; 0 sin(rots(1)) cos(rots(1)) ];
        y_rot = [cos(rots(2)) 0 sin(rots(2)) ; 0 1 0 ; -sin(rots(2)) 0 cos(rots(2)) ];
        z_rot = [cos(rots(3)) -sin(rots(3)) 0 ; sin(rots(3)) cos(rots(3)) 0 ; 0 0 1 ];
        rot_matrix(idx,:,:) = z_rot * y_rot * x_rot;
    end
end




