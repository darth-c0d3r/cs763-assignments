function rot_matrix = angles2rot(angles_list)
    %% Your code here
    % angles_list: [theta1, theta2, theta3] about the x,y and z axes,
    % respectively.
    rot_matrix = zeros(15,3,3);
    for idx = 1:size(angles_list,1)
        rots  = angles_list(idx,:);
        x_rot = [1 0 0 ; 0 cosd(rots(1)) -sind(rots(1)) ; 0 sind(rots(1)) cosd(rots(1)) ];
        y_rot = [cosd(rots(2)) 0 sind(rots(2)) ; 0 1 0 ; -sind(rots(2)) 0 cosd(rots(2)) ];
        z_rot = [cosd(rots(3)) -sind(rots(3)) 0 ; sind(rots(3)) cosd(rots(3)) 0 ; 0 0 1 ];
        rot_matrix(idx,:,:) = z_rot * y_rot * x_rot;
    end
end




