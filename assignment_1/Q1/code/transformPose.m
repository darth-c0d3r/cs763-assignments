function [result_pose] = transformPose(rotations, pose, kinematic_chain, root_location)
    % rotations: A 15 x 3 x 3 array for rotation matrices of 15 bones
    % pose: The base pose coordinates 16 x 3.
    % kinematic chain: A 15 x 2 array of joint ordering
    % root_location: the index of the root in pose vector.
    % Your code here
    [result_pose] = transformHelper(rotations, pose, kinematic_chain, root_location, eye(4));    
end

function [pose] = transformHelper(rotations, pose, kinematic_chain, root_location, composed_rot)
    for idx = 1:size(kinematic_chain,1)
        if kinematic_chain(idx,2) == root_location
            child_node = kinematic_chain(idx,1);
            M = [eye(3) pose(root_location,:)'; 0 0 0 1]*[squeeze(rotations(idx,:,:)) zeros(3,1); 0 0 0 1]*[eye(3) -pose(root_location,:)'; 0 0 0 1];
            composed_rot1 = composed_rot*M;
            [pose] = transformHelper(rotations, pose, kinematic_chain, child_node, composed_rot1);
        end
    end
    X = composed_rot*[pose(root_location,:) 1]';
    pose(root_location,:) = X(1:3)/X(4);
end