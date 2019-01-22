function [result_pose, composed_rot] = transformPose(rotations, pose, kinematic_chain, root_location)
    % rotations: A 15 x 3 x 3 array for rotation matrices of 15 bones
    % pose: The base pose coordinates 16 x 3.
    % kinematic chain: A 15 x 2 array of joint ordering
    % root_location: the index of the root in pose vector.
    % Your code here
    n_parts = size(kinematic_chain,1);
    n_joints = size(pose,1);
    composed_rot = zeros(n_joints,3,3);
    result_pose = zeros(n_joints,3);
    composed_rot(root_location,:,:) = eye(3);
    traversed = zeros(n_joints);
    traversed(1) = root_location;
    count = 1;
    ptr = 1;
    while count <= n_joints
        cur_node = traversed(count);
        result_pose(cur_node,:) = pose(cur_node,:)*squeeze(composed_rot(cur_node,:,:))';
        for idx = 1:n_parts
            if kinematic_chain(idx,2) == cur_node
                ptr = ptr+1;
                new_node = kinematic_chain(idx,1);
                composed_rot(new_node,:,:) = squeeze(composed_rot(cur_node,:,:))*squeeze(rotations(idx,:,:));
                traversed(ptr) = new_node;
            end    
        end    
        count = count+1;
    end    
end