function H = JointEntropy(Im1,Im2)
    
    Im1 = ceil(Im1/10); % bins of 10
    Im2 = ceil(Im2/10);
    
     % Since intensity can be zero, but matlab array index starts with 1, doing + 1
    Im1_int = double(Im1(:)) + 1;
    Im2_int = double(Im2(:)) + 1; 
    
    jointHist = accumarray([Im1_int Im2_int], 1);
    
    jointProb = jointHist / numel(Im1_int);
    
    % Entries in jointHist which are non zero, i.e., intensity values occured in images
    Exist_int = jointHist ~= 0;   
    jointProb = jointProb(Exist_int);
    H = -sum(jointProb.*log2(jointProb));
end