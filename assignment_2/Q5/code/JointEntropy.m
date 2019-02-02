function H = JointEntropy(X,Y)
    
    X = ceil(X/10);
    Y = ceil(Y/10);
    
    indrow = double(X(:)) + 1;
    indcol = double(Y(:)) + 1; %// Should be the same size as indrow
    jointHistogram = accumarray([indrow indcol], 1);
    jointProb = jointHistogram / numel(indrow);
    indNoZero = jointHistogram ~= 0;
    jointProb1DNoZero = jointProb(indNoZero);
    jointEntropy = -sum(jointProb1DNoZero.*log2(jointProb1DNoZero));
    H = jointEntropy;
end