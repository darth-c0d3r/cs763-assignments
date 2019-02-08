function [ H ] = ransacHomography( x1, x2, thresh )
%RANSACHOMOGRAPHY Performs Random Sample Consensus among given sets of
%points for a known threshold
%   The location vectors obtained through SURF are (w,h) instead of (h,w)
%   so first they are reversed. Then the number of iterations is set to
%   10000 and on each iteration we randomly select 4 points which are
%   enough to determine homography and pick those which yield maximum size of the consensus set.
%   Then given the consensus set, we reconstruct and return the computed
%   homography.
    x1 = x1(:,2:-1:1);
    x2 = x2(:,2:-1:1);
    n = size(x1,1);
%     disp(size(x1));
    S = 10000;
    best = 0;
    vec = 1:4;
    for iter = 1:S
        idx = randperm(n,4);
        H1 = homography(x1(idx,:),x2(idx,:));
        consensus = 0;
        for i = 1:n
            res = H1*[x2(i,:) 1]';
            res = res/res(3);
            ei = sum((x1(i,:)-res(1:2)').^2);
            if ei < thresh
                consensus = consensus+1;
            end    
        end
        
        if consensus > best
            vec = idx;
            best = consensus;
        end    
    end
   
    H1 = homography(x1(vec,:),x2(vec,:));
    consensus_set = zeros(best,1);
    cnt = 1;
    for i = 1:n
        res = H1*[x2(i,:) 1]';
        res = res/res(3);
        ei = sum((x1(i,:)-res(1:2)').^2);
        if ei < thresh
            consensus_set(cnt) = i;
            cnt = cnt + 1;
        end    
    end
%     disp(size(consensus_set));
    H = homography(x1(consensus_set,:),x2(consensus_set,:));
%     disp(H);
end

