function [d] = Hellinger_dist(P, Q)
% This function computes the Hellinger distance between two distribution matrices P and Q

% --INPUTS-----------------------------------------------------------------------
% P: a distribution matrix, each entry is in [0, 1]
% Q: another distribution matrix, each entry is in [0, 1]

% --OUTPUTS-----------------------------------------------------------------------
% d: the Hellinger distance between P and Q

% Xiaoqian Liu
% Dec. 2023
%%
sum = 0;
[m, n] = size(P);
for i=1:m
    for j=1:n
        p = P(i, j);
        q = Q(i, j);
        h2 = (sqrt(p) - sqrt(q))^2 +(sqrt(1-p) - sqrt(1-q))^2;
        sum = sum + h2; 
    end
end

d = sum/(m*n);

end
