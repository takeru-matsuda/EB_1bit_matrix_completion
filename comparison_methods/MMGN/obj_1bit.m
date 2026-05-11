function obj = obj_1bit(d, m, f)
% This function computes the objective value/loss of the MMGN method for 
% 1-bit matrix completion.

% --INPUTS-----------------------------------------------------------------------
% d: the vector of observations in the Delta matrix in the objective function. 
%    Delta=(1+Y)/2, d only includes 0 or 1
% M: the vector of observations in the current estimate of the underlying matrix
% f: the CDF of the noise (probit or logistic)

% --OUTPUTS-----------------------------------------------------------------------
% obj: the objective value/loss

% Xiaoqian Liu
% Dec. 2023

%%
    m1 = m(d>0); % entries corresponding to d_ij = 1
    m0 = m(d==0); % entries corresponding to d_ij = 0
    
    obj = -sum(log(f(m1))) - sum(log(1 - f(m0)));
 
end