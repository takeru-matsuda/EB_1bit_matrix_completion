function [SIGMA,pred_EB,cnt] = EB1(Y,nmc,max_iter,EPS_SIGMA)
%
% Input:
%   Y:          binary (0-1) data matrix (p times q)
%               set unobserved entries to NaN
%   nmc:        Monte Carlo sample size
%   max_iter:   maximum number of EM iterations
%   EPS_SIGMA:  tolerance for SIGMA
%
% Output:
%   SIGMA:      estimate of SIGMA (q times q)
%   pred_EB:    prediction of Y (p times q)
%   cnt:        number of iterations until convergence (scalar)
%
    if nargin < 4
        EPS_SIGMA = 0;
    end
    burnin = 0;
    p = size(Y,1);
    q = size(Y,2);
    pred_EB = zeros(p,q);
    SIGMA = eye(q);
    for cnt=1:max_iter
        SIGMA_prev = SIGMA;
        covM = (SIGMA_prev+eye(q))\SIGMA_prev;
        covM = (covM+covM')/2;
        Z = 2*Y-1;
        M = zeros(p,q);%mvnrnd(zeros(1,q),SIGMA_prev,p);
        pred_EB = zeros(p,q);
        SIGMA = zeros(q,q);
        for mc=1:burnin+nmc
            Z(Y==1) = trandnorm(M(Y==1), 1, 0, inf);
            Z(Y==0) = trandnorm(M(Y==0), 1, -inf, 0);
            Z(isnan(Y)) = M(isnan(Y))+randn(nnz(isnan(Y)), 1);
            M = mvnrnd(Z*covM,covM);
            if mc > burnin
                pred_EB = pred_EB+normcdf(M)/nmc;
                SIGMA = SIGMA+M'*M/p/nmc;
            end
        end
        rel_SIGMA = norm(SIGMA-SIGMA_prev,'fro')/norm(SIGMA_prev,'fro');
        [cnt,rel_SIGMA]
        if rel_SIGMA < EPS_SIGMA
            disp('SIGMA update saturated');
            break;
        end
    end
    pred_EB(pred_EB<0) = 0;
    pred_EB(pred_EB>1) = 1;
end
