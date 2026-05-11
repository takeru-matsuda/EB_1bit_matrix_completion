function [mu,SIGMA,pred_EB,cnt] = EB2(Y,nmc,max_iter,EPS_mu,EPS_SIGMA)
%
% Input:
%   Y:          binary (0-1) data matrix (p times q)
%               set unobserved entries to NaN
%   nmc:        Monte Carlo sample size
%   max_iter:   maximum number of EM iterations
%   EPS_mu:     tolerance for mu
%   EPS_SIGMA:  tolerance for SIGMA
%
% Output:
%   mu:         estimate of mu (q times 1)
%   SIGMA:      estimate of SIGMA (q times q)
%   pred_EB:    prediction of Y (p times q)
%   cnt:        number of iterations until convergence (scalar)
%
    if nargin < 4
        EPS_mu = 0;
    end
    if nargin < 5
        EPS_SIGMA = 0;
    end
    burnin = 0;
    p = size(Y,1);
    q = size(Y,2);
    pred_EB = zeros(p,q);
    mu = zeros(q,1);
    SIGMA = eye(q);
    for cnt=1:max_iter
        mu_prev = mean(mu,2);
        SIGMA_prev = mean(SIGMA,3);
        tmp = SIGMA_prev\mu_prev;
        covM = (SIGMA_prev+eye(q))\SIGMA_prev;
        covM = (covM+covM')/2;
        Z = 2*Y-1;
        M = zeros(p,q);%mvnrnd(zeros(1,q),SIGMA_prev,p);
        pred_EB = zeros(p,q);
        mom1 = zeros(q,1);
        mom2 = zeros(q,q);
        for mc=1:burnin+nmc
            Z(Y==1) = trandnorm(M(Y==1), 1, 0, inf);
            Z(Y==0) = trandnorm(M(Y==0), 1, -inf, 0);
            Z(isnan(Y)) = M(isnan(Y))+randn(nnz(isnan(Y)), 1);
            M = mvnrnd((ones(p,1)*tmp'+Z)*covM,covM);
            if mc > burnin
                pred_EB = pred_EB+normcdf(M)/nmc;
                mom1 = mom1+sum(M)'/p/nmc;
                mom2 = mom2+M'*M/p/nmc;
            end
        end
        mu = mom1;
        SIGMA = mom2-mom1*mom1';
        rel_mu = norm(mu-mu_prev)/norm(mu_prev);
        rel_SIGMA = norm(SIGMA-SIGMA_prev,'fro')/norm(SIGMA_prev,'fro');
        [cnt,rel_mu,rel_SIGMA]
        if rel_mu < EPS_mu
            disp('mu update saturated');
            break;
        end
        if rel_SIGMA < EPS_SIGMA
            disp('SIGMA update saturated');
            break;
        end
    end
    pred_EB(pred_EB<0) = 0;
    pred_EB(pred_EB>1) = 1;
end
