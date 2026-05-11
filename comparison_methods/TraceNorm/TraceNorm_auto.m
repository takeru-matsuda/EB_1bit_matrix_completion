function [Mhat_1Bit, rhat, err, outs] = TraceNorm_auto(Y, ind_omega, f, fprime, rSeq, rate, seed,  M, options)
% This function implementss the TraceNorm method in Davenport's paper
% for 1-bit matrix completion with a data-driven approach for selecting 
% the rank constraint parameter r.

% --INPUTS-----------------------------------------------------------------------
% Y: observed binary data matrix, unobserved entries are coded as zero
% ind_omega: the indicator vector of observations (column-major vectorization)
% M: the true underlying matrix (for performance tracking)
% f: the CDF of the noise
% fprime: the derivative of f
% rSeq: a grid of values for the estimated rank (default: 1 to 5)
% rate: the percentage of data to be used as the training set (default: 0.8)
% seed: set seed to genearte reproduciable outputs
% options: Options for spgSolver

        
% --OUTPUTS-----------------------------------------------------------------------
% U: the factor matrix U, Mhat = U*V'
% V: the factor matrix V, Mhat = U*V'
% rhat: the optimal rank r, equal to the number of columns of U and V
% err: norm(Mhat-M,'fro')^2/norm(M,'fro')^2 for the final Mhat,
%      given the true matrix M (only for simulation)
% outs: An output structure containing the following    
    % Mcell: Estimated M matrix (Mhat) for each r
    % CVrelerr: relative error of Mhat w.r.t. M for each r 
    % CVloglik: log-likelihood on the testing set for each r

% Xiaoqian Liu
% Dec. 2023

%% 
alpha   = max(abs(M(:)));  % oracle value of alpha for the TraceNorm method
numR = length(rSeq);

%%  Set up the outs structure
outs = [];

outs.Mcell = cell(numR, 1);
outs.CVrelerr = nan(numR, 1);
outs.CVloglik = nan(numR, 1);

%% Generate the training data set

[d1, d2] = size(Y);
y = Y(:);

 %% Generate the training data set
rng(seed);
omega = find(ind_omega);
loc_train = randsample(omega, ceil(rate*length(omega)));
loc_test =  setdiff(omega, loc_train);
ind_train = zeros(d1*d2, 1);
ind_train(loc_train)=1;
ind_test = zeros(d1*d2, 1);
ind_test(loc_test)=1;

%% for initialization
[UU,S,VV] = svd(Y);
D = (1+Y)/2;
d = D(ind_test>0);

rhat = 1;
mll = -inf;
U00 = UU(:, 1)*sqrt(S(1, 1));
V00 = VV(:, 1)*sqrt(S(1, 1));
XX0 = U00*V00';
%% main loop to estimate r
for k = 1:numR
    % set the estimated r in this loop
    r = rSeq(k);
    % initialization according to r
    U0 = UU(:, 1:r)*sqrt(S(1:r, 1:r));
    V0 = VV(:, 1:r)*sqrt(S(1:r, 1:r));
    X0 = U0*V0';
       
    % some settings for the spg solver
    funObj  = @(x) logObjectiveGeneral(x, y, loc_train , f, fprime);
    
    radius  = alpha * sqrt(d1*d2*r);
    
    % Use nuclear-norm constraint only to estimate M
    funProj = @(x,projTol,projData) projNucnorm(x,d1,d2,radius,projTol,projData);
    
    [Mhat,~] = spgSolver(funObj, funProj, X0(:), options);
    
    % trunct SVD to guarantee rank r
    Mhat = reshape(Mhat,d1,d2);
    [U,S,V] = svd(Mhat);
    Mhat = U(:,1:r)*S(1:r,1:r)*V(:,1:r)';
    
    outs.Mcell{k,1} = Mhat;
    outs.CVrelerr(k, 1) = (norm(Mhat-M,'fro')/norm(M,'fro'))^2; % this is the rel error on the whole matrix
    
    % compute the predictione error on the testing set
    outs.CVloglik(k, 1) = -obj_1bit(d, Mhat(ind_test>0), f);
    
    % for the overall
    if outs.CVloglik(k, 1)>mll
        rhat = r;
        mll = outs.CVloglik(k, 1);
        XX0 = Mhat;
    end
end
%% Now use rhat to reestimate the model
funObj  = @(x) logObjectiveGeneral(x, y, omega, f, fprime);

radius  = alpha * sqrt(d1*d2*rhat);

% Use nuclear-norm constraint only to estimate M
funProj = @(x,projTol,projData) projNucnorm(x,d1,d2,radius,projTol,projData);


[Mhat,~] = spgSolver(funObj, funProj, XX0(:), options);
Mhat = reshape(Mhat,d1,d2);
[U,S,V] = svd(Mhat);
Mhat_1Bit = U(:,1:rhat)*S(1:rhat,1:rhat)*V(:,1:rhat)'; % truncated SVD

% save the outputs
err = (norm(Mhat_1Bit-M,'fro')/norm(M,'fro'))^2;

end


