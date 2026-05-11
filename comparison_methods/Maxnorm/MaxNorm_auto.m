function [Mhat_Max, rhat, err, outs] = MaxNorm_auto(Y, ind_omega, f, fprime, M, opts)
% This function implements the MaxNorm method for 1-bit matrix completion 
% from Cai and Zhou with a data-driven approach for selecting the rank 
% constraint parameter r.

% --INPUTS-----------------------------------------------------------------------
% Y: observed binary data matrix, unobserved entries are coded as zero
% ind_omega: the indicator vector of observations (column-major vectorization)
% f: CDF of the noise
% fprime: the gradient of f
% M: the true underlying matrix (for performance tracking)
% opts: An optional struct containing various options that users may choose to set.
%       opts includes the following parameters
        % rSeq: a grid of values for the rank parameter (default: 1 to 5)
        % maxiters: the maximum number of iterations (default: 1000)
        % tol: a tolerance for early stopping (default: 1e-4)
        % stopping: the criterion used for early stopping
%               'objective': early stop the algorithm when the relative change 
%                            in the objective value is less than tol (default)
%               'estimate': early stop the algorithm when the relative change 
%                           in the estimate (squared F-norm) is less than tol
        % rate: the percentage of data to be used as the training set (default: 0.8)
        % seed : set seed for reproducibility (default: 2022)


% --OUTPUTS-----------------------------------------------------------------------
% Mhat_Max: the estimated underlying matrix 
% rhat: the optimal rank r
% err: norm(Mhat-M,'fro')^2/norm(M,'fro')^2 for the final Mhat
%      given the true matrix M (only for simulation)
% outs: an output structure containing the following
        % errSeq: the sequence of norm(Mhat-M,'fro')^2/norm(M,'fro')^2 when
        %         estimating the final Mhat with the selected r;
        % relchange: the sequence of norm(Mhat-Mhat_last,'fro')^2/norm(Mhat_last,'fro')^2
        %         when estimating the final Mhat with the selected r;
        % obj: the sequence of objective values at each iteration when
        %         estimating the final Mhat with the selected r;

        % Mcell: Estimated M matrix (Mhat) for each r
        % CVrelerr: relative error of Mhat w.r.t. M for each r
        % CVloglik: log-likelihood on the testing set for each r

% Xiaoqian Liu
% Dec. 2023

%% Set algorithm parameters from input or by using defaults.
% Make sure 'opts' struct exists, and is filled with all the options
if ~exist('opts','var')
    opts = [];
end
opts = setDefaults(opts);

% for Max_norm
maxiters = opts.maxiters;
tol = opts.tol;
stopping = opts.stopping;

% for selecting r
rSeq = opts.rSeq;
numR = length(rSeq);
seed  = opts.seed;
rate = opts.rate;
          
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
D = (1+Y)/2;
d = D(ind_test>0);

rhat = 1;
mll = -inf;
    
%% main loop to estimate r
for k = 1:numR
    % set the estimated r in this loop
    r = rSeq(k);
    
    missing_entry = setdiff(1:d1*d2,loc_train);
    y_maxnorm = y;
    y_maxnorm(missing_entry) = 0;
    Y_maxnorm = reshape(y_maxnorm,d1,d2);
    
    % initialization according to r
    %     rr = r+1;
    %     U0 = UU(:, 1:rr)*sqrt(S(1:rr, 1:rr));
    %     V0 = VV(:, 1:rr)*sqrt(S(1:rr, 1:rr));
    [UU,S,VV] = svd(Y_maxnorm);
    rr = r+1;
    U0 = UU(:, 1:rr)*sqrt(S(1:rr, 1:rr));
    V0 = VV(:, 1:rr)*sqrt(S(1:rr, 1:rr));

    [Mhat_max,relerr_max] = Max_norm(Y_maxnorm, ind_train,f,fprime,...
                                      r, max(abs(M(:))), U0, V0, M, 'maxiters',maxiters, 'tol', tol, 'stopping', stopping);
    
    
    % Project onto actual rank if known
    [U,S,V] = svd(Mhat_max);
    Mhat_Max = U(:,1:r)*S(1:r,1:r)*V(:,1:r)';
    
    outs.Mcell{k,1} = Mhat_Max;
    outs.CVrelerr(k, 1) = (norm(Mhat_Max-M,'fro')/norm(M,'fro'))^2; % this is the rel error on the whole matrix
    
    % compute the log-likelihood on the testing set
    outs.CVloglik(k, 1) = -obj_1bit(d, Mhat_Max(ind_test>0), f);
    
    % for the overall
    if outs.CVloglik(k, 1)>mll
        rhat = r;
        mll = outs.CVloglik(k, 1);
        rr = rhat+1;
        U00 = U(:,1:rr)*sqrt(S(1:rr,1:rr));
        V00 = V(:,1:rr)*sqrt(S(1:rr,1:rr));

    end
        
end

%% Now use rhat to reestimate the model
missing_entry = setdiff(1:d1*d2,omega);
y_maxnorm = y;
y_maxnorm(missing_entry) = 0;
Y_maxnorm = reshape(y_maxnorm,d1,d2);


[Mhat_max,relerr_max] =  Max_norm(Y_maxnorm, ind_omega,f,fprime,...
                                      rhat, max(abs(M(:))), U00, V00, M, 'maxiters',maxiters, 'tol', tol, 'stopping', stopping);

[U,S,V] = svd(Mhat_max);
Mhat_Max = U(:,1:rhat)*S(1:rhat,1:rhat)*V(:,1:rhat)';


% same the outputs
err = (norm(Mhat_Max-M,'fro')/norm(M,'fro'))^2;

end
%%

function opts = setDefaults(opts)

        %  rSeq
        if ~isfield(opts,'rSeq')
            opts.rSeq = 1:5;
        end
         %  maxiters
        if ~isfield(opts,'maxiters')
            opts.maxiters = 1000;
        end
        
        %  tol
        if ~isfield(opts,'tol')
            opts.tol = 1e-4;
        end
  
        %  stopping
        if ~isfield(opts,'stopping')
            opts.stopping = 'objective';
        end
        
        
         % seed     
        if ~isfield(opts,'seed')
            opts.seed = 2022;
        end
        
         % rate     
        if ~isfield(opts,'rate')
            opts.rate = 0.8;
        end
        
end


