function [U, V, rhat, err, outs] = MMGN_logist_auto(Y, ind_omega, sigma, M, opts)
% This function implements MMGN for 1-bit matrix completion under the logistic
% noise model with a data driven approach for selecting the rank constraint
% parameter r.

% --INPUTS-----------------------------------------------------------------------
% Y: observed binary data matrix, unobserved entries are coded as zero
% ind_omega: the indicator vector of observations (column-major vectorization)
% sigma: the noise level, assumed to be known 
% M: the true underlying matrix (for performance tracking)
% opts: an optional structure containing various options that users may choose to set.
%       opts includes the following parameters
        % rSeq: a grid of values for the rank parameter (default: 1 to 5)
        % maxiters: the maximum number of iterations (default: 100)
        % tol: a tolerance for early stopping (default: 1e-4)
        % solver: which solver to use for the LS problem
        % stopping: the criteron used for early stopping
        %          'objective': early stop when the relative change in the
        %                       objective is less than tol (default)
        %          'estimate': early stop when the relative change in the 
        %                      estimate (squared F-norm) is less than tol
        % rate: the percentage of data to be used as the training set (default: 0.8)
        % seed : set seed for reproducibility (default: 2022)
        % alpha0: the initial stepszie of GN for backtracking linesearch (default: 1)

        
% --OUTPUTS-----------------------------------------------------------------------
% U: the factor matrix U, Mhat = U*V'
% V: the factor matrix V, Mhat = U*V'
% rhat: the optimal rank r, equal to the number of columns of U and V
% err: norm(Mhat-M,'fro')^2/norm(M,'fro')^2 for the final Mhat,
%      given the true matrix M (only for simulation)
% outs: An output structure containing the following 
    % errSeq: the sequence of norm(Mhat-M,'fro')^2/norm(M,'fro')^2 when 
    %         estimating the final Mhat with the selected r;
    % relchange: the sequence of the ralative change in the estimate/objective
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
    
    % for MMGN
    maxiters = opts.maxiters;
    tol = opts.tol;
    solver = opts.solver;
    stopping = opts.stopping;
    alpha0 = opts.alpha0;
    % for selecting r
    rSeq = opts.rSeq;
    numR = length(rSeq);
    seed  = opts.seed;
    rate = opts.rate;

    %%
    % Logistic model
    f       = @(x) (1 ./ (1 + exp(-x/sigma)));
    %fprime  = @(x) ((1/sigma)*exp(x/sigma) ./ (1 + exp(x/sigma)).^2);
    [m, n] = size(Y);

%%  Set up the outs structure
    outs = [];
    outs.errSeq = nan(maxiters, 1); 
    outs.relchange = nan(maxiters, 1); 
    outs.obj = nan(maxiters, 1); 
    
    outs.Mcell = cell(numR, 1);
    outs.CVrelerr = nan(numR, 1);
    outs.CVloglik = nan(numR, 1);
    
%% Generate the training data set 
   rng(seed);
   omega = find(ind_omega);
   loc_train = randsample(omega, ceil(rate*length(omega)));
   loc_test =  setdiff(omega, loc_train);
   ind_train = zeros(m*n, 1);
   ind_train(loc_train)=1;
   ind_test = zeros(m*n, 1);
   ind_test(loc_test)=1;
   
%% for initialization
    [UU,S,VV] = svd(Y);
    D = (1+Y)/2;
    d = D(ind_test>0);
    
    rhat = 1;
    mll = -inf;
    U00 = UU(:, 1)*sqrt(S(1, 1));
    V00 = VV(:, 1)*sqrt(S(1, 1));  
%% main loop to select r
    for k = 1:numR
        % set the estimated r in this loop
        r = rSeq(k); 
        % initialization according to r
        U0 = UU(:, 1:r)*sqrt(S(1:r, 1:r));
        V0 = VV(:, 1:r)*sqrt(S(1:r, 1:r));

        [U, V, relerr] = MMGN_logist(Y, ind_train, sigma, r, U0, V0, M, 'maxiters',maxiters, 'tol',tol,...
                                                   'solver',solver, 'stopping',stopping, 'alpha0', alpha0);
        
        Mhat = U*V';
        outs.Mcell{k,1} = Mhat;
        outs.CVrelerr(k, 1) = relerr(end); % this is the rel error on the whole matrix
        
        % compute the log-likelihood on the testing set
        outs.CVloglik(k, 1) = -obj_1bit(d, Mhat(ind_test>0), f);
         
        % for the overall
        if outs.CVloglik(k, 1)>mll
            rhat = r;
            mll = outs.CVloglik(k, 1);
            U00 = U;
            V00 = V;
        end
    end
    %% Now use rhat to reestimate the model
    [U, V, relerr, relchange, obj] = MMGN_logist(Y, ind_omega, sigma, rhat, U00, V00, M, 'maxiters',maxiters, 'tol',tol,...
                                                             'solver',solver, 'stopping',stopping, 'alpha0', alpha0);
    
    % save the outputs
    err = relerr(end);
    outs.errSeq = relerr; 
    outs.relchange = relchange; 
    outs.obj = obj; 
end
%%

function opts = setDefaults(opts)

        %  rSeq
        if ~isfield(opts,'rSeq')
            opts.rSeq = 1:5;
        end
         %  maxiters
        if ~isfield(opts,'maxiters')
            opts.maxiters = 100;
        end
        
        
        %  tol
        if ~isfield(opts,'tol')
            opts.tol = 1e-4;
        end
  
         %  solver
        if ~isfield(opts,'solver')
            opts.solver = 'PCG';
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
        

        if ~isfield(opts,'alpha0')
            opts.alpha0 = 1;
        end
end
