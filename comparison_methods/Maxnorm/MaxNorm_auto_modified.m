function [Mhat_Max, rhat, err, outs] = MaxNorm_auto_modified(Y, ind_omega, f, fprime, M, opts)
% Max-norm 1-bit matrix completion with automatic rank selection.
% This version works even when M = [].
%
% Inputs
%   Y         : observed binary matrix; unobserved entries should be 0
%   ind_omega : observation mask, same size as Y (0/1)
%   f         : CDF link, e.g. @(x) normcdf(x,0,sigma)
%   fprime    : derivative of f, e.g. @(x) normpdf(x,0,sigma)
%   M         : true matrix for simulation only; use [] for real data
%   opts      : struct with fields:
%                 rSeq, maxiters, tol, stopping, seed, rate, alpha,
%                 tauSeq (optional)
%
% Outputs
%   Mhat_Max  : estimated matrix
%   rhat      : selected rank
%   err       : relative Frobenius error if M is supplied, else NaN
%   outs      : struct with CVloglik, CVrelerr, Mcell, final_obj, etc.

    if nargin < 6 || isempty(opts)
        opts = struct();
    end
    opts = setDefaults(opts);

    maxiters = opts.maxiters;
    tol      = opts.tol;
    stopping = opts.stopping;
    rSeq     = opts.rSeq;
    numR     = numel(rSeq);
    seed     = opts.seed;
    rate     = opts.rate;
    alpha    = opts.alpha;
    tauSeq   = opts.tauSeq;

    if isempty(alpha) || ~isscalar(alpha) || alpha <= 0
        error('For real data, set opts.alpha to a positive scalar.');
    end

    use_truth = ~isempty(M);

    outs = struct();
    outs.Mcell      = cell(numR,1);
    outs.CVrelerr   = nan(numR,1);
    outs.CVloglik   = nan(numR,1);
    outs.selected_r = nan(numR,1);

    [d1,d2] = size(Y);
    y = Y(:);

    rng(seed);
    omega = find(ind_omega);
    ntrain = ceil(rate * numel(omega));
    loc_train = randsample(omega, ntrain);
    loc_test  = setdiff(omega, loc_train);

    ind_train = zeros(d1*d2,1);
    ind_train(loc_train) = 1;
    ind_test = zeros(d1*d2,1);
    ind_test(loc_test) = 1;

    D = (1 + Y) / 2;
    dtest = D(ind_test > 0);

    rhat = rSeq(1);
    bestCV = -inf;
    bestInitU = [];
    bestInitV = [];

    for k = 1:numR
        r = rSeq(k);

        missing_entry = setdiff(1:d1*d2, loc_train);
        y_train = y;
        y_train(missing_entry) = 0;
        Y_train = reshape(y_train, d1, d2);

        [UU,S,VV] = svd(Y_train, 'econ');
        rr = min(r+1, size(S,1));
        U0 = UU(:,1:rr) * sqrt(S(1:rr,1:rr));
        V0 = VV(:,1:rr) * sqrt(S(1:rr,1:rr));

        [Mhat_tmp, ~, ~, tau_used, info] = Max_norm_modified( ...
            Y_train, ind_train, f, fprime, r, alpha, U0, V0, M, ...
            'maxiters', maxiters, 'tol', tol, 'stopping', stopping, 'tauSeq', tauSeq);

        [U,S2,V] = svd(Mhat_tmp, 'econ');
        rr2 = min(r, size(S2,1));
        Mhat_r = U(:,1:rr2) * S2(1:rr2,1:rr2) * V(:,1:rr2)';

        outs.Mcell{k} = Mhat_r;
        outs.selected_r(k) = r;
        outs.CVloglik(k) = -obj_1bit(dtest, Mhat_r(ind_test > 0), f);

        if use_truth
            outs.CVrelerr(k) = (norm(Mhat_r - M, 'fro') / norm(M, 'fro'))^2;
        end

        if outs.CVloglik(k) > bestCV
            bestCV = outs.CVloglik(k);
            rhat = r;
            rr3 = min(rhat+1, size(S2,1));
            bestInitU = U(:,1:rr3) * sqrt(S2(1:rr3,1:rr3));
            bestInitV = V(:,1:rr3) * sqrt(S2(1:rr3,1:rr3));
            outs.best_tau_cv = tau_used;
            outs.best_info_cv = info;
        end
    end

    % Refit using all observed entries
    missing_entry = setdiff(1:d1*d2, omega);
    y_all = y;
    y_all(missing_entry) = 0;
    Y_all = reshape(y_all, d1, d2);

    if isempty(bestInitU) || isempty(bestInitV)
        [UU,S,VV] = svd(Y_all, 'econ');
        rr = min(rhat+1, size(S,1));
        bestInitU = UU(:,1:rr) * sqrt(S(1:rr,1:rr));
        bestInitV = VV(:,1:rr) * sqrt(S(1:rr,1:rr));
    end

    [Mhat_tmp, relerr_seq, niterations, tau_used, info] = Max_norm_modified( ...
        Y_all, ind_omega, f, fprime, rhat, alpha, bestInitU, bestInitV, M, ...
        'maxiters', maxiters, 'tol', tol, 'stopping', stopping, 'tauSeq', tauSeq);

    [U,S,V] = svd(Mhat_tmp, 'econ');
    rr = min(rhat, size(S,1));
    Mhat_Max = U(:,1:rr) * S(1:rr,1:rr) * V(:,1:rr)';

    if use_truth
        err = (norm(Mhat_Max - M, 'fro') / norm(M, 'fro'))^2;
    else
        err = NaN;
    end

    outs.errSeq      = relerr_seq;
    outs.niterations = niterations;
    outs.final_tau   = tau_used;
    outs.final_obj   = info.best_obj;
    outs.final_relchg = info.best_relchg;
end

function opts = setDefaults(opts)
    if ~isfield(opts,'rSeq'),      opts.rSeq = 1:5; end
    if ~isfield(opts,'maxiters'),  opts.maxiters = 1000; end
    if ~isfield(opts,'tol'),       opts.tol = 1e-4; end
    if ~isfield(opts,'stopping'),  opts.stopping = 'objective'; end
    if ~isfield(opts,'seed'),      opts.seed = 2022; end
    if ~isfield(opts,'rate'),      opts.rate = 0.8; end
    if ~isfield(opts,'alpha'),     opts.alpha = []; end
    if ~isfield(opts,'tauSeq'),    opts.tauSeq = []; end
end