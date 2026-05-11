function [M_hat, relerr, niterations, best_tau, info] = Max_norm_modified( ...
    Y, ind_omega, f, fprime, r, alpha, U0, V0, M, varargin)
% Real-data-safe version of Max_norm.
% If M = [], step size is chosen by observed objective value.

    params = inputParser;
    params.addParameter('tol', 1e-4, @isscalar);
    params.addParameter('maxiters', 1e3, @(x) isscalar(x) && x > 0);
    params.addParameter('stopping', 'objective', @(x) ischar(x) || isstring(x));
    params.addParameter('tauSeq', [], @(x) isempty(x) || isvector(x));
    params.parse(varargin{:});

    tol = params.Results.tol;
    maxiters = params.Results.maxiters;
    stopping = params.Results.stopping;
    tauSeq = params.Results.tauSeq;

    [d1,d2] = size(Y);
    d = [d1,d2];

    y = Y(ind_omega > 0);
    d_obs = (1 + y) / 2;

    omega = find(ind_omega);
    s = length(omega);
    B = alpha * sqrt(r) + eps;

    if isempty(tauSeq)
        tauSeq = linspace(0.05, min(5, max(d1,d2)), 20);
    end
    T = numel(tauSeq);

    use_truth = ~isempty(M);
    if use_truth
        M0_Fro = max(sum(M(:).^2), eps);
    end

    relerr = nan(1,T);
    niterations = zeros(1,T);

    I = abs(Y);
    Yplus = (Y + I) / 2;
    Yminus = I - Yplus;

    % initialize safe fallback
    M_hat = U0 * V0';
    best_tau = tauSeq(1);
    best_obj = inf;
    best_relchg = inf;

    for count = 1:T
        tau = tauSeq(count);

        M_est = U0 * V0';
        obj_last = obj_1bit(d_obs, M_est(ind_omega > 0), f);
        U_last = U0;
        V_last = V0;
        last_relchg = inf;

        for epoch = 1:maxiters
            denom1 = max(f(M_est), 1e-12);
            denom2 = max(f(-M_est), 1e-12);

            G1 = (fprime(M_est) ./ denom1) .* Yplus / s;
            G2 = (fprime(M_est) ./ denom2) .* Yminus / s;
            G  = G2 - G1;

            U1 = U_last - tau * G  * V_last / sqrt(epoch);
            V1 = V_last - tau * G' * U_last / sqrt(epoch);

            U = proj(U1, B);
            V = proj(V1, B);
            M_new = U * V';

            if any(~isfinite(M_new(:)))
                break;
            end

            obj_new = obj_1bit(d_obs, M_new(ind_omega > 0), f);
            if ~isfinite(obj_new)
                break;
            end

            if strcmp(stopping,'objective')
                check = abs((obj_last - obj_new) / (abs(obj_last) + eps));
            else
                check = norm(M_est - M_new, 'fro')^2 / max(norm(M_est,'fro')^2, eps);
            end
            last_relchg = check;

            if epoch > 10 && check < tol
                M_est = M_new;
                obj_last = obj_new;
                break;
            end

            M_est = M_new;
            U_last = U;
            V_last = V;
            obj_last = obj_new;
        end

        niterations(count) = epoch;

        if use_truth
            relerr(count) = sum((M(:) - M_est(:)).^2) / M0_Fro;
        end

        % choose tau by observed objective, not truth
        final_obj = obj_1bit(d_obs, M_est(ind_omega > 0), f);
        if isfinite(final_obj) && final_obj < best_obj
            best_obj = final_obj;
            best_relchg = last_relchg;
            M_hat = M_est;
            best_tau = tau;
        end
    end

    info = struct();
    info.best_obj = best_obj;
    info.best_relchg = best_relchg;
end