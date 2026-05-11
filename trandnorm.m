function X = trandnorm(mu, sigma, a, b)
    if ~(a < b)
        error('trandnorm:BadInterval', 'Require a < b.');
    end

    % Standardize bounds
    alpha = (a - mu) ./ sigma;
    beta  = (b - mu) ./ sigma;

    % Standard Normal CDF via erfc (toolbox-free)
    Phi  = @(z) 0.5 .* erfc(-z ./ sqrt(2));
    % Inverse CDF via erfcinv
    PhiI = @(p) -sqrt(2) .* erfcinv(2 .* p);

    % CDF at bounds, with semi-infinite support
    Pa = (isinf(alpha) & alpha < 0) .* 0 + (~isinf(alpha)) .* Phi(alpha);
    Pb = (isinf(beta)  & beta  > 0) .* 1 + (~isinf(beta))  .* Phi(beta);

    Pa = double(Pa); Pb = double(Pb);
    if ~(Pa < Pb)
        error('trandnorm:Degenerate', 'CDF interval collapsed (check a,b, mu,sigma).');
    end

    % Uniforms on the CDF interval and map back with inverse CDF
    U = rand(length(mu),1);
    P = Pa + U .* (Pb - Pa);

    % Clamp to (0,1) for numerical safety (extreme tails)
    tiny = realmin;  % ~2.2e-308
    P = max(min(P, 1 - tiny), tiny);

    Z = PhiI(P);            % standard normal, truncated
    X = mu + sigma .* Z;    % scale & shift
end
