%%  set some problem parameters
rng(2024);
m = 1000;
n = 1000;
r = 1;
rho = 0.5;
sigma = 1;

%% generate the underlying matrix, non-spiky
U0 = unifrnd(-0.5,0.5,m, r);
V0 = unifrnd(-0.5,0.5,n, r);
M0 = U0*V0';
M0 = M0/max(abs(M0(:)));
sratio = sqrt(m*n)*max(abs(M0(:)))/norm(M0, 'fro');


%% demo 1, probit model
rng(123)
% Probit model
f      = @(x) normcdf(x,0,sigma);
fprime = @(x) normpdf(x,0,sigma);
% generate the observed 1-bit matrix Y
Y = sign(f(M0)-rand(m,n));

% generate the index set of observations
omega = randsample(m*n, floor(rho*m*n));
ind_omega = zeros(m*n, 1);
ind_omega(omega) = 1;

%%%%%%%% try MMGN 
y = Y(:);
y(ind_omega==0) = 0; % code the unobserved entries as zero
Y = reshape(y, [m n]);
[U,S,V] = svd(Y);
U0 = U(:, 1:r)*sqrt(S(1:r, 1:r));
V0 = V(:, 1:r)*sqrt(S(1:r, 1:r));

%% MMGN with PCG, around 1 second
t0 = tic;
[Uhat, Vhat, relerr, relchange, obj, alphas, nBacktracks] = MMGN_probit(Y, ind_omega, sigma, r,U0, V0, M0,...
                                                                          'solver', 'PCG','alpha0', 3,'tol', 1e-6,'maxiters', 100);
toc(t0)
Mhat = Uhat*Vhat';
norm(Mhat-M0,'fro')^2/norm(M0,'fro')^2


%% data-driven approach to select r  ***** takes about 40 seconds ****
opts = [];
opts.rSeq = 1:5;
opts.maxiters = 1e2;
opts.tol = 1e-6; % 1e-6 for spiky
opts.alpha0 = 3;
opts.solver = 'PCG';
t0 = tic;
[U_mm, V_mm, rhat_mm, relerr_mm, out_mm] = MMGN_probit_auto(Y, ind_omega, sigma,M0, opts);
toc(t0);

relerr_mm


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% demo 2, logistic model
f       = @(x) (1 ./ (1 + exp(-x/sigma)));
% generate the observed 1-bit matrix Y
Y = sign(f(M0)-rand(m,n));

% generate the index set of observations
rng(2024)
omega = randsample(m*n, floor(rho*m*n));
ind_omega = zeros(m*n, 1);
ind_omega(omega) = 1;


% Initialization
y = Y(:);
y(ind_omega==0) = 0; % code the unobserved entries as zero
Y = reshape(y, [m n]);
[U,S,V] = svd(Y);
U0 = U(:, 1:r)*sqrt(S(1:r, 1:r));
V0 = V(:, 1:r)*sqrt(S(1:r, 1:r));


%% MMGN with PCG, around 1 second
t0 = tic;
[Uhat, Vhat, relerr, relchange, obj, alphas, nBacktracks] = MMGN_logist(Y, ind_omega, sigma, r,U0, V0, M0,...
                                                                          'solver', 'PCG','alpha0', 3,'tol', 1e-6,'maxiters', 100);
toc(t0)
Mhat = Uhat*Vhat';
norm(Mhat-M0,'fro')^2/norm(M0,'fro')^2


%% data-driven approach to select r  ***** takes about 45 seconds ****
opts = [];
opts.rSeq = 1:5;
opts.maxiters = 1e2;
opts.tol = 1e-6; % 1e-6 for spiky
opts.alpha0 = 3;
opts.solver = 'PCG';
t0 = tic;
[U_mm, V_mm, rhat_mm, relerr_mm, out_mm] = MMGN_logist_auto(Y, ind_omega, sigma,M0, opts);
toc(t0);

relerr_mm



