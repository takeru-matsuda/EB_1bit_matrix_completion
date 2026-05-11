function [U, V, relerr, relchange, obj, alphas, ...
          nBacktracks] = MMLM_probit(Y, ind_omega, sigma, R, lambda, U0, V0, M, varargin)
% This function implements MMLM for 1-bit matrix completion under the probit 
% noise model, given a specific rank R

% --INPUTS-----------------------------------------------------------------------
% Y: the observed binary data matrix, unobserved entries are coded as zero 
% ind_omega: the indicator vector of observations (column-major vectorization)
% sigma: the noise level, assumed to be known
% R: the target rank 
% lambda: the regularization parameter for the ridge penalty in MMLM
% U0: the initial value of the factor matrix U
% V0: the initial value of the factor matrix V
% M: the true underlying matrix (for performance tracking)
% varargin: additional parameters, including the following
%       tol: the tolerance for the relative change in the objective value or
%            norm of the estimate. It is provided for early stopping with
%            a default value of 1e-4.
%       maxiters: the maximum number of iterations, the default is 100.
%       stopping: the criterion used for early stopping
%                 'objective': early stop the algorithm when the relative change 
%                             in the objective value is less than tol (default)
%                 'estimate': early stop the algorithm when the relative change 
%                             in the estimate (squared F-norm) is less than tol
%       alpha0: the initial stepsize of GN for backtracking linesearch (default is 1)

% --OUTPUTS-----------------------------------------------------------------------
% U: the factor matrix U, m-by-R, Mhat = U*V'
% V: the factor matrix V, n-by-R, Mhat = U*V'
% relerr: the sequence of the relative error, defined as 
%            norm(Mhat-M,'fro')^2/norm(M,'fro')^2,
%         given the true matrix M (only for simulation)
% relchange: the sequence of the ralative change in the objective/estimate
%            (based on the stoping criterion)
% obj: the sequence of the objective value at each iteration
% alphas: the sequence of the stepsize for the GN step at each iteration
% nBacktracks: the sequence of the number of backtracking at each iteration

% Xiaoqian Liu
% Dec. 2023

%% Set algorithm parameters from input or by using defaults.
params = inputParser;
params.addParameter('tol', 1e-4, @isscalar);
params.addParameter('maxiters', 1e2, @(x) isscalar(x) & x > 0);
params.addParameter('stopping', 'objective', @(x) ischar(x)||isstring(x));
params.addParameter('alpha0', 1, @isscalar); %  the initial stepsize of GN
params.parse(varargin{:});

%% Copy from params object.
tol = params.Results.tol;
maxiters = params.Results.maxiters;
stopping = params.Results.stopping;
alpha0 = params.Results.alpha0;

%%
[m, n] = size(Y);
y = Y(ind_omega>0); % vector of binary observations 
d = (1+y)/2;  

f      = @(x) normcdf(x,0,sigma);
fprime = @(x) normpdf(x,0,sigma);

%% Initialization
rng(2022);
Mhat_last = U0*V0';
mhat_last = Mhat_last(ind_omega>0);
obj_last = obj_1bit(d, mhat_last, f);

% parameters for backtracking
beta = 0.5;
gamma = 0.9;
L = 1/sigma^2;

%% allocate memeroy for outputs
obj = nan(maxiters, 1);
relerr = nan(maxiters, 1);
relchange = nan(maxiters, 1);
dec = nan(maxiters, 1);  %decrease in the objective value
alphas = nan(maxiters, 1); % stepsize
nBacktracks = nan(maxiters, 1);  % number of backtracking iteration

%%
for iter = 1:maxiters
    %=======line 3:
    %=======construct x
    x = (1/L)*y.*fprime(mhat_last)./f(y.*mhat_last);
    %======line 4:
    %====== construct J in the LS problem (25)
    P1 = kron(V0, speye(m));
    P2 = kron(speye(n), U0);
    P1(ind_omega==0, :)=[];
    P2(ind_omega==0, :)=[];
    Phi = [P1 P2];
    
    %======line 5:
    %====== solve the LS problem
    %[eta, flag] = lsqr(Phi, x, 1e-10, 1e3);
    fit = fitrlinear(Phi, x, 'Lambda', lambda, 'Regularization', 'ridge');
    eta = fit.Beta;
    
    %===== get du and dv
    du = eta(1:m*R);
    dv = eta(m*R+1:end);
    
    %======line 6:
    %====== recover dU and dV
    dU = reshape(du, [m R]);
    dV = reshape(dv, [R n])';
    
    %====line 7:
    %====update U and V with stepsize alpha chosen by backtracking
    alpha = alpha0;
    U1 = U0 + alpha*dU;
    V1 = V0 + alpha*dV;
    
    %=== compute dUV+UdV in the backtracking condition
    G = dU*V0' + U0*dV';
    g = G(ind_omega>0);
    %===
    dec(iter) = -L*sum(x.*g);  % see eqn(B.9) and the defination of x

    %=== start backtracking
    nBacktrack = 0;
    Mhat = U1*V1';
    mhat = Mhat(ind_omega>0);
    obj_new = obj_1bit(d, mhat, f);
    while (obj_new > obj_last + beta*alpha*dec(iter))  && (nBacktrack <=100)
        alpha = gamma*alpha;
        U1 = U0 + alpha*dU;
        V1 = V0 + alpha*dV;
        Mhat = U1*V1';
        mhat = Mhat(ind_omega>0);
        obj_new = obj_1bit(d, mhat, f);
        nBacktrack = nBacktrack + 1;
    end
    
    %=== save some output
    obj(iter) = obj_new; % the objective value/ loss
    alphas(iter) = alpha;  % stepsize
    nBacktracks(iter) = nBacktrack;  % number of backtracking iteration
    
    % squared relative error
    relerr(iter) = norm(Mhat-M,'fro')^2/norm(M,'fro')^2;
    
    % relative change according to stopping criteron
    if strcmp(stopping,'objective')
        relchange(iter) = abs( (obj_last-obj_new) / (obj_last+eps) );
        %  relchange(iter) = abs( (obj_last-obj_new) / (m*n) );
    else
        relchange(iter) = norm(Mhat-Mhat_last,'fro')^2/norm(Mhat_last,'fro')^2;
    end
    
    %=== early stooping
    if relchange(iter) < tol
        break;
    end

    
    %=== update for the next iteration
    U0 = U1;
    V0 = V1;
    mhat_last = Mhat(ind_omega>0);
    obj_last = obj(iter);
end

% outputs
U = U1; % output
V = V1;
obj = obj(1:iter);
relerr = relerr(1:iter);
relchange = relchange(1:iter);
alphas = alphas(1:iter);
nBacktracks = nBacktracks(1:iter);
end