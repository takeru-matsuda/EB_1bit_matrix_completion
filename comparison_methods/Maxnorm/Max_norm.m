function [M_hat, relerr, niterations, tau] = Max_norm(Y,ind_omega,f,fprime,r, alpha, U0, V0, M, varargin)
% This function implements the MaxNorm methods from Cai and Zhou.

% --INPUTS-----------------------------------------------------------------------
% Y: observed binary data matrix, unobserved entries are coded as zero
% ind_omega: the indicator vector of observations (column-major vectorization)
% f: CDF of the noise
% fprime: derivative of f
% r: the target rank
% alpha: the upper bound of infinity norm
% U0: the initial value of the factor matrix U
% V0: the initial value of the factor matrix V
% M: the true underlying matrix (for performance tracking)
% varargin: additional parameters, including the following
%       tol: the tolerance for the relative change in the objective value or
%            norm of the estimate. It is provided for early stopping with
%            a default value of 1e-4.
%       maxiters: the maximum number of iterations, default is 1000.
%       stopping: the criterion used for early stopping
%                 'objective': early stop the algorithm when the relative change 
%                             in the objective value is less than tol (default)
%                 'estimate': early stop the algorithm when the relative change 
%                             in the estimate (squared F-norm) is less than tol

% --OUTPUTS-----------------------------------------------------------------------
% Mhat: the estimated matrix
% relerr: norm(Mhat-M,'fro')^2/norm(M,'fro')^2 for the final Mhat
% niterations: number of each iterations for each stepsize
% tau: sequence of stepsizes used in the procedure

% Xiaoqian Liu
% Dec. 2023
%% Set algorithm parameters from input or by using defaults.
params = inputParser;
params.addParameter('tol', 1e-4, @isscalar);
params.addParameter('maxiters', 1e3, @(x) isscalar(x) & x > 0);
params.addParameter('stopping', 'objective', @(x) ischar(x)||isstring(x));
params.parse(varargin{:});

%% Copy from params object.
tol = params.Results.tol;
maxiters = params.Results.maxiters;
stopping = params.Results.stopping;

%%
[d1, d2] = size(Y);
y = Y(ind_omega>0); % vector of binary observations 
d = (1+y)/2; 
M0_Fro = sum(M(:).^2);


omega = find(ind_omega);
s = length(omega);
B = alpha*sqrt(r)+eps; 
err = inf;

% set stepsize.Change to choose 50 values from 1 to the dimension.
% works well in practice
tau = linspace(1, max(d1, d2), 50);
T = length(tau);

relerr = zeros(1,T);
niterations = zeros(1,T);

I = abs(Y);
Yplus = (Y + I)/2;
Yminus = I - Yplus;


 
%%
for count=1:T
    % Initialization
    M_est = U0 * V0';
    obj_last = obj_1bit(d, M_est(ind_omega>0), f);
    U_last = U0;
    V_last = V0;
    %%%%%%%%%%%%%% Iterations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for epoch = 1:maxiters
        G1 = (fprime(M_est)./f(M_est)).*Yplus/(s) ;
        G2 = (fprime(M_est)./f(-M_est)).*Yminus/(s) ;
        G =  G2 - G1;   % calculate the gradient ???????? Yes
        
        
        U1 = U_last - tau(count) * G * V_last/sqrt(epoch);   % update U
        V1 = V_last - tau(count) * G'* U_last/sqrt(epoch);   % update V
        U = proj(U1,B);  
        V = proj(V1,B);
        
        M_new = U * V';
        obj_new = obj_1bit(d, M_new(ind_omega>0), f);
         
        if count>10  %% only early stop when iter>10
            if strcmp(stopping,'objective')
                check = abs( (obj_last-obj_new) / (obj_last+eps) );
            else
                check =  norm(M_est-M_new,'fro')^2/norm(M_est,'fro')^2;
            end
            
            if check < tol
                M_est = M_new;
                break;
            end        
        end
        
        % for next iteration
         M_est = M_new;
         U_last = U;
         V_last = V;
         obj_last = obj_new;
    end
    
    Diff = (M-M_est).^2;
    diff = sum(Diff(:))/M0_Fro;
    relerr(1,count) = diff;
    niterations(1,count) = epoch;
    
    if diff < err
        M_hat = M_est;
        err = diff;
    end
end

end




