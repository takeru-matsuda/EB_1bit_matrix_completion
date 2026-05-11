addpath(genpath('./comparison_methods'));

nmc = 100;
p = 1000;
q = 100;
r = 5;
max_rank = 10;
rho = 0.5;
sigma = 1;

MSE_MMGN = zeros(1,nmc);
MSE_Trace = zeros(1,nmc);
MSE_Max = zeros(1,nmc);
MSE_EB1 = zeros(1,nmc);
MSE_EB2 = zeros(1,nmc);
hellinger_MMGN = zeros(1,nmc);
hellinger_Trace = zeros(1,nmc);
hellinger_Max = zeros(1,nmc);
hellinger_EB1 = zeros(1,nmc);
hellinger_EB2 = zeros(1,nmc);
hellinger_unobs_MMGN = zeros(1,nmc);
hellinger_unobs_Trace = zeros(1,nmc);
hellinger_unobs_Max = zeros(1,nmc);
hellinger_unobs_EB1 = zeros(1,nmc);
hellinger_unobs_EB2 = zeros(1,nmc);
KL_MMGN = zeros(1,nmc);
KL_Trace = zeros(1,nmc);
KL_Max = zeros(1,nmc);
KL_EB1 = zeros(1,nmc);
KL_EB2 = zeros(1,nmc);
KL_unobs_MMGN = zeros(1,nmc);
KL_unobs_Trace = zeros(1,nmc);
KL_unobs_Max = zeros(1,nmc);
KL_unobs_EB1 = zeros(1,nmc);
KL_unobs_EB2 = zeros(1,nmc);
time_MMGN = zeros(1,nmc);
time_Trace = zeros(1,nmc);
time_Max = zeros(1,nmc);
time_EB1 = zeros(1,nmc);
time_EB2 = zeros(1,nmc);
for mc=1:nmc
    rng(mc);
    M = (2*rand(p,r)-1)*(2*rand(r,q)-1);
    Y0 = sign(M+sigma*randn(p,q));
    Y0(Y0==-1) = 0;
    omega = randsample(p*q, floor(rho*p*q));
    ind_omega = zeros(p*q,1);
    ind_omega(omega) = 1;
    y = Y0(:);
    y(ind_omega==0) = NaN;
    Y = reshape(y,[p,q]);
    obs = double(Y>=0);

    % MMGN
    t0 = tic;
    [pred_MMGN,Mhat_MMGN,rank_MMGN] = MMGN(Y,ind_omega,sigma,max_rank);
    time_MMGN(mc) = toc(t0);
    MSE_MMGN(mc) = norm(Mhat_MMGN-M,'fro')^2/norm(M,'fro')^2;
    hellinger_MMGN(mc) = norm(sqrt(pred_MMGN)-sqrt(normcdf(M,0,sigma)),'fro')^2+norm(sqrt(1-pred_MMGN)-sqrt(1-normcdf(M,0,sigma)),'fro')^2;
    hellinger_unobs_MMGN(mc) = mean(hellinger_bernoulli(normcdf(M(obs==0),0,sigma),pred_MMGN(obs==0)));
    KL_MMGN(mc) = mean(KL_bernoulli(normcdf(M(:),0,sigma),pred_MMGN(:)));
    KL_unobs_MMGN(mc) = mean(KL_bernoulli(normcdf(M(obs==0),0,sigma),pred_MMGN(obs==0)));

    % TraceNorm
    t0 = tic;
    [pred_Trace,Mhat_Trace,rank_Trace] = TraceNorm(Y,ind_omega,sigma,max_rank);
    time_Trace(mc) = toc(t0);
    MSE_Trace(mc) = norm(Mhat_Trace-M,'fro')^2/norm(M,'fro')^2;
    hellinger_Trace(mc) = norm(sqrt(pred_Trace)-sqrt(normcdf(M,0,sigma)),'fro')^2+norm(sqrt(1-pred_Trace)-sqrt(1-normcdf(M,0,sigma)),'fro')^2;
    hellinger_unobs_Trace(mc) = mean(hellinger_bernoulli(normcdf(M(obs==0),0,sigma),pred_Trace(obs==0)));
    KL_Trace(mc) = mean(KL_bernoulli(normcdf(M(:),0,sigma),pred_Trace(:)));
    KL_unobs_Trace(mc) = mean(KL_bernoulli(normcdf(M(obs==0),0,sigma),pred_Trace(obs==0)));

    % MaxNorm
    t0 = tic;
    [pred_Max,Mhat_Max,rank_Max] = MaxNorm(Y,ind_omega,sigma,max_rank);
    time_Max(mc) = toc(t0);
    MSE_Max(mc) = norm(Mhat_Max-M,'fro')^2/norm(M,'fro')^2;
    hellinger_Max(mc) = norm(sqrt(pred_Max)-sqrt(normcdf(M,0,sigma)),'fro')^2+norm(sqrt(1-pred_Max)-sqrt(1-normcdf(M,0,sigma)),'fro')^2;
    hellinger_unobs_Max(mc) = mean(hellinger_bernoulli(normcdf(M(obs==0),0,sigma),pred_Max(obs==0)));
    KL_Max(mc) = mean(KL_bernoulli(normcdf(M(:),0,sigma),pred_Max(:)));
    KL_unobs_Max(mc) = mean(KL_bernoulli(normcdf(M(obs==0),0,sigma),pred_Max(obs==0)));

    % EB1
    t0 = tic;
    [SIGMA1,pred_EB1,cnt1] = EB1(Y,100,20);
    time_EB1(mc) = toc(t0);
    Mhat_EB1 = norminv(pred_EB1,0,sigma);
    MSE_EB1(mc) = norm(Mhat_EB1-M,'fro')^2/norm(M,'fro')^2;
    hellinger_EB1(mc) = norm(sqrt(pred_EB1)-sqrt(normcdf(M,0,sigma)),'fro')^2+norm(sqrt(1-pred_EB1)-sqrt(1-normcdf(M,0,sigma)),'fro')^2;
    hellinger_unobs_EB1(mc) = mean(hellinger_bernoulli(normcdf(M(obs==0),0,sigma),pred_EB1(obs==0)));
    KL_EB1(mc) = mean(KL_bernoulli(normcdf(M(:),0,sigma),pred_EB1(:)));
    KL_unobs_EB1(mc) = mean(KL_bernoulli(normcdf(M(obs==0),0,sigma),pred_EB1(obs==0)));

    % EB2
    t0 = tic;
    [mu2,SIGMA2,pred_EB2,cnt2] = EB2(Y,100,20);
    time_EB2(mc) = toc(t0);
    Mhat_EB2 = norminv(pred_EB2,0,sigma);
    MSE_EB2(mc) = norm(Mhat_EB2-M,'fro')^2/norm(M,'fro')^2;
    hellinger_EB2(mc) = norm(sqrt(pred_EB2)-sqrt(normcdf(M,0,sigma)),'fro')^2+norm(sqrt(1-pred_EB2)-sqrt(1-normcdf(M,0,sigma)),'fro')^2;
    hellinger_unobs_EB2(mc) = mean(hellinger_bernoulli(normcdf(M(obs==0),0,sigma),pred_EB2(obs==0)));
    KL_EB2(mc) = mean(KL_bernoulli(normcdf(M(:),0,sigma),pred_EB2(:)));
    KL_unobs_EB2(mc) = mean(KL_bernoulli(normcdf(M(obs==0),0,sigma),pred_EB2(obs==0)));
end
