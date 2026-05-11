addpath(genpath('./comparison_methods'));

nmc = 100;
p = 1000;
q = 100;
rs = 1:2:15;
max_rank = 10;
rho = 0.5;
sigma = 1;
s = 1;

MSE_MMGN = zeros(length(rs),nmc);
MSE_Trace = zeros(length(rs),nmc);
MSE_Max = zeros(length(rs),nmc);
MSE_LMF = zeros(length(rs),nmc);
MSE_EB = zeros(length(rs),nmc);
MSE_EB0 = zeros(length(rs),nmc);
hellinger_MMGN = zeros(length(rs),nmc);
hellinger_Trace = zeros(length(rs),nmc);
hellinger_Max = zeros(length(rs),nmc);
hellinger_LMF = zeros(length(rs),nmc);
hellinger_EB = zeros(length(rs),nmc);
hellinger_EB0 = zeros(length(rs),nmc);
hellinger_unobs_MMGN = zeros(length(rs),nmc);
hellinger_unobs_Trace = zeros(length(rs),nmc);
hellinger_unobs_Max = zeros(length(rs),nmc);
hellinger_unobs_LMF = zeros(length(rs),nmc);
hellinger_unobs_EB = zeros(length(rs),nmc);
hellinger_unobs_EB0 = zeros(length(rs),nmc);
KL_MMGN = zeros(length(rs),nmc);
KL_Trace = zeros(length(rs),nmc);
KL_Max = zeros(length(rs),nmc);
KL_LMF = zeros(length(rs),nmc);
KL_EB = zeros(length(rs),nmc);
KL_EB0 = zeros(length(rs),nmc);
KL_unobs_MMGN = zeros(length(rs),nmc);
KL_unobs_Trace = zeros(length(rs),nmc);
KL_unobs_Max = zeros(length(rs),nmc);
KL_unobs_LMF = zeros(length(rs),nmc);
KL_unobs_EB = zeros(length(rs),nmc);
KL_unobs_EB0 = zeros(length(rs),nmc);
time_MMGN = zeros(length(rs),nmc);
time_Trace = zeros(length(rs),nmc);
time_Max = zeros(length(rs),nmc);
time_LMF = zeros(length(rs),nmc);
time_EB = zeros(length(rs),nmc);
time_EB0 = zeros(length(rs),nmc);
for ir=1:length(rs)
    r = rs(ir);
    for mc=1:nmc
        rng(mc);
        M = s*(2*rand(p,r)-1)*(2*rand(r,q)-1);
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
        time_MMGN(ir,mc) = toc(t0);
        MSE_MMGN(ir,mc) = norm(Mhat_MMGN-M,'fro')^2/norm(M,'fro')^2;
        hellinger_MMGN(ir,mc) = norm(sqrt(pred_MMGN)-sqrt(normcdf(M,0,sigma)),'fro')^2+norm(sqrt(1-pred_MMGN)-sqrt(1-normcdf(M,0,sigma)),'fro')^2;
        hellinger_unobs_MMGN(ir,mc) = mean(hellinger_bernoulli(normcdf(M(obs==0),0,sigma),pred_MMGN(obs==0)));
        KL_MMGN(ir,mc) = mean(KL_bernoulli(normcdf(M(:),0,sigma),pred_MMGN(:)));
        KL_unobs_MMGN(ir,mc) = mean(KL_bernoulli(normcdf(M(obs==0),0,sigma),pred_MMGN(obs==0)));

        % TraceNorm
        t0 = tic;
        [pred_Trace,Mhat_Trace,rank_Trace] = TraceNorm(Y,ind_omega,sigma,max_rank);
        time_Trace(ir,mc) = toc(t0);
        MSE_Trace(ir,mc) = norm(Mhat_Trace-M,'fro')^2/norm(M,'fro')^2;
        hellinger_Trace(ir,mc) = norm(sqrt(pred_Trace)-sqrt(normcdf(M,0,sigma)),'fro')^2+norm(sqrt(1-pred_Trace)-sqrt(1-normcdf(M,0,sigma)),'fro')^2;
        hellinger_unobs_Trace(ir,mc) = mean(hellinger_bernoulli(normcdf(M(obs==0),0,sigma),pred_Trace(obs==0)));
        KL_Trace(ir,mc) = mean(KL_bernoulli(normcdf(M(:),0,sigma),pred_Trace(:)));
        KL_unobs_Trace(ir,mc) = mean(KL_bernoulli(normcdf(M(obs==0),0,sigma),pred_Trace(obs==0)));

        % MaxNorm
        t0 = tic;
        [pred_Max,Mhat_Max,rank_Max] = MaxNorm(Y,ind_omega,sigma,max_rank);
        time_Max(ir,mc) = toc(t0);
        MSE_Max(ir,mc) = norm(Mhat_Max-M,'fro')^2/norm(M,'fro')^2;
        hellinger_Max(ir,mc) = norm(sqrt(pred_Max)-sqrt(normcdf(M,0,sigma)),'fro')^2+norm(sqrt(1-pred_Max)-sqrt(1-normcdf(M,0,sigma)),'fro')^2;
        hellinger_unobs_Max(ir,mc) = mean(hellinger_bernoulli(normcdf(M(obs==0),0,sigma),pred_Max(obs==0)));
        KL_Max(ir,mc) = mean(KL_bernoulli(normcdf(M(:),0,sigma),pred_Max(:)));
        KL_unobs_Max(ir,mc) = mean(KL_bernoulli(normcdf(M(obs==0),0,sigma),pred_Max(obs==0)));

        % EB1
        t0 = tic;
        [SIGMA1,pred_EB0,cnt1] = EB1(Y,0,100,20);
        time_EB1(ir,mc) = toc(t0);
        Mhat_EB1 = norminv(pred_EB1,0,sigma);
        MSE_EB1(ir,mc) = norm(Mhat_EB1-M,'fro')^2/norm(M,'fro')^2;
        hellinger_EB1(ir,mc) = norm(sqrt(pred_EB1)-sqrt(normcdf(M,0,sigma)),'fro')^2+norm(sqrt(1-pred_EB1)-sqrt(1-normcdf(M,0,sigma)),'fro')^2;
        hellinger_unobs_EB1(ir,mc) = mean(hellinger_bernoulli(normcdf(M(obs==0),0,sigma),pred_EB1(obs==0)));
        KL_EB1(ir,mc) = mean(KL_bernoulli(normcdf(M(:),0,sigma),pred_EB1(:)));
        KL_unobs_EB1(ir,mc) = mean(KL_bernoulli(normcdf(M(obs==0),0,sigma),pred_EB1(obs==0)));

        % EB2
        t0 = tic;
        [mu2,SIGMA2,pred_EB2,cnt2] = EB2(Y,100,20);
        time_EB2(ir,mc) = toc(t0);
        Mhat_EB2 = norminv(pred_EB2,0,sigma);
        MSE_EB2(ir,mc) = norm(Mhat_EB2-M,'fro')^2/norm(M,'fro')^2;
        hellinger_EB2(ir,mc) = norm(sqrt(pred_EB2)-sqrt(normcdf(M,0,sigma)),'fro')^2+norm(sqrt(1-pred_EB2)-sqrt(1-normcdf(M,0,sigma)),'fro')^2;
        hellinger_unobs_EB2(ir,mc) = mean(hellinger_bernoulli(normcdf(M(obs==0),0,sigma),pred_EB2(obs==0)));
        KL_EB2(ir,mc) = mean(KL_bernoulli(normcdf(M(:),0,sigma),pred_EB2(:)));
        KL_unobs_EB2(ir,mc) = mean(KL_bernoulli(normcdf(M(obs==0),0,sigma),pred_EB2(obs==0)));
    end
end
