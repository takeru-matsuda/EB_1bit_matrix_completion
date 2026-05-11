addpath(genpath('./comparison_methods'));
rng('default');

movie = dlmread('movielens100K');
Y0 = zeros(max(movie(:,2)),max(movie(:,1)));
obs0 = zeros(max(movie(:,2)),max(movie(:,1)));
for i=1:size(movie,1)
    if movie(i,3)>=4
        Y0(movie(i,2),movie(i,1)) = 1;
    else
        Y0(movie(i,2),movie(i,1)) = 0;
    end
    obs0(movie(i,2),movie(i,1)) = 1;
end
tmp = sum(obs0')>=5;
Y0 = Y0(tmp,:);
obs0 = obs0(tmp,:);
p = size(Y0,1);
q = size(Y0,2);
obs = double(obs0==1 & rand(p,q)<.5);
Y = Y0;
Y(obs==0) = NaN;
ind_omega = obs(:);
max_rank = 10;
sigma = 1;

% MMGN
t0 = tic;
[pred_MMGN,Mhat_MMGN,rank_MMGN] = MMGN(Y,ind_omega,sigma,max_rank);
time_MMGN = toc(t0);

% TraceNorm
t0 = tic;
[pred_Trace,Mhat_Trace,rank_Trace] = TraceNorm(Y,ind_omega,sigma,max_rank);
time_Trace = toc(t0);

% MaxNorm
t0 = tic;
[pred_Max,Mhat_Max,rank_Max] = MaxNorm(Y,ind_omega,sigma,max_rank);
time_Max = toc(t0);

% EB1
t0 = tic;
[SIGMA1,pred_EB1,cnt1] = EB1(Y,100,20);
time_EB1 = toc(t0);

% EB2
t0 = tic;
[mu2,SIGMA2,pred_EB2,cnt2] = EB2(Y,100,20);
time_EB2 = toc(t0);

pred_loss_MMGN = zeros(p,q);
pred_loss_Trace = zeros(p,q);
pred_loss_Max = zeros(p,q);
pred_loss_EB1 = zeros(p,q);
pred_loss_EB2 = zeros(p,q);
for i=1:p
    for j=1:q
        if Y0(i,j) == 1
            pred_loss_MMGN(i,j) = -log(pred_MMGN(i,j));
            pred_loss_Trace(i,j) = -log(pred_Trace(i,j));
            pred_loss_Max(i,j) = -log(pred_Max(i,j));
            pred_loss_EB1(i,j) = -log(pred_EB1(i,j));
            pred_loss_EB2(i,j) = -log(pred_EB2(i,j));
        else
            pred_loss_MMGN(i,j) = -log(1-pred_MMGN(i,j));
            pred_loss_Trace(i,j) = -log(1-pred_Trace(i,j));
            pred_loss_Max(i,j) = -log(1-pred_Max(i,j));
            pred_loss_EB1(i,j) = -log(1-pred_EB1(i,j));
            pred_loss_EB2(i,j) = -log(1-pred_EB2(i,j));
        end
    end
end
pred_obs_MMGN = mean(pred_loss_MMGN(obs==1));
pred_obs_Trace = mean(pred_loss_Trace(obs==1));
pred_obs_Max = mean(pred_loss_Max(obs==1));
pred_obs_EB1 = mean(pred_loss_EB1(obs==1));
pred_obs_EB2 = mean(pred_loss_EB2(obs==1));
pred_unobs_MMGN = mean(pred_loss_MMGN(obs==0 & obs0==1));
pred_unobs_Trace = mean(pred_loss_Trace(obs==0 & obs0==1));
pred_unobs_Max = mean(pred_loss_Max(obs==0 & obs0==1));
pred_unobs_EB1 = mean(pred_loss_EB1(obs==0 & obs0==1));
pred_unobs_EB2 = mean(pred_loss_EB2(obs==0 & obs0==1));

accuracy_MMGN = mean((pred_MMGN(obs==0 & obs0==1)>=.5)==Y0(obs==0 & obs0==1));
accuracy_Trace = mean((pred_Trace(obs==0 & obs0==1)>=.5)==Y0(obs==0 & obs0==1));
accuracy_Max = mean((pred_Max(obs==0 & obs0==1)>=.5)==Y0(obs==0 & obs0==1));
accuracy_EB1 = mean((pred_EB1(obs==0 & obs0==1)>=.5)==Y0(obs==0 & obs0==1));
accuracy_EB2 = mean((pred_EB2(obs==0 & obs0==1)>=.5)==Y0(obs==0 & obs0==1));

nbin = 10;
calib_MMGN = zeros(1,nbin);
calib_Trace = zeros(1,nbin);
calib_Max = zeros(1,nbin);
calib_EB1 = zeros(1,nbin);
calib_EB2 = zeros(1,nbin);
ans_MMGN = zeros(1,nbin);
ans_Trace = zeros(1,nbin);
ans_Max = zeros(1,nbin);
ans_EB1 = zeros(1,nbin);
ans_EB2 = zeros(1,nbin);
cnt_MMGN = zeros(1,nbin);
cnt_Trace = zeros(1,nbin);
cnt_Max = zeros(1,nbin);
cnt_EB1 = zeros(1,nbin);
cnt_EB2 = zeros(1,nbin);
for i=1:p
    for j=1:q
        if obs0(i,j) == 0
            continue
        end
        if obs(i,j) == 1
            continue
        end
        answer = Y0(i,j);
        bin = max(1,ceil(pred_MMGN(i,j)*nbin));
        calib_MMGN(bin) = calib_MMGN(bin)+pred_MMGN(i,j)-answer;
        ans_MMGN(bin) = ans_MMGN(bin)+answer;
        cnt_MMGN(bin) = cnt_MMGN(bin)+1;
        bin = max(1,ceil(pred_Trace(i,j)*nbin));
        calib_Trace(bin) = calib_Trace(bin)+pred_Trace(i,j)-answer;
        ans_Trace(bin) = ans_Trace(bin)+answer;
        cnt_Trace(bin) = cnt_Trace(bin)+1;
        bin = max(1,ceil(pred_Max(i,j)*nbin));
        calib_Max(bin) = calib_Max(bin)+pred_Max(i,j)-answer;
        ans_Max(bin) = ans_Max(bin)+answer;
        cnt_Max(bin) = cnt_Max(bin)+1;
        bin = max(1,ceil(pred_EB1(i,j)*nbin));
        calib_EB1(bin) = calib_EB1(bin)+pred_EB1(i,j)-answer;
        ans_EB1(bin) = ans_EB1(bin)+answer;
        cnt_EB1(bin) = cnt_EB1(bin)+1;
        bin = max(1,ceil(pred_EB2(i,j)*nbin));
        calib_EB2(bin) = calib_EB2(bin)+pred_EB2(i,j)-answer;
        ans_EB2(bin) = ans_EB2(bin)+answer;
        cnt_EB2(bin) = cnt_EB2(bin)+1;
    end
end
ECE_MMGN = sum(abs(calib_MMGN))/sum(cnt_MMGN);
ECE_Trace = sum(abs(calib_Trace))/sum(cnt_Trace);
ECE_Max = sum(abs(calib_Max))/sum(cnt_Max);
ECE_EB1 = sum(abs(calib_EB1))/sum(cnt_EB1);
ECE_EB2 = sum(abs(calib_EB2))/sum(cnt_EB2);
