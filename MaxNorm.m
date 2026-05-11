function [pred_MaxNorm,Mhat_MaxNorm,rank_MaxNorm] = MaxNorm(Y,ind_omega,sigma,max_rank)
    opts = [];
    opts.rSeq = 1:max_rank;
    opts.alpha = 5;
    X = Y;
    X(X==0) = -1;
    X(isnan(X)) = 0;
    [Mhat_MaxNorm,rank_MaxNorm] = MaxNorm_auto_modified(X,ind_omega,@(x)normcdf(x,0,sigma),@(x)normpdf(x,0,sigma),[],opts);
    pred_MaxNorm = normcdf(Mhat_MaxNorm,0,sigma);
end
