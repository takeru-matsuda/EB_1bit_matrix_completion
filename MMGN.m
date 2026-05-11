function [pred_MMGN,Mhat_MMGN,rank_MMGN] = MMGN(Y,ind_omega,sigma,max_rank)
    opts = [];
    opts.rSeq = 1:max_rank;
    X = Y;
    X(X==0) = -1;
    X(isnan(X)) = 0;
    [U,V,rank_MMGN] = MMGN_probit_auto(X,ind_omega,sigma,zeros(size(X)),opts);
    Mhat_MMGN = U*V';
    pred_MMGN = normcdf(U*V');
end
