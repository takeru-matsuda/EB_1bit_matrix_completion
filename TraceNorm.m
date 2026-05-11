function [pred_Trace,Mhat_Trace,rank_Trace] = TraceNorm(Y,ind_omega,sigma,max_rank)
    options = struct();
    options.iterations = 1e4;
    options.stepMax    = 1e9;
    options.stepMin    = 1e-4;
    options.optTol     = 1e-3;
    options.verbosity  = 1;
    rSeq = 1:max_rank;
    rate = 0.8;
    alpha = 5;
    seed = 2022;
    X = Y;
    X(X==0) = -1;
    X(isnan(X)) = 0;
    [Mhat_Trace,rank_Trace] = TraceNorm_auto(X,ind_omega,@(x)normcdf(x,0,sigma),@(x)normpdf(x,0,sigma),rSeq,rate,seed,alpha*ones(size(X)),options);
    pred_Trace = normcdf(Mhat_Trace,0,sigma);
end
