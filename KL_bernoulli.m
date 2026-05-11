function KL = KL_bernoulli(p,q)
    KL = zeros(size(p));

    idx = (p > 0);
    KL(idx) = KL(idx) + p(idx).*log(p(idx)./q(idx));

    idx = (p < 1);
    KL(idx) = KL(idx) + (1-p(idx)).*log((1-p(idx))./(1-q(idx)));
end
