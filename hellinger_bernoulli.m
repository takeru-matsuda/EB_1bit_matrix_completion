function hellinger = hellinger_bernoulli(p,q)
    hellinger = (sqrt(p)-sqrt(q)).^2+(sqrt(1-p)-sqrt(1-q)).^2;
end
