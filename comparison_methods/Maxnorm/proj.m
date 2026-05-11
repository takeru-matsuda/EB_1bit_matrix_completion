function PU=proj(U,B)

dim = size(U);
n_U = sum(U.^2,2);   % vector of l2-norms of rows of P
f_U = find(n_U>B);
if length(f_U) > 0
   U(f_U,:)=sqrt(B)*U(f_U,:)./(sqrt((n_U(f_U)))*ones(1,dim(2))); 
end
PU = U;