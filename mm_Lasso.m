function Beta_lasso = mm_Lasso(data,lambda)

% check if Sigma is Identity
[I,J] = find(data.Sigma);
isI = all(I == J);

if isI % CASE WITH ORTHONORMAL INDICATORS: Sigma=I
    Beta_LS = mm_LS(data);
    dBL = abs(Beta_LS) - lambda;
    Beta_lasso = sign(Beta_LS) .* ( dBL ) ;
    Beta_lasso(dBL<0) = 0;
    
else % GENERAL CASE: Sigma is not the I matrix (indicators are correlated)
    Beta_lasso = mm_betafit(data,lambda,1);
end

end