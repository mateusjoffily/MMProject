function Beta_ridge = mm_Ridge(data,lambda)

% check if Sigma is Identity
[I,J] = find(data.Sigma);
isI = all(I == J);

if isI % CASE WITH ORTHONORMAL INDICATORS: Sigma=I
    Beta_ridge = mm_LS(data) / (1+lambda);
    
else % GENERAL CASE: Sigma is not the I matrix (indicators are correlated)
    Beta_ridge = mm_betafit(data,lambda,2);
%     Beta_ridge = inv(data.X' * data.X + lambda * eye(size(data.X,2))) * data.X' * data.Y;
end

end
