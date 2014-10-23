function Beta_SS = mm_SubsetSelection(data,lambda)

% check if Sigma is Identity
[I,J] = find(data.Sigma);
isI = all(I == J);

if isI 
    % CASE WITH ORTHONORMAL INDICATORS: Sigma=I
    Beta_SS = mm_LS(data);
    Beta_SS(abs(Beta_SS)<lambda) = 0;
else
    % GENERAL CASE: Sigma is not Identity matrix
    Beta_SS = mm_betafit(data,lambda,0);
end

end