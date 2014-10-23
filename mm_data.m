function data = mm_data(a,b,n,Beta,Sigma,sigma,distribution,sOK,qOK)

% Beta must be K-by-1
Beta = reshape(Beta,length(Beta),1);

% epsilon ~ N(0,sigma)
epsilon = normrnd(0,sigma,n,1);
epsilon = epsilon - mean(epsilon); % force mean(epsilon)=0. If sOK=true, mean(Y) will be zero and therefore alpha can be properly omitted in Lasso (see Tibshirani (1996), p.268)

% matrix X bounded within [-a b] and correlation matrix Sigma
K = size(Sigma,1);  % X columns
switch distribution % select sampling distribution
    case 'uniform'
        X = a + (b-a).*rand(n,K);
        X = X * chol(Sigma);
    case 'normal'
        X = mvnrnd(zeros(n,K),Sigma);
        if ~any(isinf([a b]))            
            Xmax = max(X(:));
            Xmin = min(X(:));
            X = ((X-Xmin) / (Xmax-Xmin)) * (b-a) + a;
        end
        
end

% standardized X
if sOK
	X = zscore(X);
end

% Quantization: we constraint the X values from the real numbers to a small
% discrete set = the integers. The models (and the real subjects) are given
% the discrete X only.
if qOK
    X = round(X);
end

% simulated data
Y = X * Beta + epsilon; 

% debug
% mu = [0 0 0 0 0 0 0 0];
% i = 1:8;
% matrix = abs(bsxfun(@minus,i',i));
% covariance = repmat(.5,8,8).^matrix;
% X = mvnrnd(mu, covariance, 20);
% Beta = [3; 1.5; 0; 0; 2; 0; 0; 0];
% ds = dataset(Beta);
% Y = X * Beta + 3 * randn(20,1);

% Output
data.Y = Y;
% data.X = [X ones(n,1)];   % include constant column
data.X = X;
data.Sigma = Sigma;

end

