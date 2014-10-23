function [lambda_opt, Beta_opt, cvMse] = mm_CrossValidation(data,method,CVO,Beta_true)

lambda_opt = [];

% this kfold is used to estimate Beta during lasso, ridge and 
% subset_selectionregression only, it is not the same as used to compare 
% the models later.
kfold = 5;  
maxK = 20;  % max lambda value for ridge and lasso_lambda

switch lower(method)
    case 'ls'
        Beta_opt = mm_LS(data);
        cvMse = crossvalidation(data,method,CVO);
        
    case 'variational_bayes'
        q = tapas_vblm(data.Y,data.X,2,0.2,10,1);
        Beta_opt = q.mu_n;
        cvMse = crossvalidation(data,method,CVO);
        
    case 'ls_matlab'
        Beta_opt = regress(data.Y,data.X);
        cvMse = crossvalidation(data,method,CVO);
        
    case 'lasso_matlab'
        % This while() loop is to avoid unexpected all zero betas solution
        Beta_opt = zeros(size(data.X,2),1);
        while all(Beta_opt==0)
            [B,Stats] = lasso(data.X, data.Y, 'CV', kfold);
            Beta_opt = B(:,Stats.IndexMinMSE);
            a=1;
        end
        lambda_opt = Stats.LambdaMinMSE;
        cvMse = crossvalidation(data,method,CVO,lambda_opt);
        
        if false
            figure(1)
            plot(Stats.Lambda,Stats.MSE)
            xlabel('Lambda');
            ylabel('MSE');
            msg1 = sprintf('Beta true    = [%s]', sprintf('%0.3f, ', Beta_true));
            msg2 = sprintf('Beta optimal = [%s]', sprintf('%0.3f, ', Beta_opt));
            title([msg1 sprintf('\n') msg2]);
            pause
        end
        
    case 'lasso_matlab_lambda'
        % To show that 'lasso_matlab_lambda' is equivalent to 'lasso_matlab'
        % we need to set: rng('default'), k = Stats.Lambda;
        k = 0:.01:maxK;
        N = length(k);
        C = cvpartition(length(data.Y),'Kfold',kfold);
        mse = zeros(1,N);
        for n = 1:N
            mse(n) = crossvalidation(data,'lasso_matlab',C,k(n));
        end
        [minMse,IndexMinMSE] = min(mse);
        lambda_opt = k(IndexMinMSE);
        Beta = lasso(data.X,data.Y,'Lambda',lambda_opt);
        Beta_opt = Beta(2:end);
        cvMse = crossvalidation(data,'lasso_matlab',CVO,lambda_opt);
        
    case 'ridge_matlab'
        k = 0:.01:maxK;
        N = length(k);
        C = cvpartition(length(data.Y),'Kfold',kfold);
        mse = zeros(1,N);
        for n = 1:N
            mse(n) = crossvalidation(data,'ridge_matlab',C,k(n));
        end
        [minMse,IndexMinMSE] = min(mse);
        lambda_opt = k(IndexMinMSE);
        Beta = ridge(data.Y,data.X,lambda_opt,0);
        Beta_opt = Beta(2:end);
%         % Equivalent solution. Note that data.X should not contain a column of 1s.
%         Beta = ridge(data.Y,data.X,lambda_opt,1); 
%         Beta_opt = Beta ./ std(data.X,0,1)';
        cvMse = crossvalidation(data,method,CVO,lambda_opt);
        
    case 'subset_matlab'
        nBeta = size(data.X,2);
        index = dec2bin(1:2^nBeta-1);
        index = index == '1';
        N = size(index,1);
        mse = zeros(1,N);
        datat = data;
        C = cvpartition(length(datat.Y),'Kfold',kfold);
        for n = 1:N
            datat.X = data.X(:,index(n,:));
            mse(n) = crossvalidation(datat,'ls_matlab',C);
        end
        [minMse,IndexMinMSE] = min(mse);
        selection = index(IndexMinMSE,:);
        Beta_opt = zeros(nBeta,1);
        Beta_opt(selection) = regress(data.Y,data.X(:,selection));
        %lambda_opt = abs(min(Beta_opt(selection)));
        datat.X = data.X(:,selection);
        cvMse = crossvalidation(datat,'ls_matlab',CVO,lambda_opt);

    case 'subset_selection_tibshirani'  % Tibshirani (1994), p.280
        nBeta = size(data.X,2);
        index = dec2bin(1:2^nBeta-1);
        index = index == '1';
        N = size(index,1);
        mse = zeros(1,N);
        p = zeros(1,N);
        datat = data;
        % Compute MSE for every subset
        for n = 1:N
            datat.X = data.X(:,index(n,:));
            Beta_opt = regress(datat.Y,datat.X);
            mse(n) = mm_mse(datat.Y,datat.X,Beta_opt);
            p(n) = sum(index(n,:));  % subset size
        end
        % Find the best subset of each size
        minMse = zeros(1,nBeta);
        IndexMinMSE = zeros(1,nBeta);
        for nb = 1:nBeta
            idx = find(p==nb);
            [minMse(nb),minIdx] = min(mse(idx));
            IndexMinMSE(nb) = idx(minIdx);
        end
        % For each cross-validation, find the best subsets of each size
        mseCV = zeros(1,nBeta);
        C = cvpartition(length(datat.Y),'Kfold',kfold);
        for nb = 1:nBeta
            datat.X = data.X(:,index(IndexMinMSE(nb),:));
            mseCV(nb) = crossvalidation(datat,'ls_matlab',C);
        end
        [minMseCV,IndexMinMseCV] = min(mseCV);
        selection = index(IndexMinMSE(IndexMinMseCV),:);
        Beta_opt = zeros(nBeta,1);
        Beta_opt(selection) = regress(data.Y,data.X(:,selection));
        %lambda_opt = abs(min(Beta_opt(selection)));
        datat.X = data.X(:,selection);
        cvMse = crossvalidation(datat,'ls_matlab',CVO,lambda_opt);
        
    otherwise
        fprintf(1, '    Lambda search...');  % feedback
        options = optimset('Display','notify','FunValCheck','on','MaxFunEvals',10^6);
        log_lambda_opt = fminsearch(@cf_lambda,0,options,data,method,CVO);
        lambda_opt = exp(log_lambda_opt);
        fprintf(1, ' done.');  % feedback
        
        fprintf(1, ' Beta search...');  % feedback
        switch lower(method)
            case 'lasso'
                Beta_opt = mm_Lasso(data,lambda_opt);
            case 'lasso_tibshirani'
                Beta_opt = mm_betafit(data,lambda_opt,4);
            case 'ridge'
                Beta_opt = mm_Ridge(data,lambda_opt);
            case 'subset_selection'
                Beta_opt = mm_SubsetSelection(data,lambda_opt);
        end
        cvMse = crossvalidation(data,method,CVO,lambda_opt);
end

end


function cvMse = cf_lambda(x,data,method,CVO)

% transform lambda (lambda>0)
lambda = exp(x);

cvMse = crossvalidation(data,method,CVO,lambda);

end


function cvMse = crossvalidation(data,method,CVO,lambda)

se = zeros(1,CVO.NumTestSets);    % initialize SE vector

for i = 1:CVO.NumTestSets
    trIdx = CVO.training(i);
    teIdx = CVO.test(i);
          
    datat = data;
    datat.X = data.X(trIdx,:);
    datat.Y = data.Y(trIdx);
    
    switch lower(method)
        case 'ls'
            Beta = mm_LS(datat);
        case 'ls_matlab'
            Beta = regress(datat.Y,datat.X);
        case 'variational_bayes'
            q = tapas_vblm(data.Y,data.X,2,0.2,10,1);
            Beta = q.mu_n;
        case 'lasso'
            Beta = mm_Lasso(datat,lambda);
        case 'lasso_tibshirani'
            Beta = mm_betafit(datat,lambda,4);
        case 'lasso_matlab'
            Beta = lasso(datat.X,datat.Y,'Lambda',lambda);
        case 'ridge'
            Beta = mm_Ridge(datat,lambda);
        case 'ridge_matlab'
            Beta = ridge(datat.Y,datat.X,lambda,0);
            Beta = Beta(2:end);
        case 'subset_selection'
            Beta = mm_SubsetSelection(datat,lambda);
    end
    
    % prediction error of the fitted model when predicting 
    % every observation out of Y(tout)
    se(i) = sum( (data.Y(teIdx)-data.X(teIdx,:)*Beta).^2 ); 
end

cvMse = sum(se)/sum(CVO.TestSize);

end
