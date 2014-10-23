close all
clear all

% N = 500;    % number of data sets
% n = 100;    % Y length
% n = 20;  % replicate T
% N = 50;   % replicate T
n = 20;  % replicate T
N = 200;   % replicate T

% Sigma = eye(length(Beta));

% compare performance of 4 methods in 3 worlds: simple, intermediary,
% and complex. 

% check results with alternative methods for cross-validation procedure
% (different values of nf)

% simple world
% Beta{1} = [0 0 0 0 5];  % simple world
Beta{1} = [5 0 0 0 0 0 0 0];  % simple world
% sigma{1} = 2; % Note: sigma should be 2 to give a chance to Best Subset Selection
world{1} = 'Simple';

% intermediary world
% Beta{2} = [3 1.5 2 0 0];
Beta{2} = [3 1.5 0 0 2 0 0 0];
% sigma{2} = 3;
world{2} = 'Interm';

% complex world
%Beta{3} = [0.85 0.85 0.85 0.85 0.85];
Beta{3} = [0.85 0.85 0.85 0.85 0.85 0.85 0.85 0.85];
% sigma{3} = 3;
world{3} = 'Complex';

distribution = 'normal'; % sampling distribution: 'normal' or 'uniform'
qOK = false;   % if true: quantize X (integers)
sOK = true;   % if true: X is standardized
a = -100;   % X values lower bound
b = 100;    % X values upper bound

% methods = {'ls' 'ls_matlab' 'lasso' 'lasso_matlab' 'lasso_tibshirani' 'ridge' 'subset_selection' 'subset_selection_tibshirani' 'subset_matlab' 'variational_bayes'};
% methods = {'ls_matlab' 'lasso_matlab'  'lasso_matlab_lambda' 'ridge_matlab' 'subset_matlab' 'variational_bayes'};
% methods = {'ls_matlab' 'lasso_matlab'  'ridge_matlab' 'subset_matlab' 'subset_selection_tibshirani' 'variational_bayes'};
% methods = {'subset_matlab' 'subset_selection_tibshirani'};
methods = {'ls_matlab' 'lasso_matlab' 'ridge_matlab' 'subset_matlab' 'subset_selection_tibshirani' 'variational_bayes'};
% methods = {'ls_matlab' 'variational_bayes'};
% methods = {'lasso_matlab'  'variational_bayes'};
% methods = {'lasso_matlab'};

sigma = [3 6 9];
nf = [1 5 10];   % nf-fold cross-validation (use nf=1 for leave-one-out); re-do the analysis with nf=10 and nf=1

% Beta = {}; sigma ={}; world = {};
% Beta{1} = [0.85 0.85 0.85 0.85 0.85 0.85 0.85 0.85];
% sigma{1} = 3;
% world{1} = 'Complex';
% nf = 1;
% methods = {'ls_matlab' 'lasso_matlab' 'ridge_matlab' 'subset_matlab' 'variational_bayes'};
% n = 20;  % replicate T
% N = 200;   % replicate T

for nnf = 1:length(nf)
%    for nw = 1:length(Beta)
    for nw = 3
        for ns = 1:length(sigma)
            fout = sprintf('_%s_nf%d_8var_n%d_N%d_SIGMA%0.0f_ALL', world{nw}, nf(nnf),n, N,sigma(ns));
            Sigma = toeplitz(0.5.^(0:length(Beta{nw})-1));  % corr(X_i,X_j)=0.5^abs(i-j)
            [out,MSE] = mm_CompPred(N,a,b,n,Sigma,Beta{nw},sigma(ns),distribution,methods,nf(nnf),sOK,qOK,fout);
        end
    end
end
