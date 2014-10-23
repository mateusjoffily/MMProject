function [out,MSE] = mm_CompPred(N,a,b,n,Sigma,Beta,sigma,distribution,methods,nf,sOK,qOK,fout)

if nargin == 0
    % Default values
    N = 1;                          % number of data sets
    a = -100;                       % X values lower bound
    b = 100;                        % X values upper bound
    n = 100;                        % Y length
    % Sigma = toeplitz(0.5.^(0:5));   % X correlation matrix K-by-K 
    Sigma = eye(5);                
    Beta = ones(size(Sigma,1),1);   % beta coefficients vector K-by-1
    sigma = 1;                      % Y standard deviation
    distribution = 'uniform'; % sampling distribution: 'normal' or 'uniform'
    qOK = false;    % if true: quantize X (integers)
    sOK = true;   % if true: X is standardized
    methods = {'ls' 'lasso' 'lasso_matlab' 'ridge' 'subset_selection'};
    nf = 5;    % nf-fold cross-validation (use nf=1 for leave-one-out)
    fout = '';
end

% Add path to VB toolbox
addpath('./VBLM_1.02');

% set random number generator to default
try
    rng('default');
catch
    rand('twister',5489);
end

% use methods
nMethods = length(methods);

% loop over data sets
for i = 1:N   
    
    % feedback
    fprintf(1, 'Simulation %d of %d:\n',i,N);
    
    % generate data
    data = mm_data(a,b,n,Beta,Sigma,sigma,distribution,sOK,qOK);
    
    % Create a cross-validation partition for data
    if nf == 1
        % leave-one-out cross-validation
        CVO = cvpartition(n,'Leaveout');
    else
        % nf-fold cross-validation
        CVO = cvpartition(n,'Kfold',nf);
    end

    % loop over methods
    for m = 1:nMethods
        fprintf(1, '  %s running:', methods{m});  % feedback
        
        % run cross-validation
        [lambda_opt, Beta_opt, cvMse] = mm_CrossValidation(data,methods{m},CVO,Beta);
        out.(methods{m}).Beta(:,i) = Beta_opt;
        if ~isempty(lambda_opt)
            out.(methods{m}).lambda(i) = lambda_opt;
        end
        out.(methods{m}).mse(i) = cvMse;
        
        fprintf(1, ' done.\n');  % feedback
    end
end

% Save final statistics
fid = fopen(sprintf('results%s.txt',fout), 'w');   % open output file

% header of statistics table
fprintf(fid, 'method\t N\t average\t median\t std\t proportion\n');

MSE = zeros(N,nMethods);   % MSE matrix
tnnb = Beta~=0;   % true non-null Beta

for m = 1:nMethods   % loop over methods
    MSE(:,m) = out.(methods{m}).mse;
    fb = out.(methods{m}).Beta(tnnb,:);  % fitted Beta within set of tnnb
    fnnb = prod(double(fb~=0));          % fitted non null Beta
    fprintf(fid, '%s\t %d\t %0.4f\t %0.4f\t %0.4f\t %0.4f\n', ...
        methods{m}, N, mean(MSE(:,m)), median(MSE(:,m)), ...
        std(MSE(:,m)), mean(fnnb));
end 

fclose(fid);  % close output file

% Save simulation results
fid = fopen(sprintf('simulations%s.txt',fout), 'w');   % open output file

% header of simulations table
fprintf(fid, 'simulation\t method\t ');
fprintf(fid, 'beta_%d\t ', 1:length(Beta));
fprintf(fid, 'lambda\n');

for ns = 1:N
    for m = 1:nMethods   % loop over methods
        fprintf(fid, '%d\t %s\t', ns, methods{m});
        fprintf(fid, '%0.4f\t ', out.(methods{m}).Beta(:,ns));
        if ~isfield(out.(methods{m}), 'lambda')
            fprintf(fid, 'NaN\n');
        else
            fprintf(fid, '%0.4f\n', out.(methods{m}).lambda(ns));
        end
    end

end

for m = 1:nMethods   % loop over methods
    fprintf(fid, ' \t %s\t', methods{m});
    fprintf(fid, '%0.4f\t ', mean((out.(methods{m}).Beta~=0),2));
    fprintf(fid, '\n');
end

fclose(fid);  % close output file

% Boxplot
figure('Color', 'w')
boxplot(MSE, 'labels', methods);
ylabel('mse');
xlabel('methods');

end