function Beta_opt = mm_betafit(data,lambda,q)

options = optimset('Display','notify','FunValCheck','on',...
                   'MaxFunEvals',10^6,'MaxIter',10^4);

% Set initial value for Beta
Beta0 = mm_LS(data);

if q == 0 
    % Subset Selection
    Beta_opt = fminsearch(@costfun_ss,Beta0,options,data,lambda);
    
elseif q == 4 
    % lasso_tibshirani: Algorithm described in Tibshirani(1996), p. 278.
    GE = sign(Beta0)';
    while true
        Beta_opt = fminsearch(@costfun_tibsh,Beta0,options,data,lambda,GE);
        if lambda*sum(abs(Beta_opt)) <= 1
            break
        end
        GE = [GE;sign(Beta_opt)'];
    end
    
else
    % Lasso and Ridge
    Beta_opt = fminsearch(@costfun,Beta0,options,data,lambda,q);
end

end


function c = costfun(Beta,data,lambda,q)

    mse = mm_mse(data.Y,data.X,Beta); 
    c = mse + lambda*sum(abs(Beta).^q);
    
end


function c = costfun_ss(Beta,data,lambda)

    mse = mm_mse(data.Y,data.X,Beta); 
%     c = mse + lambda*sum(Beta>0);
    c = mse + lambda*sum(abs(Beta)>0);

end


function c = costfun_tibsh(Beta,data,lambda,GE)

    if any( lambda*GE*Beta > 1 )
        c = Inf;
    else
        mse = mm_mse(data.Y,data.X,Beta); 
        c = mse;
    end
    
end