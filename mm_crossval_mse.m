function cvmse = mm_crossval_mse(Y,X,Beta,nf)

regf = @(XTRAIN, ytrain, XTEST)(XTEST*regress(ytrain,XTRAIN));

if nf > 1
    cvmse = crossval('mse',X,Y,'predfun',regf, 'kfold', 5);
else
    cvmse = crossval('mse',X,Y,'predfun',regf, 'Leaveout');
end

se = (Y-X*Beta).^2;
mse = sum(se)/length(se);

end


function 