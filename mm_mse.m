function mse = mm_mse(Y,X,Beta)

se = (Y-X*Beta).^2;
mse = sum(se)/length(se);

end