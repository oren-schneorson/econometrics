function [ out ] = autocorr_( y, numLags )
%AUTOCORR Sample autocorrelation
%   Compute the sample autocorrelation function (ACF) of a multivariate,
%   stochastic time series y.

    y_ = y(1:end-numLags,:);
    y = y(numLags+1:end,:);

    %{
    ax = 1;
    out = nansum(...
        (y-nanmean(y, ax)).*...
        (y_-nanmean(y_, ax)), ax)./...
        sqrt(...
        nansum((y-nanmean(y, ax)).^2, ax).*...
        nansum((y_-nanmean(y_, ax)).^2, ax));
    %}
   
    out = corrcoef(y,y_);
    out = out(1,2);

end

 
