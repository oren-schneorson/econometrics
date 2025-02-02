function [E_u_given_z, E_v_given_z] = wandering_intercept(z)
%WANDERING_INTERCEPT get conditional expectation of u and v given observed z
%   Wandering intercept procedure of Fama and Gibbons (1982), based on
%   Craig Ansley (1980) used by Fama and Gibbons to correct for changes in 
%   the expected real interest rate. This approach followed research by 
%   Hess (1975), Fama (1976), Nelson (1977) and Garbade (1978) which 
%   characterized expected real rates as a slow-moving random walk. 
%   Ansley assumed that the regression intercept follows a random walk:
%   Pi_t = alpha_{t-1} + BEI_{t-1} + u_t
%   dPi_t = dalpha_{t-1} + dBEI_{t-1} + du_t
%   dalpha_{t-1} 
%   
%   z_t = u_t - u_{t-1} + v_t  # observed signal (residual)
%
%   u_t ~ N(0, var_u)
%   dalpha_{t-1}  = v_t ~ N(0, var_v)
%
%   cov(u_t, u_s) = 0 for all s ~= t
%   cov(v_t, v_s) = 0 for all s ~= t
%   cov(u_t, v_s) = 0 for any s,t


    N = size(z, 1);

    % variances
    % ***********
    var_z = var(z); % sample variance of signal, unconditional
    autocorr_z = autocorr_(z, 1);
    var_u = - autocorr_z .* var_z;
    var_v = var_z - 2 * var_u;


    if var_v < 0
        warning('autocorrelation of signal lower than -0.50. Setting var_v to 10% of var_z')
        var_v = 0.1*var_z;
        var_u = (var_z-var_v)/2;

    end

    
    % covariances
    % ***********
    
    Sigma_u = eye(N) .* var_u;

    % construct Sigma_z
    Sigma_z_ = 3 * eye(N) - ones(N); % [(2, -1, 0,..., 0); (0, 2, -1,...,0);...]
    Sigma_z_ = Sigma_z_-spdiags(zeros(N, 3), -1:1, Sigma_z_);
    Sigma_z = Sigma_z_ .* var_u + eye(N) .* var_v;
    

    % construct Sigma_u_z
    Sigma_u_z = 2 * eye(N) - ones(N); % [(1, -1, 0,..., 0); (0, 1, -1,...,0);...]
    Sigma_u_z = Sigma_u_z-spdiags(zeros(N, 2), 0:1, Sigma_u_z);
    Sigma_u_z = Sigma_u_z .* var_u;
    

    % construct Sigma_v_z
    Sigma_v_z = eye(N) * var_v;

    % inverting Sigma_z: Sigma_z is a sum of two matrices that have closed form
    % inverse matrix.
    % invert a sum of knowln components inverses
    % see https://math.stackexchange.com/questions/17776/inverse-of-the-sum-of-matrices
    
    % first component, the inverse of the moving average variance-covariance matrix
    [~, inv_Sigma_z_] = get_H_movavg_(-1, N);
    % second component, simple inverse of identity matrix...
    
    % combining components following Miller, 1981 (Mathematics Magazine)
    inv_Sigma_z = inv_sum_(...
        Sigma_z_*var_u, eye(N)*var_v,...
        inv_Sigma_z_/var_u, eye(N)/var_v...
        );
    E_u_given_z = Sigma_u_z * inv_Sigma_z * (z-mean(z));
    E_v_given_z = Sigma_v_z * inv_Sigma_z * (z-mean(z));


end



function [H, V] = get_H_movavg_(alpha, N)
%GET_H_MOVAVG Get transformation of moving average model given alpha.
%   The Inverse of a Matrix Occurring in First-Order Moving-Average Models
%   Relevant for any model with a residual of:
%   epsilon_{t}+alpha*epsilon_{t-1}, 
%   where Cov(epsilon_{t}, epsilon_{t-1}) = 0
%
%   For further info, see V. R. R. Uppuluri and J. A. Carpenter Sankhyā:
%   The Indian Journal of Statistics, Series A (1961-2002)
%   Mar., 1969, Vol. 31, No. 1 (Mar., 1969), pp. 79-82
%   [Uppuluri-InverseMatrixOccurring-1969.pdf]
%   Assumption: unit variance.

%{
% can't handle such low alpha
if abs(alpha) < 1e15
    V = NaN;
    H = NaN;
    return
end
%}

if alpha == 0
    H = eye(N);
    V = eye(N);
    return
end

delta = NaN(N+1, 1);
delta(1:2) = [ones(size(alpha)); 1+alpha.^2];
%delta = [ones(size(alpha)); 1+alpha.^2];

for j = 2:N
    %delta = [delta; (1+alpha.^2)*delta(end)-alpha.^2*delta(end-1)];
    delta(j+1) = (1+alpha.^2)*delta(j)-alpha.^2*delta(j-1);
end

%v_1j = [];
%v_jj = [];

v_1j = NaN(N, 1);
v_jj = NaN(N, 1);

for j = 1:N
    %v_1j = [v_1j; (-1)^(j-1) * alpha.^(j-1)*delta(1+N-j)/delta(1+N)];
    %v_jj = [v_jj; delta(1+N-j)*delta(j)/delta(1+N)];
    v_1j(j) = (-1)^(j-1) * alpha.^(j-1)*delta(1+N-j)/delta(1+N);
    v_jj(j) = delta(1+N-j)*delta(j)/delta(1+N);
end


lambda_1j = v_jj./v_1j;
V = zeros(N);

for j = 2:N
    V(j, j) = v_jj(j);
for k = j+1:N
    V(j, k) = lambda_1j(j).* v_1j(k);
    V(k, j) = lambda_1j(j).* v_1j(k);
end
end

% computational error with large matrices, setting corrupt part of the
% matrix as symmetrical to the previous block up the diagonal of the matrix
[a, b] = find(isinf(V) | isnan(V));

if numel(a) > 0
    warning('Moving average parameter and sample size imply some elements could not be calculated due to computation error. setting matrix as symmetrical to the previous block up the diagonal of the matrix')
    V_computational_limit = V(min(a):end, min(b):end);

    V(min(a):end, min(b):end) = ...
    V(min(a)-size(V_computational_limit, 1):min(a)-1,...
        min(b)-size(V_computational_limit, 2):min(b)-1);
end

V(1,:) = v_1j;
V(:,1) = v_1j';



H = chol(V);

end


function [output] = inv_sum_(A, B, inv_A, inv_B)
%INV_SUM Invert sum of matrices inv(A+B) with known inverse inv(A), inv(B)
%   Following Miller, 1981 (Mathematics Magazine)
%   https://math.stackexchange.com/questions/17776/inverse-of-the-sum-of-matrices
%   Must have square matrices. Assuming full rank.

N = size(A, 1);

% unused
%C = cell(N+1, 1);
%C{1} = A;

inv_C = cell(N+1, 1);
inv_C{1} = inv_A;

for k = 1:N
    B_k = [zeros(k-1, N); B(k, :); zeros(N-k, N)];
    %C{k+1} = C{k} + B_k; % unused
    g_k = 1/(1+trace(inv_C{k}*B_k));
    inv_C{k+1} = inv_C{k} - g_k * inv_C{k} * B_k * inv_C{k};
end

%B_N = [zeros(N-1, N); B(N, :)]; % last row of B
%g_N = 1/(1+trace(inv_C{N}*B_N));
%output = inv_C{N} - g_k * inv_C{N} * B_N * inv_C{N};
output = inv_C{end};


end