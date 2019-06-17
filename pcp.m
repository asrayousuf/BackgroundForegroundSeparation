function [L, S, res] =  pcp(X, maxiter, k)
    [m,n] = size(X);
    trans = m<n;
    if trans
        X = X.T;
        [m,n] = X.shape;
    end
    lamda = 1/sqrt(m);
    op_norm = norm_op(X) ;
    Y = X / max(op_norm, (norm( X, inf) / lamda));
    mu = k * (1.25/ op_norm); 
    mu_bar = mu * 1e7; 
    rho = k * 1.5;

    d_norm =norm(X, 'fro');
    L = zeros(m, n);
    sv = 1;

    res = [];
    for i = 1:maxiter
        %fprintf("\n rank sv: %f", sv);
        X2 = X + Y/mu;    
        % Update the sparse matrix S by shrinking: original - low-rank
        S = shrink(X2 - L, lamda/mu);

        % update the low-rank matrix using truncated SVD
        [L, svp] = svd_reconstruct(X2 - S, sv, 1/mu);

        if svp < sv 
            a = 1;
        else
            a = round(0.05*n);
        end
        sv = svp + a;

        % Calculate Residual
        Z = X - L - S;
        Y = Y + (mu*Z);
        mu = mu * rho;

        res = [res ; S(140,:) ; L(140,:)];

        if m > mu_bar
             m = mu_bar;
        end
         if converged(Z, d_norm)
             %fprintf("\nResults converged\n");
             break;
         end
    end    

    if trans
      L=L.T; 
      S=S.T;
    end

function res = norm_op(M)
     k=2;
    [~, S, ~]= random_SVD(M,k);
    %[U, S, V]= svd(M, 'econ');
    s = diag(S);

    res = s(1);

function [L, svp] = svd_reconstruct(M, rnk, min_sv) 
    k=2;
    [U, S, V]= random_SVD(M,k);
    %[U,S,V] = svd(M,'econ'); %%%
    s  = diag(S);
    s = s - min_sv;
    tt      = s > 0;
    U       = U(:,tt);
    S       = diag(s(tt));
    V       = V(:,tt);
    svp     = nnz(tt);
    L= U * S * V'; 

    

function [S] = shrink(M, tau)
    S = sign(M) .* max(0, abs(M) - abs(tau));

function [res] = converged(Z, d_norm)
    TOL=1e-9;
    err = norm(Z, 'fro') / d_norm;
    %fprintf('\n error: %f', err);
    res = err < TOL;
      
  