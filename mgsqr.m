function [Q, R] = mgsqr(A)

    [m,n] = size(A);
    R = zeros(n);
    Q = zeros(m,n);

    for ii = 1:n
        R(ii,ii) = norm(A(:,ii));
        Q(:,ii) = A(:,ii) / R(ii,ii);
        for j = ii+1:n
            R(ii,j) = Q(:,ii)' * A(:,j);
            A(:,j) = A(:,j) - R(ii,j) * Q(:,ii);
        end
    end
end