function [Q,R] = houseqr(A)

[m,n] = size(A);
I = eye(m);
Q = I;

for ii = 1:n
    
    if ii < m
        v = A(ii:end,ii)+norm(A(ii:end,ii))*I(ii:end,ii);
        P = I(ii:end,ii:end)-(2/(v'*v))*v*v'; 
        P = [eye(ii-1) zeros(ii-1,m-ii+1);zeros(m-ii+1,ii-1) P];

        Q = Q*P;

        A = P*A;
    end
    
end


R = A;

end

