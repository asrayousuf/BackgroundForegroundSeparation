function [U,S,V] = mysvd(A,tol)
   
%reserve space in advance
[m,n]=size(A);
loopmax=m;
loopcount=0;

% or use Bidiag(A) to initialize U, S, and V
U=eye(m);
S=A';
V=eye(n);
err=10;

while err>tol && loopcount<loopmax
    
    [q,S]=qr(S'); U=U*q;
    [q,S]=qr(S'); V=V*q;

    err=norm(triu(S,1))/norm(diag(S));
    loopcount=loopcount+1;
    disp(["err=",err,"and loopcount=",loopcount])
end

%fix the signs in S
U = U.*sign(diag(S))';
S=[abs(diag(diag(S))),zeros([m,n-m])];

end
