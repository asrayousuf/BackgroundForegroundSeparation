function [U,S,V]= random_SVD(A,k)
[m,n] = size(A);
p = 0;
Omega = randn([n,k+p]);
Y = A*Omega;

% To increase the decay of the S diag to further improve the approxmation
Y = A*(A'*Y);
Y = A*(A'*Y);
[Q,~] = mgsqr(Y);
[U,S,V] = svd(Q'*A);
U = Q*U(:,1:k);
S = S(1:k,:);
A_m=U*S*V';
