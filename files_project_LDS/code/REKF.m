function [Xrekf,V]=REKF(x0,y,V0,B,D,c,T)
% Robust EKF 
%
% Input:
%       x0: Initial condition x(1|0)
%       V0 : Initial condition variance P(1|0)
%       B  : Matrix B of the noise inputs
%       D  : Matrix D of the noise inputs
%       y  : Measured data
%       T  : Length of the dataset, and so the length of the iterations

% Outputs:
%       Xrekf: state predcited   x(t+1|t)
%       V:     state erro performance V(t+1|t)


%% Noice covariance
Q=B*B';
R=D*D';

%% SAVE
n=size(Q,1);
p=size(R,1);
Xrekf=zeros(n,T+1);
Xrekf(:,1)=x0;
Xn=zeros(n,T);
V=zeros(n,n,T+1);
V(:,:,1)=V0;
A=zeros(n,n,T);
C=zeros(p,n,T);
G=zeros(n,p,T);
th=zeros(1,T);


%% Iteration
for i=1:T
    %C_t
    C(:,:,i)=func_hl(Xrekf(:,i));
    %L_t
    L=V(:,:,i)*C(:,:,i)'*inv(C(:,:,i)*V(:,:,i)*C(:,:,i)'+R); 
    %h(\hat x_t)
    hn= func_h(Xrekf(:,i));
    %\hat x_t|t
    Xn(:,i)=Xrekf(:,i)+L*(y(:,i)-hn);
    %A_t
    A(:,:,i)=func_fl(Xn(:,i));
    %G_t
    G(:,:,i)=A(:,:,i)*L;
    %\hat x_t+1
    Xrekf(:,i+1)=func_f(Xn(:,i));    
    %P_t+1
    P=A(:,:,i)*V(:,:,i)*A(:,:,i)'-A(:,:,i)*V(:,:,i)*C(:,:,i)'*inv(C(:,:,i)*V(:,:,i)*C(:,:,i)'+R)*C(:,:,i)*V(:,:,i)*A(:,:,i)'+Q;
    %th_t
    th(i) = theta(P,c,n);
    %V_t+1
    V(:,:,i+1)=inv(inv(P)-th(i)*eye(n));
end

