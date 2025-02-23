clc
clear

%% model
a1= 0.1; 
B = chol([1.9608 0.0195; 0.0195 1.9605]);
D = 1;
n = size(B,1);
p = size(D,1);
B = [B zeros(n,p)];
D = [zeros(p,n) D];

sym_x = sym('a',[1,n],'real')';
f = [(cos(a1*sym_x(2)) - 1)  +  0.1*sym_x(1) + sym_x(2); 0.98*sym_x(2)];
h = ((sym_x(2))^2 - sym_x(2) -(sym_x(1))^2 + sym_x(1)) ;

%% initial
V0= 0.001*eye(n);
hatxn0=zeros(n,1);

%% data
N = 100;  %length of data to generate the lfm
xn = zeros(n,N+1); % nominal state
yn = zeros(p,N);   % nominal output
xn0= hatxn0+sqrtm(V0)*randn(n,1);
xn(:,1)=xn0;
w = randn(n+p,N);  
for t=1:N
    xn(:,t+1)= subs(f,sym_x,xn(:,t))+B*w(:,t);
    yn(:,t)  = subs(h,sym_x,xn(:,t))+D*w(:,t);
end

%% filter
% tolearace
c=10^-3;
Rx=REKF(hatxn0,yn,V0,B,D,c,N);

%% picture
figure
plot(Rx(1,:),'r'),hold on, plot(xn(1,:),'b')
