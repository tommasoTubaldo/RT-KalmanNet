function [tt] = theta(P_pred,c,n)
% c: tolerance
% P_pred:  P_{t+1|t}   
% tt: \theta_t

value=1;
t1=0;
e = eig(P_pred);
r = max(abs(e));
t2=(1-10^-5)*(r)^-1;
while abs(value)>=10^-9
    tt=0.5*(t1+t2);
    value=trace(inv(eye(n)-tt*P_pred)-eye(n)) + log(det(eye(n)-tt*P_pred))-c;
    if value>0
        t2=tt;
    else
        t1=tt; %tt:  
    end
end
end

