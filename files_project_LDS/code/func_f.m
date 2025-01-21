function [z] = func_f(sym_x)
%% f(x)
z= [(cos(0.1*sym_x(2)) - 1)  +  0.1*sym_x(1) + sym_x(2); 0.98*sym_x(2)];
end

