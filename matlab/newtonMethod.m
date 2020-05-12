% Example implementation of the Newton method of optimization.

syms x y; % Define number of symbolic variabls in the function
sym_vars = [x, y]; % An array of symbolic variables
epsilon = 10^-3; % Error tolerance for norm of gradient
f = (x - 2)^4 + (x - 2*y)^2; % Function defition using symbolic variables
ini_pnt = [1; 1]; % Initial starting point
% Call to our Neuton method
[k, x_final] = newton(f, sym_vars, ini_pnt, epsilon);

% Function to run an n-variable Newton Method
function[k, x_k] = newton(f, sym_vars, ini_pnt, epsilon)
    x_k = ini_pnt;
    k = 0;
    % Compute symbolic Hessian
    hess = hessian(f, sym_vars);
    % Compute symbolic Gradient
    grad = gradient(f);
    grad_pnt = inf;
    % Convenient size call to reshapce to use substitute method
    [m, ~] = size(x_k);
    
    % The main iteration loop
    while norm(grad_pnt) > epsilon
        % Compute the Hessian at a given point x_k
        hess_pnt = double(subs(hess, sym_vars, reshape(x_k, 1, m)));
        % Compute the Gradient at a given point x_k
        grad_pnt = double(subs(grad, sym_vars, reshape(x_k, 1, m)));
        x_k = x_k - hess_pnt\grad_pnt;
        fprintf("Iteration k = %d \n", k);
        disp(x_k);
        k = k + 1;
    end
end