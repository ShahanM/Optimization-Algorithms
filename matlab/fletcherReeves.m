% Example implementation of the Fletcher-Reeves method of optimization.

syms x y; % Define number of symbolic variabls in the function
sym_vars = [x, y]; % An array of symbolic variables
epsilon = 10^-3; % Error tolerance for norm of gradient
f = (x - 2)^4 + (x - 2*y)^2; % Function defition using symbolic variables
ini_pnt = [0; 3];  % Initial starting point
[k, x_final] = fr(f, sym_vars, ini_pnt, epsilon);

function[k, x_k] = fr(f, sym_vars, ini_pnt, epsilon)
    x_k = ini_pnt;
    k = 0;
        
    % Convenient size call to reshapce to use substitute method
    [m, ~] = size(x_k);
    
    % Compute symbolic Gradient
    grad = gradient(f);
    grad_pnt = double(subs(grad, sym_vars, reshape(x_k, 1, m)));
    
    % Just initializing for our linesearch loop.
    lin_cond = inf;
    fk = double(subs(f, sym_vars, reshape(x_k, 1, m)));
    
    % Initialize our line search parameters
    lambda = 1;
    tau = 0.6;
    
    v_k = -grad_pnt;
    
    % The main iteration loop
    while norm(grad_pnt) > epsilon
        grad_pnt = double(subs(grad, sym_vars, reshape(x_k, 1, m)));
        
        % Line search minimizer, this is very slow.
        % TODO refactor our to try different line search algorithms
        while lin_cond >= fk
            condition = x_k + lambda*v_k;
            lin_cond = double(subs(f, sym_vars, reshape(condition, 1, m)));
            fk = double(subs(f, sym_vars, reshape(x_k, 1, m)));
            lambda = tau * lambda;
        end
        x_k = x_k + lambda*v_k;
        grad_pnt_k = double(subs(grad, sym_vars, reshape(x_k, 1, m)));

        alpha = (grad_pnt_k.'*grad_pnt_k)/(grad_pnt.'*grad_pnt);
        v_k = -grad_pnt_k + alpha*v_k;
                
        fprintf("Iteration k = %d \n", k);
        disp(x_k.');
        k = k + 1;
    end
end