% Example implementation of the Davidon-Fletcher-Powell method of
% optimization.

syms x y; % Define number of symbolic variabls in the function
sym_vars = [x, y]; % An array of symbolic variables
epsilon = 10^-3; % Error tolerance for norm of gradient
f = (x - 2)^4 + (x - 2*y)^2; % Function defition using symbolic variables
ini_pnt = [0; 3];  % Initial starting point
[k, x_final] = dfp(f, sym_vars, ini_pnt, epsilon);

function[k, x_k, H] = dfp(f, sym_vars, ini_pnt, epsilon)
    x_k = ini_pnt;
    k = 0;
    % Compute symbolic Gradient
    grad = gradient(f);
    [~, n] = size(grad);
    grad_pnt = inf;
    H = eye(n);
    % Convenient size call to reshapce to use substitute method
    [m, ~] = size(x_k);
    
    % Just initializing for our linesearch loop.
    lin_cond = inf;
    fk = double(subs(f, sym_vars, reshape(x_k, 1, m)));
    
    % Initialize our line search parameters
    alpha = 1;
    tau = 0.6;
    
    % The main iteration loop
    while norm(grad_pnt) > epsilon
        grad_pnt = double(subs(grad, sym_vars, reshape(x_k, 1, m)));
        p = -H*grad_pnt;
        
        % Line search minimizer, this is very slow.
        % TODO refactor our to try different line search algorithms
        while lin_cond >= fk
            condition = x_k + alpha*p;
            lin_cond = double(subs(f, sym_vars, reshape(condition, 1, m)));
            fk = double(subs(f, sym_vars, reshape(x_k, 1, m)));
            alpha = tau * alpha;
        end
        
        s = alpha*p;
        x_k = x_k + s;
        grad_pnt_k = double(subs(grad, sym_vars, reshape(x_k, 1, m)));
        y = grad_pnt_k - grad_pnt;
        
        A = (H*(y*y.')*H)/(y.'*H*y);
        B = (s*s.')/(s.'*y);
        H = H - A + B;
        
        fprintf("Iteration k = %d \n", k);
        disp(x_k.');
        k = k + 1;
    end
end