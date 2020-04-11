function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta = theta;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    y_hat = X * theta;

    diff = [(y_hat .- y) (y_hat .- y) (y_hat .- y)];

    diff(:,2) = diff(:,2) .* X(:, 2);
    diff(:,3) = diff(:,3) .* X(:, 3);

    rhs = sum(diff);

    lhs = alpha/m;

    movement = lhs.*rhs;

    movement = movement';

    theta = theta - movement;

    % ============================================================

    % Save the cost J in every iteration
    % computeCostMulti(X, y, theta);
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
