function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % y_hat = X.*theta(2) .+theta(1);
    %
    % diff = (y_hat .- y).*X;
    %
    % rhs = sum(diff);
    %
    % lhs = alpha/m;
    %
    % movement = lhs.*rhs;
    %
    % theta = theta .- movement;

% ==================================================== %
    y_hat = X(:,2).*theta(2) .+theta(1);

    % diff = [(y_hat .- y) (y_hat .- y)]

    diff = [(y_hat .- y) (y_hat .- y)];

    diff(:,2) = diff(:,2) .* X(:, 2);

    % diff(1, :)
    % diff(1, 2);

    %diff = [(y_hat .- y), (y_hat .- y).*X]

    rhs = sum(diff);

    diff(1) = 1;

    %diff

    lhs = alpha/m;

    movement = lhs.*rhs;

    movement = movement';

    theta = theta - movement;




    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

end

end
