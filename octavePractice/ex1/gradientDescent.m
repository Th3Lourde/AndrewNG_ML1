% function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

% theta = gradientDescent(X, y, theta, alpha, iterations);

data = load('ex1data1.txt');
%X = data(:, 1);

y = data(:, 2);

m = length(y); % number of training examples

X = [ones(m, 1), data(:,1)];

iterations = 1500;
num_iters = 1500;
alpha = 0.01;
theta = zeros(2, 1);


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % cost = getCostF(X, y, theta)

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
    J_history(iter) = getCostF(X, y, theta);

end

theta
J_history(num_iters)
