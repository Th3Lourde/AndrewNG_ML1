function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
thetaColumns = length(theta);


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    printf("Inputed Parameters:\n");

    %X(1:3,:)
    theta;


    y_hat = X * theta;

    % diff = [(y_hat .- y) (y_hat .- y) (y_hat .- y)];

    z = y_hat .- y;

    diff = z * ones (1, thetaColumns);

    %diff(:,2) = diff(:,2) .* X(:, 2);
    %diff(:,3) = diff(:,3) .* X(:, 3);

    for param = 2:thetaColumns

      diff(:, param-1) = diff(:, param-1) .* X(:, param);

    end

    %diff(:, 2:thetaColumns) = diff(:,2:thetaColumns) .* X(:, 2:thetaColumns);

    rhs = sum(diff);

    lhs = alpha/m;

    movement = lhs.*rhs;

    movement = movement';

    theta = theta .- movement;

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
