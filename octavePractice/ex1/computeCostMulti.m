

% [X mu sigma] = featureNormalize(X);

% Add intercept term to X
% X = [ones(m, 1) X];







function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% Don't actually know if this is write b/c theta is zero.
% Assume that it is correct.

% Y = X .* theta'

% J = 5;

% middle = (X * theta) - y;
%
% rhs = middle' * middle;

J = 1/(2*m) * ((X * theta - y)'*(X * theta - y));

% y_hat = (X(:,2) .*theta(3)) .+ (X(:,2) .*theta(2)) .+ theta(1);
%
% diff = y_hat .- y



% =========================================================================

end
