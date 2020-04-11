% function [J, grad] = costFunction(theta, X, y)
% function [cost] = costFunction(theta, X, y)

%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% data = load('ex2data1.txt');
% X = data(:, [1, 2]); y = data(:, 3);

% [m, n] = size(X); % Useful variables concerning the dimensions of our data.
%                   % But it isn't saved... How do we index this?
%                   % So it turns out it is just stored as m,n
%                   % This must be how octave handles functions that return
%                   % multiple values. Happy I did the research :)

% Add intercept term to x and X_test
% X = [ones(m, 1) X];

% Initialize fitting parameters
% initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
% [cost, grad] = costFunction(initial_theta, X, y);
data = load('ex2data1.txt');



X = data(:, [1, 2]); y = data(:, 3);

[m, n] = size(X); % Useful variables concerning the dimensions of our data.
                  % But it isn't saved... How do we index this?
                  % So it turns out it is just stored as m,n
                  % This must be how octave handles functions that return
                  % multiple values. Happy I did the research :)


X = [ones(m, 1) X];

% Initialize fitting parameters
theta = zeros(n + 1, 1);

% Initialize some useful values
m = length(y); % number of training examples

cost = zeros(m);
% grad = zeros(size(theta));

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% Cost function
yHat = sigmoid(X * theta);
cost = ((-1) .* y .* log(yHat)) - ((1 .- y) .* log(1 .- yHat));
J = (1/m) * sum(cost);
% -------------------------------------------------------------
% Calculating the gradient:
thetaColumns = length(theta);
yHat = sigmoid(X * theta);
z = yHat .- y;

diff = z * ones (1, thetaColumns);

% grad = grad .+ z(:)

for param = 2:thetaColumns
  diff(:, param) = diff(:, param) .* X(:, param);
end
%
diff = (1/m)*sum(diff);
%
grad = diff';





% =============================================================

% end
