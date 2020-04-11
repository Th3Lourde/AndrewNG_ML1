data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;


theta = ones(size(X,2),1);
lambda = 10
% [cost, grad] = costFunctionReg(test_theta, X, y, 10);




m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
% Cost function
yHat = sigmoid(X * theta);
cost = ((-1) .* y .* log(yHat)) - ((1 .- y) .* log(1 .- yHat));
J = (1/m) * sum(cost);

length(theta)

rhs = theta .^ 2;
lhs = ((lambda)/(2*m))
rhs = lhs*sum(rhs);
J = J + rhs % Everthing should be a scalar by this point.
printf("Want 3.16\n");
% -------------------------------------------------------------
