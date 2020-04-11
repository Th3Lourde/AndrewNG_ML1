function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of elements of theta

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
%J = (1/m) * sum(cost);

rhs = theta(2:n) .^ 2;
rhs = ((lambda)/(2*m))*sum(rhs);
J = (1/m)*sum(cost) + rhs; % Everthing should be a scalar by this point.
% -------------------------------------------------------------
% Calculating the gradient:
thetaColumns = length(theta);
yHat = sigmoid(X * theta);
z = yHat .- y;

diff = z * ones(1, thetaColumns);

% grad = grad .+ z(:)

for param = 1:thetaColumns
  if param == 1
    diff(:, param) = diff(:, param) .* X(:, param);
    
  elseif param > 1
    diff(:, param) = diff(:, param) .* X(:, param) .+ ((lambda)/(m)) .* theta(param);
    
  endif
end
%
diff = (1/m)*sum(diff);
%diff(2:thetaColumns) = diff(2:thetaColumns) .+ (lambda/m) .* diff(2:thetaColumns);
%
grad = diff';



% =============================================================

end
