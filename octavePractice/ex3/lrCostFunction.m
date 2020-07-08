function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% 1) Compute the cost

yHat = sigmoid(X*theta);
y;

left = -1*y;;
leftRight = log(yHat);
left = -1*y.*log(yHat);

right = (1-y);
rightLeft = log(1-yHat);
right = (1-y).*log(1-yHat);

ans = -1*y.*log(yHat)-(1-y).*log(1-yHat);
ansNoReg = (1/m)*sum(ans);

reg = (lambda/(2*m))*sum(theta(2:size(theta)(1)) .^ 2);

J = ansNoReg + reg;





% 2) Compute the gradient
gradientsBase = (yHat-y);

% Now we add in the x^i_j
for j = 1:size(theta)(1)
  grad(j) = (1/m)*sum(gradientsBase.*X(:,j));

  if j > 1
    grad(j) = grad(j)+(lambda/m)*theta(j)
  endif

% =============================================================

grad = grad(:);

end
