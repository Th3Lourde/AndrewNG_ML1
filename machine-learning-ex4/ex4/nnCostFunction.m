function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ones(m, 1) X];

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
preds = zeros(size(y), num_labels);

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Get the prediction
l1 = sigmoid(Theta1*X');
l1 = [ones(1, size(l1)(2)); l1];
l2 = sigmoid(Theta2*l1);
yHat = l2';
preds = yHat;
% [x, xi] = max(yHat);
% yHat = xi';

actual = zeros(size(preds));

% size(yHat)
% size(y)
size(preds);
size(actual);

% size(yHat)
% yHat(1:5, :)


for i = 1:size(yHat)(1)
  % preds(i, yHat(i)) = 1;
  actual(i, y(i)) = 1;
endfor

% size(preds)
% size(actual)
% actual(1:5, :)

% actualT = actual(1:10, :);
% predT = preds(1:10, :);

% size(yHat)
% size(y)



% ans = -1*actualT.*log(predT)-(1-actualT).*log(1-predT);
ans = -1*actual.*log(preds)-(1-actual).*log(1-preds);
% ans = -1*y.*log(yHat)-(1-y).*log(1-yHat);
% ans(3);
ansNoReg = (1/m)*sum(sum(ans));
J = ansNoReg;

% reg = (lambda/(2*m))*sum(theta(2:size(theta)(1)) .^ 2);

% J = ansNoReg + reg;


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
