data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

% Choose some alpha value
alpha = 0.01;
num_iters = 400;

% Init Theta and Run Gradient Descent
theta = zeros(3, 1);
% theta = [1;1;1];
% J_history = gradientDescentMulti(X, y, theta, alpha, num_iters);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

thetaActual = normalEqn(X, y)

1650*theta(1) + 3*theta(2)
