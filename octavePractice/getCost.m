

data = load('ex1data1.txt');
y = data(:, 2);

m = length(y); % number of training examples

X = [ones(m, 1), data(:,1)];

J = 0;

% J = computeCost(X, y, [-1 ; 2]); % I am assuming -1 is theta_1 and 2 is theta_2?

theta = [-1;2];

y_hat = X(:,2).*theta(2) .+theta(1)

diff = (y_hat .- y);

diff = diff .** 2;

rhs = sum(diff);

lhs = 1/((2)*length(y));

J = lhs*rhs
