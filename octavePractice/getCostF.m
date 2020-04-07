function J = getCostF(X, y, theta)
J = 0;

m = length(y); % number of training examples

y_hat = X(:,2).*theta(2) .+theta(1);

diff = (y_hat .- y);

diff = diff .** 2;

rhs = sum(diff);

lhs = 1/((2)*length(y));

J = lhs.*rhs;

end
