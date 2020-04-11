
fprintf('Plotting Data ...\n');
data = load('ex1data1.txt');
X = data(:, 1); 
y = data(:, 2);
% m = length(y); % number of training examples

% x = [1; 2; 3; 4];
% y = [3; 4; 5; 6];

% x = linspace(0,1,6);

% y = x.^ 2;

c = X .* y;
scatter(X,y, 50, c, "filled");

xlabel("Population");
ylabel("Profit");

title ("Population/Profit Scatter Plot");

%c = X .* y;
%scatter (x, y, 50, c, "filled");
%title ("scatter() with colored filled bubbles");

%plot(x,y,’r-x’);

% t = typeinfo (X);

% fprintf()

% fprintf('X is :{}\n', X);
% fprintf('y is :\n');


% fprintf('With theta = [0 ; 0]\nCost computed = %f\n', J);

% scatter(x,y);
