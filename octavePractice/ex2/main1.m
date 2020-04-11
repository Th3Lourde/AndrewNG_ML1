data = load('ex2data1.txt');



X = data(:, [1, 2]); y = data(:, 3);

[m, n] = size(X); % Useful variables concerning the dimensions of our data.
                  % But it isn't saved... How do we index this?
                  % So it turns out it is just stored as m,n
                  % This must be how octave handles functions that return
                  % multiple values. Happy I did the research :)


X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

yHat = costFunction(initial_theta, X, y);
