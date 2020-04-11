function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

%printf(length(X));
% Ok, let's just use the given implementation.
% We came up with the right idea to plot two separate graphs.
% Let's just focus on the ML portion.



% scatter(X(:,1), X(:,2));

% data = load('ex2data1.txt');
% X = data(:, [1, 2]); y = data(:, 3);
%
pos = find(y== 1); neg = find(y == 0);

plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, ...
'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', ...
'MarkerSize', 7);

% length(X)
%
% accepted = [];
% rejected = [];
%
%
% for iter = 1:length(X)
%
% end



% gotIn = (y>=1);
% Create 2 copies of y
% in = y;
% out = y;

% in(gotIn) = NaN;

% out(~gotIn) = NaN;

% '''
% I think that brute force is the way to go here :/
% '''

% scatter(X,in,'+',X,out,'o',y)

% scatter(X(:,1), X(:,2), in, '+', X(:,1), X(:,2), in, 'o')
% scatter(in, '+', out, 'o')


% =========================================================================



hold off;

end
