function [f,g] = linear_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example.  y(j) is the target for example j.
  %
  
  m=size(X,2);
  n=size(X,1);
  J = @(t) .5 * sum((X' * t - y').^2);
  f = J(theta);
  g = X * (X' * theta - y');
  
  ti = randi(numel(theta));
  e1 = zeros(numel(theta),1);
  e1(ti) = 1e-4;
  gt = (J(theta + e1) - J(theta - e1)) / (2 * 1e-4);
  fprintf('Feature %d difference %0.4e\n', ti, g(ti) - gt);
%   g = zeros(size(theta));
%   pn = X' * theta - y';
%   for j=1:size(X,1)
%       g(j) = X(j,:) * pn;
%   end
  %
  % TODO:  Compute the linear regression objective by looping over the examples in X.
  %        Store the objective function value in 'f'.
  %
  % TODO:  Compute the gradient of the objective with respect to theta by looping over
  %        the examples in X and adding up the gradient for each example.  Store the
  %        computed gradient in 'g'.
  
%%% YOUR CODE HERE %%%
