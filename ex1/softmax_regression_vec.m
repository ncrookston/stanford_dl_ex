function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);
 
  theta=reshape(theta, n, []);
  theta=[theta zeros(size(theta,1),1)];
  num_classes=size(theta,2);

  etx = exp(theta' * X);
  yeye = eye(10);
  yos = yeye(:,y);
  f = -sum(yos .* log(etx ./ sum(etx,1)), 'all');
  g = -X * (yos - etx ./ sum(etx, 1))';
  g = reshape(g(:,1:end-1),[],1);

