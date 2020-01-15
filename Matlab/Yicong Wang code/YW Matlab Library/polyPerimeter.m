function [ L ] = polyPerimeter(X, Y)
% 09/15/2015: return the perimeter of the polyarea
X = [X(:);X(1)];
Y = [Y(:);Y(1)];
L = sum(abs(X(2:end)-X(1:end-1) + 1i*(Y(2:end) - Y(1:end-1))));


end

