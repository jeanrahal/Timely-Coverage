function [ Prob ] = ProbFacingAfterRotation(omega, dist)
% 05/18/2015 This function compute the probability that tx is still facing
% rx after rotation
syms theta1 theta2;
fun_prob = @(theta1, theta2) (1 - min(abs(theta1), 2*pi - omega)./omega).*(1 - min(abs(theta2),2*pi - omega)./ omega).*dist(theta1).*dist(theta2);
Prob = dblquad(@(theta1, theta2) fun_prob(theta1, theta2), - pi, pi, -pi, pi);

end

