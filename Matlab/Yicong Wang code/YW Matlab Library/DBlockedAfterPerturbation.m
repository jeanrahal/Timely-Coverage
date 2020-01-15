function Prob = DBlockedAfterPerturbation(delta_max, d_blockage)
% Compute the equivalent width that some LOS link is blocked
% after pertubation
syms x theta r;
fun_blocking = @(r, theta, x) ((r.*cos(theta) < (d_blockage./2 - x))) & ((r.*cos(theta)> (- d_blockage - x)));
fun_prob_notblocking = @(x) dblquad(@(r, theta) fun_blocking(r, theta, x).*r./pi./ delta_max.^2, 0, delta_max, 0, 2*pi);
Prob = 2* quadv(@(x) fun_prob_notblocking(x), d_blockage./ 2, delta_max + d_blockage./2);
end

