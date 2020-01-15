function [ width, area , fun_avgWidth, fun_TX, fun_RX] = WidthTXRXPerturbation(d_blockage, d_perturb, pdf_perturb, perturb_range)
% 05/17/2015 This script compute the equivalent ADDITIONAL width of the
% area where no user should present in case the Strong Interferer (SI) remains
% SI after TX and RX (also other nodes) are perturbed
% d_perturb is the equivalent range when perturbed, d_perturb >= d_blockage
syms r1 theta1 r2 theta2;
% WidthHelper = @(d1, d2) 0 + (d1>0 & d2>0).*(d1+d2)./2 + (d1<0 & d2<0).*0 + (d1>0& d2<0).*d1.^2./2./(d1-d2) + (d2>0 & d1<0).*d2.^2./2./(d2-d1);
fun_avgWidth = @(r1, theta1, r2, theta2) WidthHelper(r1.*cos(theta1) + d_perturb./2 - d_blockage./2, r2.*cos(theta2) + d_perturb./2 - d_blockage./2) + ...
    WidthHelper(-(r1.*cos(theta1) - d_perturb./2 + d_blockage./2), -(r2.*cos(theta2) - d_perturb./2 + d_blockage./2));
fun_TX_theta = @(r1, theta1, r2) quadv(@(theta2) fun_avgWidth(r1, theta1, r2, theta2).*pdf_perturb(r2, theta2).*r2, 0, 2*pi);
fun_TX = @(r1, theta1) quadv(@(r2) fun_TX_theta(r1, theta1, r2), 1e-6, perturb_range);
% dblquad(@(r2, theta2) fun_avgWidth( r1, theta1, r2, theta2).*pdf_perturb(r2, theta2).*r2, 0.001, perturb_range, 0, 2*pi);
% width = dblquad(@(r1, theta1) fun_TX(r1, theta1)./pi ./ perturb_range.*2.*r1, 0.5, perturb_range, 0., 2*pi);

fun_RX = @(r1) quadv(@(theta1) fun_TX(r1, theta1).*pdf_perturb(r1, theta1).*r1, 0, 2*pi);
width = quadv(@(r1) fun_RX(r1), 1e-6, perturb_range);

end

