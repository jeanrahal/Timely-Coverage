% 09/16/2015: test the performance of reflection class in YW Matlab Library
clear all;
r = reflection();
% r.RefractiveIndex
% r.WaveLength
% r.Thickness
% r.setRefractiveIndex(1.5-0.01i)
% r.getTE(0.1)
% r.getTM(0.1)
% [te, tm] = r.getReflectionCoefficient(0.1)
% r.getTE([0.1,0.2])
% r.getTM([0.1,0.2])
% [te, tm] = r.getReflectionCoefficient([0.1,0.2])
r.plotReflectionCoefficient();
save_figure(gca, gcf, 12, 10, '');
d = 0.6:0.1:10;
H = 2.8;
h_device = 1.0;
theta = atan(d./2./(H-h_device));
[te, tm] = r.getReflectionCoefficient(theta);
% te = ones(size(te))*0.31623;
% tm = te;
% PL = - 68 - 20*log10(2*abs(d./2 + (H-h_device)*1i)) + 10*log10((abs(te) + abs(tm))./2);
% PL = - 68 - 20*log10(2*abs(d./2 + (H-h_device)*1i)) + 10*log10((abs(te).^2 + abs(tm).^2)./2);
h = zeros(1,2);
figure, h(1) = plot(d, PL,'b-'); hold on, grid on, box on;
h(2) = plot(d, ones(size(d)).*(-88),'r--' );
legend('Reflected Channel PL','Threshold(PL_{LOS} at 10m)');
xlabel('Distance to Origin(m)','FontSize',12.0);
ylabel('PathLoss (dB)','FontSize',12.0);
set(h,'LineWidth', 2.0);
set(gca, 'FontSize', 12.0);
title(['PL of Reflected Signal',' Height=', num2str(H)]);
save_figure(gca, gcf, 12, 10, '');

