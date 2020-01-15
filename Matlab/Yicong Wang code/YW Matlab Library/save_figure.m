function [] = save_figure(ha,hf,W,H, my_title)
% 08/29/2014 Function to adjust figure size and save figure
% takes several parameters: ha,hf, W, H, title
% 09/02/2014: revise function to support multiple AX
% 01/11/2016: change font
% 12/03/2018: Add display info.
set(ha, 'FontName', 'Arial','FontSize', 11);
% set(ha, 'FontName', 'Arial');
N_axis = length(ha); % number of axis
for i = 1:N_axis
    set(ha(i),'units','centimeters');
end
set(hf,'units','centimeters');
% pos = get(ha,'Position');
ti = get(ha,'TightInset');
total_ti = zeros(N_axis,4);
if N_axis>1
    for i = 1:N_axis
        total_ti(i,:) = ti{i};
    end
else
    total_ti(1,:) = ti;
end
ti = max(total_ti,[],1);
ti(2) = ti(2) + 0.02;
for i = 1:N_axis
    set(ha(i),'Position',[ti(1),ti(2),W-ti(1)-ti(3), H-ti(2)-ti(4)]);
end
set(hf,'Position',[10,5,W,H]);
set(hf, 'PaperUnits','centimeters');
set(hf, 'PaperSize', [W H]);
set(hf, 'PaperPositionMode', 'manual');
set(hf, 'PaperPosition',[0 0 W H]);
% my_title = '../Writeup_figure/ratewhenscheduled.pdf';
disp(strcat('Paused. Press any key to continue saving figure to file: ', my_title));
pause;
saveas(hf,my_title);
end

