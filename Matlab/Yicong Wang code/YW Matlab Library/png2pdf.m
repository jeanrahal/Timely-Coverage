% 11/11/2015 Convert png file to pdf file and change size accordingly
function [ output_args ] = png2pdf(filename)
%PNG2PDF read image file and save to pdf
imshow(imread(filename, 'png','BackgroundColor',[1,1,1]));
set(gca,'position',[0 0 1 1],'units','normalized')
hf = gcf;
set(hf, 'Units','Inches');
pos = get(hf,'Position');
set(hf,'PaperUnits','Inches','PaperSize',[pos(3), pos(4)],...
    'PaperPositionMode','manual', 'PaperPosition',[0,0,pos(3), pos(4)]);
end

