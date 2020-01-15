function [] = set_page( hf, W, H )
% Set page of figure, units in Centimeters
set(hf, 'Units', 'Centimeters', 'Position', [0, 0, W, H], 'PaperUnits', 'Centimeters', 'PaperSize', [W, H]);
end

