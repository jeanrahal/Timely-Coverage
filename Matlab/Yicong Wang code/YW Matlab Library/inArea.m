function [result] = inArea(location, AREA)
% Check whether points are within range
    result = (imag(location) >= AREA(3)) & (imag(location) <= AREA(4)) & ...
             (real(location) >= AREA(1)) & (real(location) <= AREA(2));
end

