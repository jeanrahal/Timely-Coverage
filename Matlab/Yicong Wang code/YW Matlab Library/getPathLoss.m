% 04/06/2016: compute the free space path loss given the distance
% distance is in meter, by default
function [pl, pl_db] = getPathLoss(dist, varargin)
    freq = 60e9; % frequency, default value is 60GHz
    C = 3e8; % speed of light, 3e8 m/s
    if nargin>0
        num = 1;
        while num < nargin
            if num + 1 == nargin
                break;
            elseif strcmp(varargin{num}, 'freq')
                freq = varargin{num+1};
            end
            num = num + 2;
        end
    end
    pl = 1./(4.*pi.*dist.*freq./C).^2;
    pl_db = 20.*log10(dist) + 20.*log10(freq) + 20.*log10(4.*pi./C);
    pl_db = - pl_db;
end