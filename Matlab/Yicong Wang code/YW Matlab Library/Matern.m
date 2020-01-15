% Generte Nodes' location based on Mater Process
function [ans_location] = Matern(varargin)
% 09/14 2015 revise the process for adding new processes, avoiding generate
% points overlapping with center or existing points
% 09/14/2015 added to YW Matlab Library
% 05/22/2015 [location] = Matern(AREA, N, GAP)
% [location] = Matern(AREA, N, GAP, 'Center') place a node at center,
% location(1)
% [location] = Matern(AREA, N, GAP, Points) place Points in the plane
% before placing other points, Points are first placed in location
% location is in 2-D complex plane
if nargin > 4
    error('YW: Matern.m Too many arguments');
    return;
end
AREA = varargin{1};
N = varargin{2};
GAP = varargin{3};
Points = [];
MAX_TRIAL_FACTOR = 200; % Max times of trying to find new location
if nargin == 4
    if ischar(varargin{4})
        if lower(varargin{4}) == 'center'
            Points = 0.5*(AREA(2) + AREA(1)) + 1i*0.5*(AREA(4)+AREA(3));
        else
            error(['YW: Matern.m Unkown option ', varargin{4}]);
        end
    elseif ~isnumeric(varargin{4})
        error('YW: Matern.m Unkown option', varargin{4});
    else
        Points = varargin{4};
        temp = size(Points);
        if length(size(Points))>2
            error('YW: Matern.m Points invalid');
        end
        if(temp(1)~=1 && temp(2)~=1)
            error('YW: Matern.m Points invalid');
        end
    end
else
    Points = rand(1)*(AREA(2)-AREA(1)) + rand(1)*(AREA(4) - AREA(3))*1i + AREA(1) + 1i*AREA(3);
end
if (N <= length(Points))
    ans_location = Points(1:N);
    return;
end
ans_location = zeros(N,1);
ans_location(1:length(Points)) = Points(:);

location = rand(N*2,1)*(AREA(2)-AREA(1)) + rand(N*2,1)*(AREA(4) - AREA(3))*1i + AREA(1) + 1i*AREA(3);
location(1:length(Points)) = Points(:); % 09/14/2015 YW 
distance = tril(abs(repmat(location,1,2*N) - repmat(location.',N*2,1)),-1) + triu(inf.*ones(N*2));
j = length(Points) + 1;
% figure,axis(AREA),hold on;

for i = j:2*N
    if min(distance(i,:))>GAP
        ans_location(j) = location(i);
%         plot(real(ans_location(j)),imag(ans_location(j)),'bo');
        j = j+1;
        if j>N
            break;
        end
    else
        distance(:,i) = inf;
    end
end

trial_num = 1;
max_trial = MAX_TRIAL_FACTOR*N;
while j <= N && trial_num < max_trial
    trial_num = trial_num + 1;
    new_location = rand(1)*(AREA(2)-AREA(1)) + rand(1)*(AREA(4) - AREA(3))*1i + AREA(1) + 1i*AREA(3);
    if min(abs(ans_location(1:j-1) - new_location)) > GAP
        ans_location(j) = new_location;
        j = j+1;
    end
end
if trial_num == max_trial
    error('Matern.m fails to generate the points, try smaller user number');
end

% end