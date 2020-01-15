function [ los_blockage_matrix, ceiling_reflection_blockage_matrix, in_range, reflection_in_range, is_strong_interferer] = checkInterfering(OriginLocation, NeighborLocation, para)
% interfering_matrix(i,j) indicate whether user i is blocked by user j,
% default maximum interfering distance is 10 meter
% 09/14/2015 base case: use polar form and relative location to compute the value
% 09/16/2015: consider reflection over the ceiling, user para to pass
% information
% checkInterfering(OriginLocation, NeighborLocation, UserRadius, varargin)
% to checkInterfering(OriginLocation, NeighborLocation, para)
% Revision: increase the area that users might interferer reflection to
% account for the fact that devices are on human body
% 12/25/2015: add parameter to decide using REAL reflection coefficient or FIXED reflection coeff 
field = {};
value = {};
field{1} = 'BodyR'; value{1} = 0.3;
field{2} = 'H'; value{2} = 2.8;
field{3} = 'h_device'; value{3} = 1.0;
field{4} = 'h_body'; value{4} = 1.75;
field{5} = 'MAX_RANGE'; value{5} = 10;
field{6} = 'CORRECTION_FACTOR'; value{6} = 1; % 0 for no correction in Revision; 1 for FULL correction
field{7} = 'REFLECTION_MODE'; value{7} = 'REAL'; % REAL for actual model, FIXED for fixed value
field{8} = 'REFLECTION_COEFFICIENT'; value{8} = 0.2166;
if ~isstruct(para) %YW: Went wrong before..., should be ~isstruct instead of isstruct
    para = struct();
end
for f_iter = 1: length(field)
    if ~isfield(para,field{f_iter})
        para.(field{f_iter}) = value{f_iter};
    end
end
RelativeLocation = NeighborLocation(:) - OriginLocation;
UserDist = abs(RelativeLocation);
UserTheta = angle(RelativeLocation);
NeighborNum = length(NeighborLocation);
los_blockage_matrix = zeros(NeighborNum);
ceiling_reflection_blockage_matrix = zeros(NeighborNum);
in_range = (UserDist(:) < para.MAX_RANGE);
% Check reflection is in range:
Reflection = reflection();
% Reflection.plotReflectionCoefficient();
theta = atan(UserDist./2./(para.H-para.h_device));
if strcmp(para.REFLECTION_MODE, 'REAL')
    [te, tm] = Reflection.getReflectionCoefficient(theta);
elseif strcmp(para.REFLECTION_MODE, 'FIXED')
    te = para.REFLECTION_COEFFICIENT;
    tm = te;
else
    error('REFLECTION_MODE must be REAL or FIXED');
    return;
end
PL = - 68 - 20*log10(2*abs(UserDist./2 + (para.H-para.h_device)*1i)) + 10*log10((abs(te) + abs(tm))./2);
PL0 = -68 - 20*log10(para.MAX_RANGE);
% reflection_in_range = zeros(NeighborNum, 1);
reflection_in_range = (PL(:)>PL0);
%The ratio of the distant: neighbors close with in ratio*UserDist to either typical user or interferer will block the reflection 
ReflectionAffectingRatio = (para.h_body - para.h_device)./(para.H - para.h_device);
corrected_dist = para.BodyR * para.CORRECTION_FACTOR;
for i = 1: NeighborNum
    theta_diff = UserTheta - UserTheta(i);
    flag_vector = [abs(UserDist.*sin(theta_diff))<= para.BodyR & cos(theta_diff)>0 & UserDist.*cos(theta_diff)<UserDist(i)];
    flag_vector(i) = 0;
    los_blockage_matrix(i,:) = flag_vector(:).';
    temp_dist_to_line = abs(UserDist.*sin(theta_diff));
    half_occupied_segment_length = sqrt(para.BodyR.^2 - temp_dist_to_line.^2).*(temp_dist_to_line<=para.BodyR);
    seg_min = UserDist.*cos(theta_diff) - half_occupied_segment_length;
    seg_max = UserDist.*cos(theta_diff) + half_occupied_segment_length;
    % Naive version
%     flag_vector = [temp_dist_to_line <= para.BodyR & cos(theta_diff)>0 & ((seg_min < ReflectionAffectingRatio*UserDist(i)/2) | (seg_max > UserDist(i)-ReflectionAffectingRatio*UserDist(i)/2)) & UserDist.*cos(theta_diff)<UserDist(i)];
    % 09/16/2015 Corrected to account for the fact that devices are on human surface
    flag_vector = [temp_dist_to_line <= para.BodyR & cos(theta_diff)>0 & ((seg_min < (ReflectionAffectingRatio*(UserDist(i) - 2*corrected_dist)/2 + corrected_dist)) | (seg_max > UserDist(i) - corrected_dist - ReflectionAffectingRatio*(UserDist(i) - 2*corrected_dist)/2)) & UserDist.*cos(theta_diff)<UserDist(i)];
    flag_vector(i) = 0;
    ceiling_reflection_blockage_matrix(i,:) = flag_vector(:).';
end
is_strong_interferer.LOS = (max(los_blockage_matrix,[],2) == 0) & in_range(:);
is_strong_interferer.NLOS = (max(ceiling_reflection_blockage_matrix, [], 2) == 0) & reflection_in_range(:);
is_strong_interferer.Total = is_strong_interferer.LOS | is_strong_interferer.NLOS;

end

