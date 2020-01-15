% 07/30/2017 Plot figure illustrating analytical model and simulation model
addpath('Yicong Wang code');
close all;
clear all;
clc;

sample_gap = 0.2;
color_sensor = [0, 100,0]./255; % sensor color is dark green
color_sensed = [132, 255, 132]./255; % sensed region is ligh green
color_blockage = [250, 0, 0]./255; % blockage color is red
color_blocked = [255, 132, 132]./255; % blocked region

plot_analytical = false;
plot_simulation = true;
plot_model = true;

%% 1. Analytical Model
if plot_analytical
    density_object = 0.01;
    display(['density: ', num2str(density_object)]);
    if plot_model
        penetration = 0;
    else
        penetration = 0.15;
    end
    display(['penetration: ', num2str(penetration)]);
    region_radius = 60;
    r_object = 1.67; % radius of objects
    r_sensing = 100; % max sensing range
    if plot_model
        region = [0,60,0,24];
    else
        region = region_radius.*[-1, 1, -1, 1];
    end
        
    area = abs((region(2) - region(1))*(region(4) - region(3)));
    display(['sensing range: ', num2str(r_sensing), ...
             ' object radius: ', num2str(r_object)]);
    
    num_object = poissrnd(density_object*area);
    locations_object = rand(num_object, 1)*(region(2) - region(1)) + region(1) ...
                       + 1i*(region(3) + rand(num_object, 1)*(region(4) - region(3)));
    if plot_model
        % plot sensing in analytical model 
        locations_object(1) = (30+12i);
    end
    
    locations_vec_object = complex2vector(locations_object);
    object_rand_value = rand(num_object, 1);
    
    % plot model
    if plot_model
        object_rand_value(1) = 0;
    end
    sensor_set = find(object_rand_value <= penetration);
    blockage_set = find(object_rand_value > penetration);
    
    
    temp_location_matrix = repmat(locations_object(:), 1, num_object);
    distance_object_to_object = abs(temp_location_matrix - temp_location_matrix.');
    
    
    
    % plot analytical scenario
    figure('Name', 'Scenario: analytical model');
    axis([0, 60, 0, 24]);
    hold on, box on;
    for circle_iter = 1: num_object
        filledCircle(locations_vec_object(circle_iter, :), r_object, 1000, color_blockage);
    end
    h_reference_object = filledCircle([30, 12], r_object, 1000, color_sensor);
    set(gca, 'FontSize', 12.0);
    save_figure(gca, gcf, 12.7, 5.5, '');
    pause;
    
    sample_point_x = [region(1):sample_gap:region(2)];
    num_sample_point_x = length(sample_point_x(:));
    sample_point_y = [region(3):sample_gap:region(4)];
    num_sample_point_y = length(sample_point_y(:));
    location_sample_point = repmat(sample_point_x(:).', num_sample_point_y, 1) ...
                    + 1i*repmat(sample_point_y(:), 1, num_sample_point_x);
    location_sample_point = location_sample_point(:);
    num_sample_point = length(location_sample_point);
    display(['# sample points: ', num2str(num_sample_point)]);
    distance_object_to_point = abs(repmat(locations_object(:), 1, num_sample_point) - repmat(location_sample_point(:).', num_object, 1));
    
    
    I_point_in_sensor = find(min(distance_object_to_point(sensor_set(:), :), [], 1) <= r_object).';
    I_point_in_blockage = find(min(distance_object_to_point(blockage_set(:), :), [], 1) <= r_object).';
    I_void_space = find(min(distance_object_to_point, [], 1) > r_object).';
    num_sample_point_void = length(I_void_space);

    covered_record = zeros(num_sample_point_void , 1);
    % compute whether a point is sensed
    for point_iter = 1: num_sample_point_void
        if (mod(point_iter, 1000) == 0)
            display(['point iter: ', num2str(point_iter)]);
        end
        temp_point_id = I_void_space(point_iter);
        temp_simu_point_location = location_sample_point(temp_point_id);
        temp_simu_point_location_vec = complex2vector(temp_simu_point_location);
        temp_point_sensed = 0;
        I_in_range_sensor = sensor_set(distance_object_to_point(sensor_set ,temp_point_id) <= r_sensing);
        if isempty(I_in_range_sensor)
            continue;
        end
        I_in_range_blockage_total = find(distance_object_to_point(:, temp_point_id) <= r_sensing + r_object);
        temp_distance_to_point_vec = distance_object_to_point(I_in_range_blockage_total, temp_point_id);
        for object_id = I_in_range_sensor(:).'
            temp_can_sense = 1;
            temp_los_channel = Segment(temp_simu_point_location_vec, locations_vec_object(object_id, :));
            temp_dist_object_to_point = distance_object_to_point(object_id, temp_point_id);
            % only consider blockage such that the sum of dist_blockage_to_object
            % and dist_blockage_to_point is smaller than
            % dist_object_to_point + 2*R_OBJECT
            I_in_range_blockage_object = find(temp_distance_to_point_vec + distance_object_to_object(object_id, I_in_range_blockage_total(:)).'  <= temp_dist_object_to_point + 2*r_object);
            for blocking_object_id = I_in_range_blockage_total(I_in_range_blockage_object(:)).'
                if blocking_object_id ~= object_id
                    if temp_los_channel.getDistToPoint(locations_vec_object(blocking_object_id, :)) <= r_object
                        temp_can_sense = 0;
                        break;
                    end
                end
            end
            if temp_can_sense == 1
                covered_record(point_iter) = 1;
                break;
            end
        end
    end
    covered_record = logical(covered_record);
    figure('Name', 'Collaborative sensing: Analytical model');
    if plot_model
        axis(region);
    else
        axis([-10, 10, -10, 10]);
    end
    hold on, box on;
    h_sensor = plot(real(location_sample_point(I_point_in_sensor)), imag(location_sample_point(I_point_in_sensor)), '.');
    h_blockage = plot(real(location_sample_point(I_point_in_blockage)), imag(location_sample_point(I_point_in_blockage)), '.');
    h_sensed = plot(real(location_sample_point(I_void_space(covered_record(:)))), imag(location_sample_point(I_void_space(covered_record(:)))), '.');
    h_blocked = plot(real(location_sample_point(I_void_space(~covered_record(:)))), imag(location_sample_point(I_void_space(~covered_record(:)))), '.');
    set(h_sensor, 'Color', color_sensor);
    set(h_blockage, 'Color', color_blockage);
    set(h_sensed, 'Color', color_sensed);
    set(h_blocked, 'Color', color_blocked);
    set(gca, 'FontSize', 12.0);
end


%% 2. Highway scenario
if plot_simulation
    LANE_NUM = 6;
    LANE_WIDTH = 4;
    density_lane = 0.04;
    display(['density: ', num2str(density_lane)]);
    if plot_model
        penetration = 0;
    else
        penetration = 0.15;
    end
    display(['penetration: ', num2str(penetration)]);
    MAX_LENGTH = 100;
    AREA = [0, MAX_LENGTH, 0, 0];
    GAP = 10; % minimum gap between vehicles 
    RANGE = 100; % max range for sensing
    LANE_VEHICLE_NUM = round(MAX_LENGTH*density_lane);
    Y_LOCATION = - LANE_WIDTH/2 + [1: LANE_NUM].*LANE_WIDTH;
    Y_RANGE = [-0.5, 0.5]; % y location center of user falls in Y_RANGE referenced to the center of the lane
    
    VEHICLE_LENGTH = 4.8;
    VEHICLE_WIDTH = 1.8;
     
    % vehicle location
    xlocation = cell(LANE_NUM, 1);
    ylocation = cell(LANE_NUM, 1);
    locations_vehicle = [];
    for lane_iter = 1: LANE_NUM
        xlocation{lane_iter} = sort(real(Matern(AREA, LANE_VEHICLE_NUM, GAP)));
        ylocation{lane_iter} = rand(size(xlocation{lane_iter})).*(Y_RANGE(2) - Y_RANGE(1)) + Y_RANGE(1) + Y_LOCATION(lane_iter);
        temp_vehicle_location = xlocation{lane_iter} + 1i.*ylocation{lane_iter};
        locations_vehicle = [locations_vehicle;temp_vehicle_location(:)];
    end
    num_vehicle = length(locations_vehicle(:));
    
    v2v_distance_matrix = repmat(locations_vehicle(:), 1, num_vehicle);
    v2v_distance_matrix = abs(v2v_distance_matrix - v2v_distance_matrix.');

    % Associate each location with vehicle, polygon (rectangle) of size 1.8m by
    % 4.8m
    EDGE = [1,2;2,3;3,4;4,1];
    vehicle_corner_points = [0; VEHICLE_LENGTH; VEHICLE_LENGTH + 1i*VEHICLE_WIDTH; 1i*VEHICLE_WIDTH] ...
             - (VEHICLE_LENGTH + 1i*VEHICLE_WIDTH)/2;
    display(['Vehicle number: ', num2str(num_vehicle)]);
    vehicle_polygons = Polygon.empty(length(locations_vehicle(:)), 0); % return an array of empty objects
    for vehicle_iter = 1: num_vehicle
        temp_points = locations_vehicle(vehicle_iter) + vehicle_corner_points;
        temp_points = complex2vector(temp_points(:));
        vehicle_polygons(vehicle_iter) = Polygon(temp_points);
    %     Vehicles(iter).plotPolygon(gca);
    end

    boundary_points = [0,0; MAX_LENGTH, 0; MAX_LENGTH, LANE_NUM*LANE_WIDTH; 0, LANE_NUM*LANE_WIDTH;];
    boundary_points(:,2) = boundary_points(:,2) - LANE_WIDTH.*LANE_NUM./2; % 04/25/2017 add line to correct boundary
    boundary = Polygon(boundary_points);

    locations_vec_vehicle = [real(locations_vehicle(:)), imag(locations_vehicle(:))];

    DIST_INTEREST = 2.6; % vehicle may block the segment if distance between center of a vehicle and the segment is smaller than threshold
    
    % plot scenario
    figure('Name', 'Scenario: highway simulation');
    temp_scenario_max_range = 60;
    axis_region = [0, temp_scenario_max_range, 0, LANE_WIDTH*LANE_NUM];
    axis(axis_region);
    centor_location = [axis_region(2) - axis_region(1), axis_region(4) - axis_region(3)]./2;
    I_central_lane_vehicles = find(inArea(locations_vehicle, [temp_scenario_max_range/2, inf, LANE_WIDTH*(LANE_NUM/2 - 1), LANE_WIDTH*(LANE_NUM/2 + 1)]));
    [~, I_center] = min(abs(locations_vehicle(I_central_lane_vehicles) - (temp_scenario_max_range/2+1i*LANE_WIDTH*LANE_NUM/2)),[],1);
    I_center_vehicle = I_central_lane_vehicles(I_center);
    new_central_vehicle_location = locations_vec_vehicle(I_center_vehicle,:);
    new_scenario_xlim = [[new_central_vehicle_location(1) - temp_scenario_max_range/2, new_central_vehicle_location(1) + temp_scenario_max_range/2]];
    xlim(new_scenario_xlim);
    hold on, box on;
    vehicle_corner_points_vector = [real(vehicle_corner_points(:)), imag(vehicle_corner_points(:))];
    for vehicle_iter = 1: num_vehicle
        temp_vehicle_location = locations_vec_vehicle(vehicle_iter, :);
        fill(temp_vehicle_location(1) + vehicle_corner_points_vector(:, 1), ...
             temp_vehicle_location(2) + vehicle_corner_points_vector(:, 2), ...
             color_blockage);
    end
    fill(new_central_vehicle_location(1) + vehicle_corner_points_vector(:, 1), ...
         new_central_vehicle_location(2) + vehicle_corner_points_vector(:, 2), ...
             color_sensor);
    for lane_iter = 1: LANE_NUM - 1 % plot lane separation line
        plot([0, MAX_LENGTH], lane_iter.*LANE_WIDTH.*[1,1], 'k--');
    end
    plot([0, MAX_LENGTH], LANE_NUM/2*LANE_WIDTH*[1,1], 'k'); % plot central line
    set(gca, 'FontSize', 12.0);
    save_figure(gca, gcf, 12.7, 5.5, '');
    pause;
    
    % find sensing vehicles
    vehicle_rand_value = rand(num_vehicle, 1);
    if plot_model
        vehicle_rand_value(I_center_vehicle) = 0;
    end
    vehicle_is_sensing = vehicle_rand_value <= penetration;
    sensor_set = find(vehicle_is_sensing == 1);
    blockage_set = find(vehicle_is_sensing == 0);
    
    % sample points location
    sample_point_x = [0:sample_gap:MAX_LENGTH];
    num_sample_point_x = length(sample_point_x(:));
    sample_point_y = [0:sample_gap:LANE_NUM*LANE_WIDTH];
    num_sample_point_y = length(sample_point_y(:));
    location_sample_point = repmat(sample_point_x(:).', num_sample_point_y, 1) ...
                    + 1i*repmat(sample_point_y(:), 1, num_sample_point_x);
    location_sample_point = location_sample_point(:);
    num_sample_point = length(location_sample_point);
    location_sample_point_vector = complex2vector(location_sample_point);
    display(['# sample points: ', num2str(num_sample_point)]);
    distance_vehicle_to_point = abs(repmat(locations_vehicle(:), 1, num_sample_point) - repmat(location_sample_point(:).', num_vehicle, 1));
    
    is_covered_sensor = zeros(num_sample_point, 1);
    is_covered_blockage = zeros(num_sample_point, 1);
    
    for vehicle_id = 1:num_vehicle
        temp_vehicle_location = locations_vec_vehicle(vehicle_id, :);
        temp_coverd = (abs(location_sample_point_vector(:, 1) - temp_vehicle_location(1)) <= VEHICLE_LENGTH/2) & ...
                           (abs(location_sample_point_vector(:, 2) - temp_vehicle_location(2)) <= VEHICLE_WIDTH/2); % points covered by vehicle
        if vehicle_is_sensing(vehicle_id) == 1
            is_covered_sensor(temp_coverd) = 1;
        else
            is_covered_blockage(temp_coverd) = 1;
        end
    end
    I_point_in_sensor = find(is_covered_sensor(:)).';
    I_point_in_blockage = find(is_covered_blockage(:)).';
    I_void_space = find( ~is_covered_sensor(:) & ~is_covered_blockage(:)).';
    
    void_point_locations = location_sample_point(I_void_space(:), :);
    num_sample_point_void = length(void_point_locations(:));
    vehicle_covered_record = zeros(num_sample_point_void , 1);
    for point_iter = 1: num_sample_point_void
        if mod(point_iter,1000) == 1
            display(['point: ', num2str(point_iter)]);
        end
        temp_point_id = I_void_space(point_iter);
        temp_simu_point_location_vec = location_sample_point_vector(temp_point_id, :);
        temp_point_sensed = 0;
        I_in_range_sensing_vehicle = sensor_set(distance_vehicle_to_point(sensor_set ,temp_point_id) <= RANGE);
        if isempty(I_in_range_sensing_vehicle)
            continue;
        end
        I_in_range_blockage_total = find(distance_vehicle_to_point(:, temp_point_id) <= RANGE + 2*DIST_INTEREST);
        if isempty(I_in_range_sensing_vehicle)
            break;
        end
        for sensing_vehicle_id = I_in_range_sensing_vehicle(:).'
            ray = Segment(temp_simu_point_location_vec, locations_vec_vehicle(sensing_vehicle_id, :));
            cansense = 1;
            if ~isempty(I_in_range_blockage_total)
                temp_vehicle_to_point_dist = distance_vehicle_to_point(sensing_vehicle_id, temp_point_id);
                temp_blocking_vehicle_set = find(distance_vehicle_to_point(I_in_range_blockage_total(:), temp_point_id) ...
                                                 + v2v_distance_matrix(sensing_vehicle_id, I_in_range_blockage_total(:)).' ...
                                                 <= temp_vehicle_to_point_dist + 2*DIST_INTEREST);
                for blocking_vehicle = I_in_range_blockage_total(temp_blocking_vehicle_set(:))'
                    if blocking_vehicle ~= sensing_vehicle_id && ...
                       isSegmentIntersectPolygon(ray, vehicle_polygons(blocking_vehicle)) % end of if condition
                        cansense = 0;
                        break;
                    end
                end
            end
            if (cansense == 1)
                vehicle_covered_record(point_iter) = 1;
                break;
            end
        end
    end
    vehicle_covered_record = logical(vehicle_covered_record);
    figure('Name', 'Collaborative sensing: Simulation model');
    axis([0, MAX_LENGTH, 0, LANE_WIDTH*LANE_NUM]);
    hold on, box on;
    if plot_model
        xlim(new_scenario_xlim);
    end
    h_sensor = plot(real(location_sample_point(I_point_in_sensor)), imag(location_sample_point(I_point_in_sensor)), '.');
    h_blockage = plot(real(location_sample_point(I_point_in_blockage)), imag(location_sample_point(I_point_in_blockage)), '.');
    h_sensed = plot(real(location_sample_point(I_void_space(vehicle_covered_record(:)))), imag(location_sample_point(I_void_space(vehicle_covered_record(:)))), '.');
    h_blocked = plot(real(location_sample_point(I_void_space(~vehicle_covered_record(:)))), imag(location_sample_point(I_void_space(~vehicle_covered_record(:)))), '.');
    for lane_iter = 1: LANE_NUM - 1 % plot lane separation line
        plot([0, MAX_LENGTH], lane_iter.*LANE_WIDTH.*[1,1], 'k--');
    end
    plot([0, MAX_LENGTH], LANE_NUM/2*LANE_WIDTH*[1,1], 'k'); % plot central line
    set(h_sensor, 'Color', color_sensor);
    set(h_blockage, 'Color', color_blockage);
    set(h_sensed, 'Color', color_sensed);
    set(h_blocked, 'Color', color_blocked);
    set(gca, 'FontSize', 12.0);
end
