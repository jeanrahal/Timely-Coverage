clear all;clc;

% Number of total sensors: N = 10

%Generate sensors radii:
% Half of them have a radius of 50m, the other half have a radius of 100m

sensorsRadii = 50*[1,1,1,1,1,2,2,2,2,2]';

% Sensors' coordinates from the Python code: test_algos.py
sensorsCoord = 50*[0.346256	0.794008;
17.6222	1.67842;
1.60685	1.52488;
17.6952	0.376898;
14.8532	1.3532;
5.21618	1.56915;
17.8326	0.501913;
13.8915	0.141149;
0.0616458	0.807074;
12.3948	0.727091
];


% Plot the initial sensors in the square of dimensions: 1000mx100m
a = figure(1); hold on; title('Sensor placement');
color = 'black';
line = 'linewidth';
l = 1;
%thick = '--';
for ii =1:length(sensorsCoord)
    h = circle(sensorsCoord(ii,1),sensorsCoord(ii,2),sensorsRadii(ii),color,line,l);
end
axis([0 1000 0 100])
axis image


%%%%%%%%%%%%% lambda = 1 %%%%%%%%%%%%%
% Order of sensor selection: 10-7-6-9
selectedSensorsCoord = 50*[12.3948	0.727091;
    17.8326	0.501913;
    5.21618	1.56915;
    0.0616458	0.807074
];

b=figure(2); hold on; title('\lambda = 1');
for ii =1:length(sensorsCoord)
    if ii == 10 || ii == 7 || ii == 6 || ii == 9
        color = 'red';
        line = 'linewidth';
        l = 2;
        %thick = '--';
    else
        color = 'black';
        line = 'linewidth';
        l = 1;
        %thick = '-';
    end
    h = circle(sensorsCoord(ii,1),sensorsCoord(ii,2),sensorsRadii(ii),color,line,l);
end
axis([0 1000 0 100])
axis image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% lambda = lambda_min = 0.001833 %%%%%%%%%%%%%
% Order of sensor selection: 
selectedSensorsCoord = 50*[
];

b=figure(3); hold on; title('\lambda = d+\frac{2}{3r} = 0.001833');
for ii =1:length(sensorsCoord)
    if ii == 10 || ii == 7 || ii == 6 || ii == 8
        color = 'blue';
        line = 'linewidth';
        l = 2;
        %thick = '--';
    else
        color = 'black';
        line = 'linewidth';
        l = 1;
        %thick = '-';
    end
    h = circle(sensorsCoord(ii,1),sensorsCoord(ii,2),sensorsRadii(ii),color,line,l);
end
axis([0 1000 0 100])
axis image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%%%%%%%%%%%%% lambda = lambda_min = 0.0015 %%%%%%%%%%%%%
% Order of sensor selection: 
selectedSensorsCoord = 50*[
];

b=figure(4); hold on; title('\lambda = d+\frac{1}{2r} = 0.0015');
for ii =1:length(sensorsCoord)
    if ii == 10 || ii == 8 || ii == 5 || ii == 7
        color = 'green';
        line = 'linewidth';
        l = 2;
        %thick = '--';
    else
        color = 'black';
        line = 'linewidth';
        l = 1;
        %thick = '-';
    end
    h = circle(sensorsCoord(ii,1),sensorsCoord(ii,2),sensorsRadii(ii),color,line,l);
end
axis([0 1000 0 100])
axis image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

