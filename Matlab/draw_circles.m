clear all; clc
% scale by a factor of 5

sensorsRadii = [100*ones(1,7),50*ones(1,7)]';
sensorsCoord = 50*[8.73644	0.0203496;
0.945702	0.0954615;
7.37637	0.0560039;
8.42441	0.0585362;
9.29849	0.144649;
0.215903	0.0678067;
3.15625	0.193529;
2.48066	0.0406667;
8.45596	0.0825648;
8.08148	0.108054;
8.2672	0.155429;
0.666787	0.19286;
7.27055	0.185332;
2.8861	0.125174;
];
                                    
                                    

a = figure(1); hold on; 
for ii =1:length(sensorsCoord)
    h = circle(sensorsCoord(ii,1),sensorsCoord(ii,2),sensorsRadii(ii));
end
axis([0 500 0 10])
axis image
% rectangle('Position',[0 100 0 100])
% viscircles(sensorsCoord,sensorsRadii)
% axis([0 120 -40 20])

%%%%%%%%%%%%% lambda = 30 %%%%%%%%%%%%%
selectedSensorsCoord = 50*[0.945702	0.0954615;
7.37637	0.0560039;
8.42441	0.0585362;
3.15625	0.193529];

selectedsensorsRadii = [100*ones(1,4)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% lambda = 0.01 %%%%%%%%%%%%%
selectedSensorsCoord = 50*[0.945702	0.0954615;
7.37637	0.0560039;
8.42441	0.0585362;
3.15625	0.193529];

selectedsensorsRadii = [100*ones(1,4)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%% lambda = 15 %%%%%%%%%%%%%
selectedSensorsCoord = 50*[0.945702	0.0954615;
7.37637	0.0560039;
8.42441	0.0585362;
3.15625	0.193529];

selectedsensorsRadii = [100*ones(1,4)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


b=figure(2); hold on; 
for ii =1:length(selectedSensorsCoord)
    h = circle(selectedSensorsCoord(ii,1),selectedSensorsCoord(ii,2),selectedsensorsRadii(ii));
end
axis([0 500 0 10])
axis image