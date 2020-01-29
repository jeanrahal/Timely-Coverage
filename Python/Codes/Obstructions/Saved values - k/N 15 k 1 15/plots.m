clear all;
clc;
close all;

%% Parameters
N = 15;
k = 1:15;

%% Coverage of typical sensor

coverageAreaBaseline = load('coverageAreaBaseline-k');
noCollabCoverageTypicalSensor = load('noCollabCoverageTypicalSensor-k');
coverageTypicalSensorSensSelec_1 = load('coverageTypicalSensorSensSelec_1-k');
coverageTypicalSensorSensSelec_2 = load('coverageTypicalSensorSensSelec_2-k');

stdcoverageAreaBaseline = load('stdcoverageAreaBaseline-k');
stdnoCollabCoverageTypicalSensor = load('stdnoCollabCoverageTypicalSensor-k');
stdcoverageTypicalSensorSensSelec_1 = load('stdcoverageTypicalSensorSensSelec_1-k');
stdcoverageTypicalSensorSensSelec_2 = load('stdcoverageTypicalSensorSensSelec_2-k');


coverageAreaBaseline = reshape(coverageAreaBaseline.coverageAreaBaseline,[1,15]);
noCollabCoverageTypicalSensor = reshape(noCollabCoverageTypicalSensor.noCollabCoverageTypicalSensor,[1,15]);
coverageTypicalSensorSensSelec_1 = reshape(coverageTypicalSensorSensSelec_1.coverageTypicalSensorSensSelec_1,[1,15]);
coverageTypicalSensorSensSelec_2 = reshape(coverageTypicalSensorSensSelec_2.coverageTypicalSensorSensSelec_2,[1,15]);

stdcoverageAreaBaseline = reshape(stdcoverageAreaBaseline.stdcoverageAreaBaseline,[1,15]);
stdnoCollabCoverageTypicalSensor = reshape(stdnoCollabCoverageTypicalSensor.stdnoCollabCoverageTypicalSensor,[1,15]);
stdcoverageTypicalSensorSensSelec_1 = reshape(stdcoverageTypicalSensorSensSelec_1.stdcoverageTypicalSensorSensSelec_1,[1,15]);
stdcoverageTypicalSensorSensSelec_2 = reshape(stdcoverageTypicalSensorSensSelec_2.stdcoverageTypicalSensorSensSelec_2,[1,15]);


figure();
p1 = plot(k,coverageAreaBaseline); hold on; grid on;
p2 = plot(k,noCollabCoverageTypicalSensor);
%p3 = plot(k,coverageTypicalSensorSensSelec_1);
%p4 = plot(k,coverageTypicalSensorSensSelec_2);

p5 = errorbar(k,coverageTypicalSensorSensSelec_1,stdcoverageTypicalSensorSensSelec_1);
p6 = errorbar(k,coverageTypicalSensorSensSelec_2,stdcoverageTypicalSensorSensSelec_2);

% for ii=min(k):1:max(k)
%     %plot([k(ii),k(ii)],[coverageAreaBaseline(ii)-stdcoverageAreaBaseline(ii),coverageAreaBaseline(ii)+stdcoverageAreaBaseline(ii)], 'black');
%     %plot([k(ii),k(ii)],[noCollabCoverageTypicalSensor(ii)-stdnoCollabCoverageTypicalSensor(ii),noCollabCoverageTypicalSensor(ii)+stdnoCollabCoverageTypicalSensor(ii)]);
% end

legend([p1,p2,p5,p6],'Baseline','No collaboration','Aggregate regional interest - Sensor Selection','Spatially uniform interest - Sensor Selection');
xlabel('Number of selected sensors k');
ylabel('Coverage of typical sensor [%]');



%% Age of RoI of typical sensor

weightedAgeBaseline = load('weightedAgeBaseline-k');
weightedAgeSensSelec_1 = load('weightedAgeSensSelec_1-k');
weightedAgeMinAge_1 = load('weightedAgeMinAge_1-k');
weightedAgeSensSelec_2 = load('weightedAgeSensSelec_2-k');


stdweightedAgeBaseline = load('stdweightedAgeBaseline-k');
stdweightedAgeSensSelec_1 = load('stdweightedAgeSensSelec_1-k');
stdweightedAgeMinAge_1 = load('stdweightedAgeMinAge_1-k');
stdweightedAgeSensSelec_2 = load('stdweightedAgeSensSelec_2-k');


weightedAgeBaseline = reshape(weightedAgeBaseline.weightedAgeBaseline,[1,15]);
weightedAgeSensSelec_1 = reshape(weightedAgeSensSelec_1.weightedAgeSensSelec_1,[1,15]);
weightedAgeMinAge_1 = reshape(weightedAgeMinAge_1.weightedAgeMinAge_1,[1,15]);
weightedAgeSensSelec_2 = reshape(weightedAgeSensSelec_2.weightedAgeSensSelec_2,[1,15]);

stdweightedAgeBaseline = reshape(stdweightedAgeBaseline.stdweightedAgeBaseline,[1,15]);
stdweightedAgeSensSelec_1 = reshape(stdweightedAgeSensSelec_1.stdweightedAgeSensSelec_1,[1,15]);
stdweightedAgeMinAge_1 = reshape(stdweightedAgeMinAge_1.stdweightedAgeMinAge_1,[1,15]);
stdweightedAgeSensSelec_2 = reshape(stdweightedAgeSensSelec_2.stdweightedAgeSensSelec_2,[1,15]);


figure();
p1 = plot(k , weightedAgeBaseline); hold on; grid on;
%plot(k , weightedAgeSensSelec_1);
%plot(k , weightedAgeMinAge_1);
%plot(k , weightedAgeSensSelec_2);

p2 = errorbar(k , weightedAgeSensSelec_1, stdweightedAgeSensSelec_1);
p3 = errorbar(k , weightedAgeMinAge_1 , stdweightedAgeMinAge_1);
p4 = errorbar(k , weightedAgeSensSelec_2, stdweightedAgeSensSelec_2);

legend('Baseline','Aggregate regional interest - Sensor Selection','Aggregate regional interest - Age minimization','Spatially uniform interest - Sensor Selection');
xlabel('Number of selected sensors k');
ylabel('Spatial average age [msec]');