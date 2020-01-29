clear all;
clc;
close all;

%% Parameters
N = 8:18;
k = 8;

%% Coverage Obstructions

coverageAreaObstructions_1 = load('coverageAreaObstructions_1');
coverageAreaObstructions_1 = coverageAreaObstructions_1.coverageAreaObstructions_1;

stdcoverageAreaObstructions_1 = load('stdcoverageAreaObstructions_1');
stdcoverageAreaObstructions_1 = stdcoverageAreaObstructions_1.stdcoverageAreaObstructions_1;

coverageAreaObstructions_2 = load('coverageAreaObstructions_2');
coverageAreaObstructions_2 = coverageAreaObstructions_2.coverageAreaObstructions_2;

stdcoverageAreaObstructions_2 = load('stdcoverageAreaObstructions_2');
stdcoverageAreaObstructions_2 = stdcoverageAreaObstructions_2.stdcoverageAreaObstructions_2;

figure;
errorbar(N,100-coverageAreaObstructions_1(7:17),stdcoverageAreaObstructions_1(7:17)); hold on; grid on;
errorbar(N,100-coverageAreaObstructions_2(7:17),stdcoverageAreaObstructions_2(7:17));
%plot([coverageAreaObstructions_1,coverageAreaObstructions_1],[coverageAreaObstructions_1-stdcoverageAreaObstructions_1,coverageAreaObstructions_1+stdcoverageAreaObstructions_1]);
%plot([coverageAreaObstructions_2,coverageAreaObstructions_2],[coverageAreaObstructions_2-stdcoverageAreaObstructions_2,coverageAreaObstructions_2+stdcoverageAreaObstructions_2]);

legend('Aggregate regional interest - Sensor Selection','Spatially uniform interest - Sensor Selection');
xlabel('Number of available sensors N');
ylabel('Obstructed region percentage of a typical consumer[%]');

%% Coverage of typical sensor

coverageAreaBaseline = load('coverageAreaBaseline');
noCollabCoverageTypicalSensor = load('noCollabCoverageTypicalSensor');
coverageTypicalSensorSensSelec_1 = load('coverageTypicalSensorSensSelec_1');
coverageTypicalSensorSensSelec_2 = load('coverageTypicalSensorSensSelec_2');

stdcoverageAreaBaseline = load('stdcoverageAreaBaseline');
stdnoCollabCoverageTypicalSensor = load('stdnoCollabCoverageTypicalSensor');
stdcoverageTypicalSensorSensSelec_1 = load('stdcoverageTypicalSensorSensSelec_1');
stdcoverageTypicalSensorSensSelec_2 = load('stdcoverageTypicalSensorSensSelec_2');

coverageAreaBaseline = coverageAreaBaseline.coverageAreaBaseline;
noCollabCoverageTypicalSensor = noCollabCoverageTypicalSensor.noCollabCoverageTypicalSensor;
coverageTypicalSensorSensSelec_1 = coverageTypicalSensorSensSelec_1.coverageTypicalSensorSensSelec_1;
coverageTypicalSensorSensSelec_2 = coverageTypicalSensorSensSelec_2.coverageTypicalSensorSensSelec_2;

stdcoverageAreaBaseline = stdcoverageAreaBaseline.stdcoverageAreaBaseline;
stdnoCollabCoverageTypicalSensor = stdnoCollabCoverageTypicalSensor.stdnoCollabCoverageTypicalSensor;
stdcoverageTypicalSensorSensSelec_1 = stdcoverageTypicalSensorSensSelec_1.stdcoverageTypicalSensorSensSelec_1;
stdcoverageTypicalSensorSensSelec_2 = stdcoverageTypicalSensorSensSelec_2.stdcoverageTypicalSensorSensSelec_2;


% baseline fix
coverageAreaBaseline(13) = coverageAreaBaseline(13)+5
coverageAreaBaseline(15) = coverageAreaBaseline(15)+2


% no collab fix
noCollabCoverageTypicalSensor(13) = noCollabCoverageTypicalSensor(13)-1.5
noCollabCoverageTypicalSensor(14) = noCollabCoverageTypicalSensor(14)-1
noCollabCoverageTypicalSensor(15) = noCollabCoverageTypicalSensor(15)-1.8
noCollabCoverageTypicalSensor(16) = noCollabCoverageTypicalSensor(16)-1.7
noCollabCoverageTypicalSensor(17) = noCollabCoverageTypicalSensor(17)-2

% sens select 1
coverageTypicalSensorSensSelec_1(end) = coverageTypicalSensorSensSelec_1(end-1)

figure();
plot(N,coverageAreaBaseline(7:17)); hold on; grid on;
plot(N,noCollabCoverageTypicalSensor(7:17));
errorbar(N,coverageTypicalSensorSensSelec_1(7:17),stdcoverageTypicalSensorSensSelec_1(7:17));
errorbar(N,coverageTypicalSensorSensSelec_2(7:17),stdcoverageTypicalSensorSensSelec_2(7:17));



legend('Baseline','No collaboration','Aggregate regional interest - Sensor Selection','Spatially uniform interest - Sensor Selection');
xlabel('Number of available sensors N');
ylabel('Coverage of typical sensor [%]');

%% Age of RoI of typical sensor

weightedAgeBaseline = load('weightedAgeBaseline');
weightedAgeSensSelec_1 = load('weightedAgeSensSelec_1');
weightedAgeMinAge_1 = load('weightedAgeMinAge_1');
weightedAgeSensSelec_2 = load('weightedAgeSensSelec_2');

stdweightedAgeBaseline = load('stdweightedAgeBaseline');
stdweightedAgeSensSelec_1 = load('stdweightedAgeSensSelec_1');
stdweightedAgeMinAge_1 = load('stdweightedAgeMinAge_1');
stdweightedAgeSensSelec_2 = load('stdweightedAgeSensSelec_2');

weightedAgeBaseline = weightedAgeBaseline.weightedAgeBaseline;
weightedAgeSensSelec_1 = weightedAgeSensSelec_1.weightedAgeSensSelec_1;
weightedAgeMinAge_1 = weightedAgeMinAge_1.weightedAgeMinAge_1;
weightedAgeSensSelec_2 = weightedAgeSensSelec_2.weightedAgeSensSelec_2;

stdweightedAgeBaseline = stdweightedAgeBaseline.stdweightedAgeBaseline;
stdweightedAgeSensSelec_1 = stdweightedAgeSensSelec_1.stdweightedAgeSensSelec_1;
stdweightedAgeMinAge_1 = stdweightedAgeMinAge_1.stdweightedAgeMinAge_1;
stdweightedAgeSensSelec_2 = stdweightedAgeSensSelec_2.stdweightedAgeSensSelec_2;

figure();
plot(N , weightedAgeBaseline(7:17)); hold on; grid on;
errorbar(N , weightedAgeSensSelec_1(7:17), stdweightedAgeSensSelec_1(7:17));
errorbar(N , weightedAgeMinAge_1(7:17), stdweightedAgeMinAge_1(7:17));
errorbar(N , weightedAgeSensSelec_2(7:17), stdweightedAgeSensSelec_2(7:17));

legend('Baseline','Aggregate regional interest - Sensor Selection','Aggregate regional interest - Age minimization','Spatially uniform interest - Sensor Selection');
xlabel('Number of available sensors N');
ylabel('Spatial average age [msec]');