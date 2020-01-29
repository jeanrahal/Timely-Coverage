clear all;
clc;
close all;

%% Parameters
N = 15;
k = 8:15;

%% Coverage of typical sensor

coverageAreaBaseline = load('coverageAreaBaseline-k');
noCollabCoverageTypicalSensor = load('noCollabCoverageTypicalSensor-k');
coverageTypicalSensorSensSelec_1 = load('coverageTypicalSensorSensSelec_1-k');
coverageTypicalSensorSensSelec_2 = load('coverageTypicalSensorSensSelec_2-k');

coverageAreaBaseline = [coverageAreaBaseline.coverageAreaBaseline(1),coverageAreaBaseline.coverageAreaBaseline(2),coverageAreaBaseline.coverageAreaBaseline(3),coverageAreaBaseline.coverageAreaBaseline(4),coverageAreaBaseline.coverageAreaBaseline(5),coverageAreaBaseline.coverageAreaBaseline(6),coverageAreaBaseline.coverageAreaBaseline(7),coverageAreaBaseline.coverageAreaBaseline(8)];
noCollabCoverageTypicalSensor = [noCollabCoverageTypicalSensor.noCollabCoverageTypicalSensor(1),noCollabCoverageTypicalSensor.noCollabCoverageTypicalSensor(2),noCollabCoverageTypicalSensor.noCollabCoverageTypicalSensor(3),noCollabCoverageTypicalSensor.noCollabCoverageTypicalSensor(4),noCollabCoverageTypicalSensor.noCollabCoverageTypicalSensor(5),noCollabCoverageTypicalSensor.noCollabCoverageTypicalSensor(6),noCollabCoverageTypicalSensor.noCollabCoverageTypicalSensor(7),noCollabCoverageTypicalSensor.noCollabCoverageTypicalSensor(8)];
coverageTypicalSensorSensSelec_1 = [coverageTypicalSensorSensSelec_1.coverageTypicalSensorSensSelec_1(1),coverageTypicalSensorSensSelec_1.coverageTypicalSensorSensSelec_1(2),coverageTypicalSensorSensSelec_1.coverageTypicalSensorSensSelec_1(3),coverageTypicalSensorSensSelec_1.coverageTypicalSensorSensSelec_1(4),coverageTypicalSensorSensSelec_1.coverageTypicalSensorSensSelec_1(5),coverageTypicalSensorSensSelec_1.coverageTypicalSensorSensSelec_1(6),coverageTypicalSensorSensSelec_1.coverageTypicalSensorSensSelec_1(7),coverageTypicalSensorSensSelec_1.coverageTypicalSensorSensSelec_1(8)];
coverageTypicalSensorSensSelec_2 = [coverageTypicalSensorSensSelec_2.coverageTypicalSensorSensSelec_2(1),coverageTypicalSensorSensSelec_2.coverageTypicalSensorSensSelec_2(2),coverageTypicalSensorSensSelec_2.coverageTypicalSensorSensSelec_2(3),coverageTypicalSensorSensSelec_2.coverageTypicalSensorSensSelec_2(4),coverageTypicalSensorSensSelec_2.coverageTypicalSensorSensSelec_2(5),coverageTypicalSensorSensSelec_2.coverageTypicalSensorSensSelec_2(6),coverageTypicalSensorSensSelec_2.coverageTypicalSensorSensSelec_2(7),coverageTypicalSensorSensSelec_2.coverageTypicalSensorSensSelec_2(8)];

figure();
plot(k,coverageAreaBaseline); hold on; grid on;
plot(k,noCollabCoverageTypicalSensor);
plot(k,coverageTypicalSensorSensSelec_1);
plot(k,coverageTypicalSensorSensSelec_2);
legend('Baseline','No collaboration','Aggregate regional interest - Sensor Selection','Spatially uniform interest - Sensor Selection');
xlabel('Number of selected sensors k');
ylabel('Coverage of typical sensor [%]');

%% Age of RoI of typical sensor

weightedAgeBaseline = load('weightedAgeBaseline-k');
weightedAgeSensSelec_1 = load('weightedAgeSensSelec_1-k');
weightedAgeMinAge_1 = load('weightedAgeMinAge_1-k');
weightedAgeSensSelec_2 = load('weightedAgeSensSelec_2-k');

weightedAgeBaseline = [weightedAgeBaseline.weightedAgeBaseline(1),weightedAgeBaseline.weightedAgeBaseline(2),weightedAgeBaseline.weightedAgeBaseline(3),weightedAgeBaseline.weightedAgeBaseline(4),weightedAgeBaseline.weightedAgeBaseline(5),weightedAgeBaseline.weightedAgeBaseline(6),weightedAgeBaseline.weightedAgeBaseline(7),weightedAgeBaseline.weightedAgeBaseline(8)];
weightedAgeSensSelec_1 = [weightedAgeSensSelec_1.weightedAgeSensSelec_1(1),weightedAgeSensSelec_1.weightedAgeSensSelec_1(2),weightedAgeSensSelec_1.weightedAgeSensSelec_1(3),weightedAgeSensSelec_1.weightedAgeSensSelec_1(4),weightedAgeSensSelec_1.weightedAgeSensSelec_1(5),weightedAgeSensSelec_1.weightedAgeSensSelec_1(6),weightedAgeSensSelec_1.weightedAgeSensSelec_1(7),weightedAgeSensSelec_1.weightedAgeSensSelec_1(8)];
weightedAgeMinAge_1 = [weightedAgeMinAge_1.weightedAgeMinAge_1(1),weightedAgeMinAge_1.weightedAgeMinAge_1(2),weightedAgeMinAge_1.weightedAgeMinAge_1(3),weightedAgeMinAge_1.weightedAgeMinAge_1(4),weightedAgeMinAge_1.weightedAgeMinAge_1(5),weightedAgeMinAge_1.weightedAgeMinAge_1(6),weightedAgeMinAge_1.weightedAgeMinAge_1(7),weightedAgeMinAge_1.weightedAgeMinAge_1(8)];
weightedAgeSensSelec_2 = [weightedAgeSensSelec_2.weightedAgeSensSelec_2(1),weightedAgeSensSelec_2.weightedAgeSensSelec_2(2),weightedAgeSensSelec_2.weightedAgeSensSelec_2(3),weightedAgeSensSelec_2.weightedAgeSensSelec_2(4),weightedAgeSensSelec_2.weightedAgeSensSelec_2(5),weightedAgeSensSelec_2.weightedAgeSensSelec_2(6),weightedAgeSensSelec_2.weightedAgeSensSelec_2(7),weightedAgeSensSelec_2.weightedAgeSensSelec_2(8)];

figure();
plot(k , weightedAgeBaseline); hold on; grid on;
plot(k , weightedAgeSensSelec_1);
plot(k , weightedAgeMinAge_1);
plot(k , weightedAgeSensSelec_2);
legend('Baseline','No collaboration','Aggregate regional interest - Sensor Selection','Spatially uniform interest - Sensor Selection');
xlabel('Number of selected sensors k');
ylabel('Normalized weighted average age [msec]');