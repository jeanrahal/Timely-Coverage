clear all;
clc;
close all;

%% Parameters
N = 2:18;
k = 8;

%% Coverage 

coverageAreaBaseline = load('coverageAreaBaseline');
coverageAreaBaseline = coverageAreaBaseline.coverageAreaBaseline;

coverageAreaSensSelec = load('coverageAreaSensSelec');
coverageAreaSensSelec = coverageAreaSensSelec.coverageAreaSensSelec;

figure;
plot(N,coverageAreaBaseline); hold on; grid on;
plot(N,coverageAreaSensSelec);

legend('Baseline','Sensor Selection');
xlabel('Number of available sensors N');
ylabel('Coverage [%]');

%% Age of coverage

areaWeightedAgeBaseline = load('areaWeightedAgeBaseline');
areaWeightedAgeSensSelec = load('areaWeightedAgeSensSelec');
areaWeightedAgeAgeMin = load('areaWeightedAgeAgeMin');

areaWeightedAgeBaseline = areaWeightedAgeBaseline.areaWeightedAgeBaseline;
areaWeightedAgeSensSelec = areaWeightedAgeSensSelec.areaWeightedAgeSensSelec;
areaWeightedAgeAgeMin = areaWeightedAgeAgeMin.areaWeightedAgeAgeMin;

figure();
plot(N,areaWeightedAgeBaseline); hold on; grid on;
plot(N, areaWeightedAgeSensSelec);
plot(N, areaWeightedAgeAgeMin)
legend('Baseline','Sensor Selection','Age minimization');
xlabel('Number of available sensors N');
ylabel('Normalized area weighted average age [msec]');