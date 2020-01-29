clear all;
clc;
close all;

%% Parameters
k = 1:15;

%% Coverage 

coverageAreaBaseline = load('coverageBaseline-k');
coverageAreaBaseline = reshape(coverageAreaBaseline.coverageBaseline,[1 15]);

coverageAreaSensSelec = load('coverageSensSelec-k');
coverageAreaSensSelec = reshape(coverageAreaSensSelec.coverageSensSelec,[1 15]);

figure;
plot(k,[79.8916666666667]*ones(1,15)); hold on; grid on;
plot(k,coverageAreaSensSelec);

legend('Baseline','Sensor Selection');
xlabel('Number of selected sensors k');
ylabel('Coverage [%]');

%% Age of coverage

areaWeightedAgeBaseline = load('areaweightedAgeBaseline-k');
areaWeightedAgeSensSelec = load('areaweightedAgeSensSelec-k');
areaWeightedAgeAgeMin = load('areaweightedAgeAgeMin-k');

areaWeightedAgeBaseline = reshape(areaWeightedAgeBaseline.areaweightedAgeBaseline,[1 15]);
areaWeightedAgeSensSelec = reshape(areaWeightedAgeSensSelec.areaweightedAgeSensSelec,[1 15]);
areaWeightedAgeAgeMin = reshape(areaWeightedAgeAgeMin.areaweightedAgeAgeMin,[1 15]);

figure();
plot(k,areaWeightedAgeBaseline); hold on; grid on;
plot(k, areaWeightedAgeSensSelec);
plot(k, areaWeightedAgeAgeMin)
legend('Baseline','Sensor Selection','Age minimization');
xlabel('Number of selected sensors k');
ylabel('Normalized area weighted average age [msec]');