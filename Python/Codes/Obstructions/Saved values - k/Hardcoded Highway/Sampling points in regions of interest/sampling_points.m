clear all;
clc;
close all;

%% Parameters
N = 24;
k = 8;
num_sampled_points = [1,6,11,16,21,26,31,36,41,46,51];
%% Age of RoI of typical sensor

weightedAgeBaseline = load('weightedAgeBaseline-k');
weightedAgeSensSelec_1 = load('weightedAgeSensSelec_1-k');
weightedAgeMinAge_1 = load('weightedAgeMinAge_1-k');
weightedAgeSensSelec_2 = load('weightedAgeSensSelec_2-k');


stdweightedAgeBaseline = load('stdweightedAgeBaseline-k');
stdweightedAgeSensSelec_1 = load('stdweightedAgeSensSelec_1-k');
stdweightedAgeMinAge_1 = load('stdweightedAgeMinAge_1-k');
stdweightedAgeSensSelec_2 = load('stdweightedAgeSensSelec_2-k');

ss = length(num_sampled_points);
weightedAgeBaseline = reshape(weightedAgeBaseline.weightedAgeBaseline,[1,ss]);
weightedAgeSensSelec_1 = reshape(weightedAgeSensSelec_1.weightedAgeSensSelec_1,[1,ss]);
weightedAgeMinAge_1 = reshape(weightedAgeMinAge_1.weightedAgeMinAge_1,[1,ss]);
weightedAgeSensSelec_2 = reshape(weightedAgeSensSelec_2.weightedAgeSensSelec_2,[1,ss]);

stdweightedAgeBaseline = reshape(stdweightedAgeBaseline.stdweightedAgeBaseline,[1,ss]);
stdweightedAgeSensSelec_1 = reshape(stdweightedAgeSensSelec_1.stdweightedAgeSensSelec_1,[1,ss]);
stdweightedAgeMinAge_1 = reshape(stdweightedAgeMinAge_1.stdweightedAgeMinAge_1,[1,ss]);
stdweightedAgeSensSelec_2 = reshape(stdweightedAgeSensSelec_2.stdweightedAgeSensSelec_2,[1,ss]);


figure();
p1 = errorbar(num_sampled_points , weightedAgeBaseline,stdweightedAgeBaseline); hold on; grid on;
%plot(k , weightedAgeSensSelec_1);
%plot(k , weightedAgeMinAge_1);
%plot(k , weightedAgeSensSelec_2);

p2 = errorbar(num_sampled_points , weightedAgeSensSelec_1, stdweightedAgeSensSelec_1);
%p3 = errorbar(num_sampled_points , weightedAgeMinAge_1 , stdweightedAgeMinAge_1);
p4 = errorbar(num_sampled_points , weightedAgeSensSelec_2, stdweightedAgeSensSelec_2);

legend('Baseline','Aggregated spatial interest - Sensor Selection','Uniform aggregated spatial interest - Sensor Selection');
xlabel('Number of randomly sampled anchor points per sensor');
ylabel('Spatial average age [msec]');