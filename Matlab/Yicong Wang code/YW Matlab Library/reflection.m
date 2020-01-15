% 09/16/2015: class for reflection coefficient
% See 94 K. Sato, Measurements of Reflection Characteristics
% and Refractive Indices of Interior Construction Materials in
% Millimeter-Wave Bands for model.
% REMARK: Refractive Index = sqrt(Relative Permittivity * Relative Permeability)
% For simplicity, treat Relative Permeability as one
% Default refraction index for reflection over ceiling, Refracrive Index: 1.5 - 0.01i OR 1.74
% - 0.023i, thickness 9 mm, wavelength 0.05217
% Slide: Complex Permittivity = 2.48 - 0.03i, wave length = 4.839 mm,
% thickness unclear
% CMU Ceiling tile: 1.55 - 0.026i, thickness = 15.9, see 14 Propagation
% Characterization of an Office Building in he 60GHz Band, wave length =
% 5mm, thickness is 15.9 mm: LOW reflection
% getTE, getTM return the Fresnel;s reflection coefficients for TE and TM
classdef reflection < handle
    properties
%         RefractiveIndex = 1.55 - 0.026*1i;
%         ComplexPermittivity = (1.5 - 0.01i)^2;
        ComplexPermittivity = 2.48 - 0.03i;
%           ComplexPermittivity = 8.9 - 10.9i; % Body?
%         ComplexPermittivity = (1.55 - 0.026i);
%         ComplexPermittivity = 2.70 - 0.026i; % Plexiglass, in CMU paper
        WaveLength = 0.005;
        Thickness = 0.009; % Thickness of sample plate
        Title;
    end
    methods
        function obj = reflection()
%             display(['Default n: ', num2str(obj.ComplexPermittivity),' Wavelength: ', num2str(obj.WaveLength), ' Thickness: ', num2str(obj.Thickness)]);
        end
        function setComplexPermittivity(obj, ri)
            obj.RefractiveIndex = ri;
            display(['New Complex Permittivity: ', num2str(obj.ComplexPermittivity)]);
        end
        function setWaveLength(obj, wave)
            obj.WaveLength = wave;
            display(['New Wave Length: ', num2str(obj.WaveLength)]);
        end
        function setThickness(obj, th)
            obj.Thickness = th;
            display(['New Thickness: ', num2str(obj.Thickness)]);
        end
        function te = getTE(obj, theta)
            dividend =  cos(theta) - sqrt(obj.ComplexPermittivity - sin(theta).^2);
            divisor = cos(theta) + sqrt(obj.ComplexPermittivity - sin(theta).^2);
            te = dividend./divisor;
        end
        function tm = getTM(obj, theta)
            dividend = obj.ComplexPermittivity.*cos(theta) - sqrt(obj.ComplexPermittivity - sin(theta).^2);
            divisor = obj.ComplexPermittivity.*cos(theta) + sqrt(obj.ComplexPermittivity - sin(theta).^2);
            tm = dividend./divisor;
        end
        function [r_te, r_tm] = getReflectionCoefficient(obj, theta)
            delta = 2.*pi.*obj.Thickness./obj.WaveLength.*sqrt(obj.ComplexPermittivity - sin(theta).^2);
            te = obj.getTE(theta);
            tm = obj.getTM(theta);
            r_te = (1-exp(-1i.*2.*delta))./(1 - te.^2.*exp(-1i.*2.*delta)).*te;
            r_tm = (1-exp(-1i.*2.*delta))./(1 - tm.^2.*exp(-1i.*2.*delta)).*tm;
        end
        function H = plotReflectionCoefficient(obj)
            obj.Title = ['Reflection Coefficient \epsilon_r=', num2str(obj.ComplexPermittivity), ' d=', num2str(obj.Thickness), ' \lambda=', num2str(obj.WaveLength)];
            theta = [0:1:90];
            [rte, rtm] = obj.getReflectionCoefficient(theta./180*pi);
            H = figure('name', obj.Title);
            axis([0,90,0,1]), hold on, grid on,box on;
            h = zeros(1,3);
            h(1) = plot(theta, abs(rte),'bs-');
            h(2) = plot(theta, abs(rtm), 'r+-');
            h(3) = plot(theta, (abs(rte) + abs(rtm))./2, 'g-.');
            set(h, 'LineWidth', 2.0);
            legend('TE', 'TM', 'Avg');
            xlabel('Incident Degree', 'FontSize', 12.0);
            ylabel('Reflection Coefficient','FontSize',12.0);
            set(gca, 'FontSize', 12.0);
            title(obj.Title);
        end
    end
end