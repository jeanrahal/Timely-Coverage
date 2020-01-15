classdef environment < handle
    % Analyze # of Strong Interferers based on channel model and pdf of
    % interferers following NumOfNonBlockedStrongInterferer.m
    % Won't change once initialized 
    % Import Research\15 Spring\MAC Design in Wearable Network    
    % 09/19/2015: revision
    % 1. add correction_factor for distance between users
    % 10/07/2015:
    % Add getMarkovPara(obj, distance) to return the parameters of Markov
    % channel model
    % 12/08/2015:
    % Return prob function of having a clear channel
    % 01/08/2016: add ProbClear, return probability of having a clear
    % channel as a function of distance
    % edit correction_factor: from factor*d_body to factor*R_min
    % 04/05/2016: add getSumInterference(), return the sum interference
    % from unblocked interferers
    properties
        Lambda;
        Para;
        Perturbation;
        AuxiliaryPara;
        PDF_function;
        PDF_blockage;
        PDF_noSelfBlockage;
        ProbBlocked; % Probability that channel is blocked as a function of r
        ProbClear; % Prob not blocked as a function of r
        PathLoss; % function of path loss
        InterferenceDensity; % Interference density function 
    end
    methods
        function obj = environment(lambda, para, perturbation)
            obj.Lambda = lambda;
            R = 10;
            R_min = 0.6;
            R_max = R;
            d = sqrt(0.25*0.46); % Model human body by ellipse with a = 0.33, b = 0.46, equivalent d is sqrt(a*b)
            omega = 2*pi*2/3;
            H = 2.8; % Height of ceiling
            h_body = 1.524; % height of body
            h = 1.754; % total heigh of person
            h_device = 1.0; % height of device
            reflection_coeff = 0.2166; % reflection coefficient
            d_body = d;
            d_head = sqrt(0.25*0.15);
            correction_factor = 0;
            pathloss_exponent = 2;
            if(isstruct(para))
                if isfield(para,'R')
                    R = para.R;
                end
                if isfield(para, 'R_min')
                    R_min = para.R_min;
                end
                if isfield(para, 'd')
                    d_body = para.d(1);
                    d_head = para.d(2);
                end
                if isfield(para, 'omega')
                    omega = para.omega;
                end
                if isfield(para, 'H')
                    H = para.H;
                end
                if isfield(para, 'h')
                    h = para.h(1);
                    h_body = para.h(2);
                    h_device = para.h(3);
                end
                if isfield(para, 'reflection_coeff')
                    reflection_coeff = para.reflection_coeff;
                end
                if isfield(para, 'R_max')
                    R_max = para.R_max;
                end
                if isfield(para, 'correction_factor')
                    correction_factor = para.correction_factor;
                end
                if isfield(para, 'pathloss_exponent')
                    pathloss_exponent = para.pathloss_exponent;
                end
            end
            TRANSLATION_RANGE = d/2;
            ROTATION_RANGE = 0.1*omega;
            if ~isempty(perturbation)
                if isfield(perturbation,'translation')
                    TRANSLATION_RANGE = perturbation.translation;
                end
                if isfield(perturbation, 'rotation')
                    ROTATION_RANGE = perturbation.rotation;
                end
            end
            obj.Perturbation = struct('translation', TRANSLATION_RANGE, 'rotation', ROTATION_RANGE);
            R_reflected = R*10^(10*log10(reflection_coeff)./20); % PL ~ L^-2 thus use 20 in this equation
            R_reflected = sqrt(R_reflected^2 - (2*(H-h_device))^2); % The maximum radius in 2-D plane for a strong interferer which may have a strong reflection over ceiling
            obj.Para = struct('R', R, 'R_min', R_min, 'R_max', R_max, 'd_body', d_body, 'd_head', d_head ,'d', ...
                d, 'omega', omega, 'H', H, 'h', h,'h_body', h_body, 'h_device', h_device, 'reflection_coeff', ...
                reflection_coeff,'correction_factor', correction_factor, 'pathloss_exponent', pathloss_exponent);
            syms r theta;
            d_body_perturb = DBlockedAfterPerturbation(TRANSLATION_RANGE, d_body);
            d_head_perturb = DBlockedAfterPerturbation(TRANSLATION_RANGE, d_head);
            pdf_perturb = @(r, theta) 1./ pi./TRANSLATION_RANGE.^2;
            d_body_txrx_perturb = WidthTXRXPerturbation(d_body, d_body + d_body_perturb, pdf_perturb, TRANSLATION_RANGE);
            d_head_txrx_perturb = WidthTXRXPerturbation(d_head, d_head + d_head_perturb, pdf_perturb, TRANSLATION_RANGE);
            rotation_dist = @(theta) 1./(2*ROTATION_RANGE).*(abs(theta)<ROTATION_RANGE); % Valid for range < pi
            Prob_rotation_perturb = ProbFacingAfterRotation(omega, rotation_dist); % Probability that RX TX are still facing each other after random rotation
            obj.AuxiliaryPara = struct('R_reflected', R_reflected, 'd_body_perturb', d_body_perturb, ...
                'd_head_perturb', d_head_perturb, 'd_body_txrx_perturb',d_body_txrx_perturb, 'd_head_txrx_perturb', d_head_txrx_perturb,'Prob_rotation_perturb', Prob_rotation_perturb);
            obj.helperGetFunctions();
        end
        function helperGetFunctions(obj)
            syms r theta;
            para = obj.Para;
            R = para.R;
            R_min = para.R_min;
            omega = para.omega;
            h_body = para.h_body;
            h = para.h;
            H = para.H;
            h_device = para.h_device;
            d_body = para.d_body;
            d_head = para.d_head;
            correction_factor = para.correction_factor;
            d_body_txrx_perturb = obj.AuxiliaryPara.d_body_txrx_perturb;
            d_body_perturb = obj.AuxiliaryPara.d_body_perturb;
            R_reflected = obj.AuxiliaryPara.R_reflected;
            d_head_txrx_perturb = obj.AuxiliaryPara.d_head_txrx_perturb;
            d_head_perturb = obj.AuxiliaryPara.d_head_perturb;
            Prob_rotation_perturb = obj.AuxiliaryPara.Prob_rotation_perturb;
            fun_reflected = @(r) omega ./ 2./ pi .*exp(-( (h_body - h_device).*d_body + (h - h_body).*d_head)./(H - h_device).*obj.Lambda.*(r + correction_factor*d_body)).*r;
            fun_LOS_clear  = @(r) exp(-d_body.*obj.Lambda.*(r - correction_factor*d_body));
            fun_reflected_clear = @(r) exp(-( (h_body - h_device).*d_body + (h - h_body).*d_head)./(H - h_device).*obj.Lambda.*(r + correction_factor*d_body));
            fun_LOS = @(r) omega./2./pi.*exp(-d_body.*obj.Lambda.*(r - correction_factor*d_body)).*r;
            fun_LOS_perturb = @(r) omega./2./pi.*exp(-(d_body + d_body_perturb).*obj.Lambda.*(r - correction_factor*d_body)).*r;
            fun_reflected_perturb = @(r) omega ./ 2./ pi .*exp(-( (h_body - h_device).*(d_body + d_body_perturb) + (h - h_body).*(d_head + d_head_perturb))./(H - h_device).*obj.Lambda.*(r + correction_factor*d_body)).*r;
            fun_LOS_txrx_perturb = @(r)omega./2./pi.*exp(-(d_body + d_body_txrx_perturb).*obj.Lambda.*(r - correction_factor*d_body)).*r;
            fun_reflected_txrx_perturb = @(r) omega ./ 2./ pi .*exp(-( (h_body - h_device).*(d_body + d_body_txrx_perturb) + (h - h_body).*(d_head + d_head_txrx_perturb))./(H - h_device).*obj.Lambda.*(r + correction_factor*d_body)).*r;
            field1 = 'df_LOS';
            field2 = 'df_LOS_perturb';
            field3 = 'df_LOS_txrx_perturb';
            field4 = 'df_ceiling';
            field5 = 'df_ceiling_perturb';
            field6 = 'df_ceiling_txrx_perturb';
            value1 = @(r) (fun_LOS(r) .*omega .* obj.Lambda.* (r < R & r>R_min));
            value2 = @(r) Prob_rotation_perturb*fun_LOS_perturb(r).*omega.*obj.Lambda .* (r < R & r>R_min);
            value3 = @(r) Prob_rotation_perturb*fun_LOS_txrx_perturb(r).*omega.*obj.Lambda .* (r < R & r>R_min);
            value4 = @(r) (fun_reflected(r).*(r<R_reflected & r>R_min) + fun_LOS(r).* (r>= R_reflected & r < R)).* omega .*obj.Lambda;
            value5 = @(r) Prob_rotation_perturb*(fun_reflected_perturb(r).*(r<R_reflected & r>R_min) + fun_LOS_perturb(r).* (r>= R_reflected & r < R)).* omega.* obj.Lambda;
            value6 = @(r) Prob_rotation_perturb*(fun_reflected_txrx_perturb(r).*(r<R_reflected & r>R_min) + fun_LOS_txrx_perturb(r).* (r>= R_reflected & r < R)).* omega.* obj.Lambda;
            obj.PDF_function = struct(field1, value1, field2, value2, field3, value3, field4, value4, field5, value5, field6, value6);
            field1 = 'LOS';
            field2 = 'Reflected';
            field3 = 'Total';
            value1 = @(r) (fun_LOS(r)*2*pi./omega).*(r>R_min).* obj.Lambda;
            value2 = @(r) ((fun_reflected(r) - fun_LOS(r))*2*pi./omega).*(r>R_min).* obj.Lambda;
            value3 = @(r) (fun_reflected(r)*2*pi./omega).*(r>R_min).* obj.Lambda;
            obj.PDF_blockage = struct(field1, value1, field2, value2, field3, value3);
            field1 = 'LOS';
            field2 = 'Reflected';
            field3 = 'Total';
            value1 = @(r) 2*pi*(fun_LOS(r)*2*pi./omega).*(r>R_min).* obj.Lambda;
            value2 = @(r) 2*pi*((fun_reflected(r) - fun_LOS(r))*2*pi./omega).*(r<R_reflected & r>R_min).* obj.Lambda;
            value3 = @(r) 2*pi*(fun_LOS(r).*(r>R_reflected) + fun_reflected(r).*(r<R_reflected & r>R_min))*2*pi./omega.* obj.Lambda;
            obj.PDF_noSelfBlockage = struct(field1, value1, field2, value2, field3, value3);
            field1 = 'LOS';
            field2 = 'Reflected';
            field3 = 'Total';
            value1 = @(r) (fun_LOS(r)*2*pi./omega).*(r < R & r>R_min);
            value2 = @(r) ((fun_reflected(r) - fun_LOS(r))*2*pi./omega).*(r>R_min);
            value3 = @(r) (fun_reflected(r)*2*pi./omega).*(r>R_min);
            obj.ProbBlocked = struct(field1, value1, field2, value2, field3, value3);
            field1 = 'LOS';
            field2 = 'Reflected';
            field3 = 'Total';
            value1 = @(r) fun_LOS_clear(r).*(r < R & r>R_min);
            value2 = @(r) (fun_reflected_clear(r) - fun_LOS_clear(r)).*(r < R & r>R_min);
            value3 = @(r) fun_reflected(r).*(r < R & r>R_min);
            obj.ProbClear = struct(field1, value1, field2, value2, field3, value3);
            % simple functions pathloss
            field1 = 'LOS';
            field2 = 'Reflected';
            value1 = @(r) r.^(-obj.Para.pathloss_exponent);
            value2 = @(r) obj.Para.reflection_coeff.*sqrt(r.^2 + (2*(obj.Para.H - obj.Para.h_device)).^2).^(-obj.Para.pathloss_exponent);
            obj.PathLoss = struct(field1, value1, field2, value2);
            % import getPathLoss.m and reflection.m to compute detailed
            % interference
            field1 = 'LOS';
            field2 = 'Reflected';
            value1 = @(r) getPathLoss(r);
            value2 = @(r) getPathLoss(abs(2i.*r + 2.*(obj.Para.H - obj.Para.h_device)));
            obj.InterferenceDensity = struct(field1, value1, field2, value2);
        end
        function setDensity(obj, lambda)
            obj.Lambda = lambda;
            obj.helperGetFunctions();
        end
        function setPerturbation(obj, perturb)
            error('Not Finished yet');
            obj.Perturbation = perturb;
            obj.helperGetAuxPara();
            obj.helperGetFunctions();
        end
        function EN_LOS = getENLOS(obj, rmin, rmax)
            if rmin>=rmax
                EN_LOS = [];
                return;
            end
            EN_LOS = zeros(1,3);
            EN_LOS(1) = integral(obj.PDF_function.df_LOS, rmin, rmax);
            EN_LOS(2) = integral(obj.PDF_function.df_LOS_perturb, rmin, rmax);
            EN_LOS(3) = integral(obj.PDF_function.df_LOS_txrx_perturb, rmin, rmax);
        end
        function EN_ceiling = getENCeiling(obj, rmin, rmax)
            if rmin>=rmax
                EN_ceiling = [];
                return;
            end
            EN_ceiling = zeros(1,3);
            EN_ceiling(1) = integral(obj.PDF_function.df_ceiling, rmin, rmax);
            EN_ceiling(2) = integral(obj.PDF_function.df_ceiling_perturb, rmin, rmax);
            EN_ceiling(3) = integral(obj.PDF_function.df_ceiling_txrx_perturb, rmin, rmax);
        end
        function pdf_function = getPDF(obj) 
            pdf_function = obj.PDF_function;
        end
        function pdf_blockage = getBlockagePDF(obj) % return the pdf of interferers ONLY considering blockage
            pdf_blockage = obj.PDF_blockage;
        end
        function pdf_noSelfBlockage = getNoSelfBlockagePDF(obj)
            pdf_noSelfBlockage = obj.PDF_noSelfBlockage;
        end
        function prob_blocked = getBlockProb(obj)
            prob_blocked = obj.ProbBlocked;
        end
        function para = getPara(obj)
            para = obj.Para;
        end
        function lambda = getLambda(obj)
            lambda = obj.Lambda;
        end
        function perturbation = getPerturbation(obj)
            perturbation = obj.Perturbation;
        end
        function aux_para = getAuxPara(obj)
            aux_para = obj.AuxiliaryPara;
        end
        function reflected_r = getReflectedRadius(obj, r)
            reflected_r = r*10^(10*log10(obj.Para.reflection_coeff)./20); % PL ~ L^-2 thus use 20 in this equation
            reflected_r = sqrt(reflected_r^2 - (2*(obj.Para.H - obj.Para.h_device))^2); % The maximum radius in 2-D plane for a strong interferer which may have a strong reflection over ceiling
        end
        function [pb, pc, P_LOS] = getMarkovPara(obj, d)
            % Currently consider only LOS, tx rx also perturbed
            P_LOS = obj.PDF_function.df_LOS(d)./2./pi./d./obj.Lambda;
            pc = 1 - obj.PDF_function.df_LOS_txrx_perturb(d)./obj.PDF_function.df_LOS(d);
            pb = P_LOS.*pc./(1 - P_LOS);
        end
        function result = getSumInterference(obj, varargin)
            rmin = 0.6;
            rmax = 100;
            if nargin >0
                n = 1;
                while n<nargin
                    if strcmp(varargin{n}, 'rmin')
                        rmin = varargin{n+1};
                        n = n+1;
                    elseif strcmp(varargin{n}, 'rmax')
                        rmax = varargin{n+1};
                        n = n+1;
                    end
                    n = n+1;
                end
            end
            syms r;
            E_in_los = integral(@(r) obj.PDF_function.df_LOS(r).*obj.InterferenceDensity.LOS(r), rmin, rmax);
            E_in_ceiling = integral(@(r) obj.PDF_function.df_ceiling(r).*obj.InterferenceDensity.Reflected(r), rmin, rmax);
            result = struct();
            result.sum = E_in_los + E_in_ceiling;
        end
    end
    
end

