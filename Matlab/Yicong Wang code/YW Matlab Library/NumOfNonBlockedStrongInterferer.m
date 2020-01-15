function [EN_LOS, EN_ceiling, pdf_fun] = NumOfNonBlockedStrongInterferer(LambdaSet, para, perturbation)
% 09/18/2015 Copy to YW Matlab Library, change to a class and modify
% parameters
% 05/16/2015 Compute the expected number of strong
%interfererers given that there is perturbation, EN_LOS = [EN_LOS, EN_LOS_pertubed, proportion]
% Return pdf_fun, the struct of density (not normalized as pdf!) functions
% for Strong Interferers with different distance away
% 05/18/2015 Adding the case for LOS, with TX and RX are perturbed with all
% other points, modified EN_LOS
% 08/11/2015 Modify para, allow para to contain some fields
% Initialization of Parameters
R = 13;
R_min = 0.6;
R_max = R;
d = sqrt(0.25*0.46); % Model human body by ellipse with a = 0.33, b = 0.46, equivalent d is sqrt(a*b)
omega = 2*pi*2/3;
H = 2.8; % Height of ceiling
h_body = 1.524; % height of body
h = 1.754; % total heigh of person
h_device = 1.0; % height of device
reflection_coeff = 0.3; % reflection coefficient
d_body = d;
d_head = sqrt(0.25*0.15);
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
R_reflected = R*10^(10*log10(reflection_coeff)./20); % PL ~ L^-2 thus use 20 in this equation
R_reflected = sqrt(R_reflected^2 - (2*(H-h_device))^2); % The maximum radius in 2-D plane for a strong interferer which may have a strong reflection over ceiling

R = min(R, R_max);
R_reflected = min(R_reflected, R_max);

syms r theta;
d_body_perturb = DBlockedAfterPerturbation(TRANSLATION_RANGE, d_body);
d_head_perturb = DBlockedAfterPerturbation(TRANSLATION_RANGE, d_head);

pdf_perturb = @(r, theta) 1./ pi./TRANSLATION_RANGE.^2;
d_body_txrx_perturb = WidthTXRXPerturbation(d_body, d_body + d_body_perturb, pdf_perturb, TRANSLATION_RANGE);
d_head_txrx_perturb = WidthTXRXPerturbation(d_head, d_head + d_head_perturb, pdf_perturb, TRANSLATION_RANGE);

rotation_dist = @(theta) 1./(2*ROTATION_RANGE).*(abs(theta)<ROTATION_RANGE); % Valid for range < pi
Prob_rotation_perturb = ProbFacingAfterRotation(omega, rotation_dist); % Probability that RX TX are still facing each other after random rotation

if(isscalar(LambdaSet))
    fun_reflected = @(r) omega ./ 2./ pi .*exp(-( (h_body - h_device).*d_body + (h - h_body).*d_head)./(H - h_device).*LambdaSet.*r).*r;
    fun_LOS = @(r) omega./2./pi.*exp(-d_body.*LambdaSet.*r).*r ;
    fun_LOS_perturb = @(r) omega./2./pi.*exp(-(d_body + d_body_perturb).*LambdaSet.*r).*r;
    fun_reflected_perturb = @(r) omega ./ 2./ pi .*exp(-( (h_body - h_device).*(d_body + d_body_perturb) + (h - h_body).*(d_head + d_head_perturb))./(H - h_device).*LambdaSet.*r).*r;
    fun_LOS_txrx_perturb = @(r)omega./2./pi.*exp(-(d_body + d_body_txrx_perturb).*LambdaSet.*r).*r;
    fun_reflected_txrx_perturb = @(r) omega ./ 2./ pi .*exp(-( (h_body - h_device).*(d_body + d_body_txrx_perturb) + (h - h_body).*(d_head + d_head_txrx_perturb))./(H - h_device).*LambdaSet.*r).*r;
%     EN_LOS = zeros(1,5);
    EN_ceiling = zeros(1,4);
    EN_ceiling(1) = omega*LambdaSet*(integral(fun_reflected, R_min, R_reflected) + integral(fun_LOS, R_reflected, R));
    EN_ceiling(2) = Prob_rotation_perturb*omega*LambdaSet*(integral(fun_reflected_perturb, R_min, R_reflected) + integral(fun_LOS_perturb, R_reflected, R));
    EN_ceiling(3) = EN_ceiling(2)./EN_ceiling(1);
    EN_ceiling(4) = Prob_rotation_perturb*omega*LambdaSet*(integral(fun_LOS_txrx_perturb, R_min, R_reflected)) + ...
        Prob_rotation_perturb*omega*LambdaSet*(integral(fun_LOS_txrx_perturb, R_reflected, R));
    EN_LOS = zeros(1,5);
    EN_LOS(1) = omega*LambdaSet*(integral(fun_LOS, R_min, R));
    EN_LOS(2) = Prob_rotation_perturb*omega*LambdaSet*(integral(fun_LOS_perturb, R_min, R));
    EN_LOS(4) = EN_LOS(2)./EN_LOS(1);
    EN_LOS(3) = Prob_rotation_perturb*omega*LambdaSet*(integral(fun_LOS_txrx_perturb, R_min, R));
    EN_LOS(5) = EN_LOS(3)./ EN_LOS(1);
    field1 = 'df_LOS';
    field2 = 'df_LOS_perturb';
    field3 = 'df_LOS_txrx_perturb';
    field4 = 'df_ceiling';
    field5 = 'df_ceiling_perturb';
    value1 = @(r) (fun_LOS(r) .*omega .* LambdaSet.* (r < R & r>R_min));
    value2 = @(r) Prob_rotation_perturb*fun_LOS_perturb(r).*omega.*LambdaSet .* (r < R & r>R_min);
    value3 = @(r) Prob_rotation_perturb*fun_LOS_txrx_perturb(r).*omega.*LambdaSet .* (r < R & r>R_min);
    value4 = @(r) (fun_reflected(r).*(r<R_reflected & r>R_min) + fun_LOS(r).* (r>= R_reflected & r < R)).* omega .*LambdaSet;
    value5 = @(r) Prob_rotation_perturb*(fun_reflected_perturb(r).*(r<R_reflected & r>R_min) + fun_LOS_perturb(r).* (r>= R_reflected & r < R)).* omega.* LambdaSet;
    pdf_fun = struct(field1, value1, field2, value2, field3, value3, field4, value4, field5, value5);
else
    EN_LOS = zeros(length(LambdaSet(:)),5);
    EN_ceiling = zeros(length(LambdaSet(:)),3);
    for i = 1:length(LambdaSet(:))
        lambda = LambdaSet(i);
        fun_reflected = @(r) omega ./ 2./ pi .*exp(-( (h_body - h_device).*d_body + (h - h_body).*d_head)./(H - h_device).*lambda.*r).*r;
        fun_LOS = @(r) omega./2./pi.*exp(-d_body.*lambda.*r).*r ;
        fun_LOS_perturb = @(r) omega./2./pi.*exp(-(d_body + d_body_perturb).*lambda.*r).*r;
        fun_reflected_perturb = @(r) omega ./ 2./ pi .*exp(-( (h_body - h_device).*(d_body + d_body_perturb) + (h - h_body).*(d_head + d_head_perturb))./(H - h_device).*lambda.*r).*r;
        fun_LOS_txrx_perturb = @(r)omega./2./pi.*exp(-(d_body + d_body_txrx_perturb).*lambda.*r).*r;

        EN_ceiling(i,1) = omega*lambda*(integral(fun_reflected, R_min, R_reflected) + integral(fun_LOS, R_reflected, R));
        EN_ceiling(i,2) = Prob_rotation_perturb*omega*lambda*(integral(fun_reflected_perturb, R_min, R_reflected) + integral(fun_LOS_perturb, R_reflected, R));
        EN_ceiling(i,3) = EN_ceiling(i,2)./EN_ceiling(i,1);
        
        EN_LOS(i,1) = omega*lambda*(integral(fun_LOS, R_min, R));
        EN_LOS(i,2) = Prob_rotation_perturb*omega*lambda*(integral(fun_LOS_perturb, R_min, R));
        EN_LOS(i,4) = EN_LOS(i,2)./EN_LOS(i,1);
        EN_LOS(i,3) = Prob_rotation_perturb*omega*lambda*(integral(fun_LOS_txrx_perturb, R_min, R));
        EN_LOS(i,5) = EN_LOS(i,3)./ EN_LOS(i,1);
    end
    pdf_fun = struct(); % Empty Struct
end

end



