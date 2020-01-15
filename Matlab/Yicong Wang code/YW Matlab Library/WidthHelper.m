function [w] = WidthHelper(d1, d2)
% 05/17/2015 This function returns the addtional width induced by particulat TX/RX perturb
% Notice: Only servers small perturbation, e.g., range = d_blockage
    if(d1 == 0)
        w = (d2>0).*d2./2;
    elseif (d2 == 0)
        w = (d1>0).*d1./2;
    elseif(d1 == d2 & d1 < 0)
        w = 0;
    elseif(d1 == d2 & d1 >0)
        w = d1;
    else
        w = (d1>0 & d2>0).*(d1+d2)./2 + (d1<0 & d2<0).*0 + (d1>0& d2<0).*d1.^2./2./(d1-d2) + (d2>0 & d1<0).*d2.^2./2./(d2-d1);
    end
end

