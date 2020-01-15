function [isInPolygon] = isPointInPolygon(point, seg_sets)
%GETPOINTINPOLYGON 
num_positive = 0;
num_negative = 0;
if isa(seg_sets, 'Polygon')
    seg_sets = seg_sets.getSides();
else
    assert(isa(seg_sets(1), 'Segment'));
end
for seg = seg_sets(:)'
    temp_point = seg.getEndPoint1();
    temp_n = seg.getNormalVector();
    temp_val = dot(point - temp_point, temp_n);
    if temp_val>0
        num_positive = num_positive + 1;
    elseif temp_val<0
        num_negative = num_negative + 1;
    end
end
isInPolygon = (num_positive == length(seg_sets) || num_negative == length(seg_sets));

