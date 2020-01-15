function [ isIntersecting ] = isSegmentIntersectPolygon(seg, poly)
%GETSEGMENTINTERSECTPOLYGON return whether a segment is intersecting with
% a polygon consisting of a set of Segments
if isa(poly, 'Polygon')
    seg_sets = poly.getSides();
else
    seg_sets = poly;
    assert(isa(seg_sets(1), 'Segment'));
end
isIntersecting = false;
if(isPointInPolygon(seg.getEndPoint1(), seg_sets) || isPointInPolygon(seg.getEndPoint2(), seg_sets))
    isIntersecting = true;
else
    for side = seg_sets(:)'
        if side.getIsIntersecting(seg)
            isIntersecting = true;
            break;
        end
    end
end

end

