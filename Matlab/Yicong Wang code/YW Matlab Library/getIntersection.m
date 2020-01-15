function [t1, t2] = getIntersection(point1, v1, point2, v2)
%GETINTERSECTION return the time till intersection
v_cross = det([v1;v2]);
pos = point1 - point2;
if norm(v_cross) == 0
    if det([v1; pos]) == 0 && det([v2; pos]) == 0
        t1 = 0;
        t2 = norm(pos);
    else
        t1 = inf;
        t2 = inf;
    end
else
    t1 = -det([pos; v2])./v_cross;
    t1 = t1(1);
    t2 = -det([pos; v1])./v_cross;
    t2 = t2(1);
end

end

