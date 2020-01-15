function [interval] = getSegmentIntersectionTime(seg0,speed0, seg1, speed1)
% get the intersection time of two moving lines
n0 = seg0.getNormalVector();
u0 = seg0.getUnitVector();
len0 = seg0.getLength();
relative_v = speed1 - speed0;
points0 = seg0.getEndPoint();
points1 = seg1.getEndPoint();
X0 = points0{1};
Y0 = points0{2};
X1 = points1{1};
Y1 = points1{2};
[lx, tx] = getIntersection(X0, u0, X1, relative_v);
[ly, ty] = getIntersection(X0, u0, Y1, relative_v);
if ly < lx
    t = tx;
    tx = ty;
    ty = t;
    l = lx;
    lx = ly;
    ly = l;
end
    

if ly<0 ||lx>len0
    if lx == inf && det([relative_v; u0]) == 0
        u1 = seg1.getUnitVector();
        len1 = seg1.getLength();
        [t1, t2] = getIntersection(X0, u0, X1, u1);
        if t2 < 0 || t2 > len1
            interval = [inf, inf];
        else
            if norm(relative_v) == 0
                interval = [-inf, inf];
            else
                temp_v = relative_v./u0;
                temp_v = temp_v(1);
                interval = [-t1./temp_v, (len0 - t1)./temp_v];
            end
        end
    else
        interval = [inf, inf];
    end
elseif lx>=0 && ly<=len0
    interval = [tx, ty];
else
    t1 = tx;
    t2 = ty;
    if lx < 0
        t1 = t2 - ly./(ly-lx).*(ty-tx);
    end
    if ly > len0
        t2 = tx + (len0 - lx)./(ly-lx).*(ty-tx);
    end
    interval = [t1, t2];
end
interval = [min(interval), max(interval)];

end

