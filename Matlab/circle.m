function h = circle(x,y,r,color,line,l)
hold on
th = 0:pi/50:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
h = plot(xunit, yunit,color,line,l);
hold off