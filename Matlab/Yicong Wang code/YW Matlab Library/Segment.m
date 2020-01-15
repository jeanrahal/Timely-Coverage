classdef Segment < handle
    %LINE Summary of this class goes here
    % 1/22/2017 : add getIntersection function
    
    properties
        endpoint = {[0,0], [1, 0]};
        len = 1;
        n = [0, 1];
        v = [1, 0];
    end
    methods
        function obj = Segment(varargin)
            if(nargin == 2)
                if ~isrow(varargin{1}) || ~isrow(varargin{2})
                    warning('Invalid input');
                    return;
                end
                if length(varargin{1})~= 2 || length(varargin{2})~=2 %TODO: may expand for 3D case
                    warning('Invalid input point');
                    return;
                end
                obj.endpoint = {varargin{1}, varargin{2}};
                obj.len = norm(obj.endpoint{2}-obj.endpoint{1});
                if obj.len == 0
                    % warning('Length is 0');
                else
                    temp = obj.endpoint{2}-obj.endpoint{1};
                    obj.v = temp./obj.len;
                    obj.n = [-obj.v(2), obj.v(1)];
                end
            end
        end
        function points = getEndPoint(obj)
            points = obj.endpoint;
        end
        function point = getEndPoint1(obj)
            point = obj.endpoint{1};
        end
        function point = getEndPoint2(obj)
            point = obj.endpoint{2};
        end
        function l = getLength(obj)
            l = obj.len;
        end
        function n = getNormalVector(obj)
            n = obj.n;
        end
        function v = getUnitVector(obj)
            v = obj.v;
        end
        function isOnSegment = getIsOnSegment(obj, point)
            epsilon = 1e-8;
            isOnSegment = false;
            point_v = point - obj.endpoint{1};
            if (abs(dot(point_v, obj.n)) <= epsilon)
                temp_len = dot(point_v,obj.v);
                if(temp_len>=0 && temp_len<=obj.len)
                    isOnSegment = true;
                end
            end
        end
        function shift(obj, vec)
            obj.endpoint{1} = obj.endpoint{1} + vec;
            obj.endpoint{2} = obj.endpoint{2} + vec;
        end
        function isOnDifferentSide = getIsOnDifferentSide(obj, seg)
            seg_end = seg.getEndPoint();
            temp_vec1 = seg_end{1} - obj.endpoint{1};
            temp_vec2 = seg_end{2} - obj.endpoint{1};
            temp_dot1 = dot(temp_vec1, obj.n);
            temp_dot2 = dot(temp_vec2, obj.n);
            isOnDifferentSide = (temp_dot1*temp_dot2<0);    
        end
        function isIntersecting = getIsIntersecting(obj, seg)
            seg_end = seg.getEndPoint();
            if (obj.getIsOnDifferentSide(seg) && seg.getIsOnDifferentSide(obj))
                isIntersecting = true;
            else
                isIntersecting = (obj.getIsOnSegment(seg_end{1})|| obj.getIsOnSegment(seg_end{2}) || seg.getIsOnSegment(obj.endpoint{1}) || seg.getIsOnSegment(obj.endpoint{2}));
            end
        end
        function [intersection, dist2end1]= getIntersection(obj, seg)
            intersection = [];
            dist2end1 = obj.len;
            seg_end = seg.getEndPoint();
            if (obj.getIsOnDifferentSide(seg) && seg.getIsOnDifferentSide(obj))
                temp_vec1 = seg_end{1} - obj.endpoint{1};
                temp_vec2 = seg_end{1} - obj.endpoint{2};
                temp_dot1 = dot(temp_vec1, seg.n);
                temp_dot2 = dot(temp_vec2, seg.n);
                dist2end1 = obj.len.*(temp_dot1./(temp_dot1 - temp_dot2));
                intersection = obj.endpoint{1} + obj.v.*dist2end1;
            else
                if obj.getIsOnSegment(seg_end{1})
                    intersection = seg_end{1};
                    dist2end1 = norm(seg_end{1}-obj.endpoint{1});
                elseif obj.getIsOnSegment(seg_end{2}) 
                    intersection = seg_end{2};
                    dist2end1 = norm(seg_end{1}-obj.endpoint{1});
                elseif seg.getIsOnSegment(obj.endpoint{1})
                    intersection = obj.endpoint{1};
                    dist2end1 = 0;
                elseif seg.getIsOnSegment(obj.endpoint{2})
                    intersection = obj.endpoint{2};
                    dist2end1 = obj.len; % notice: endpoint 1 not on seg
                end
            end
        end
        function dist = getDistToPoint(obj, point)
            temp_vec = point - obj.endpoint{1};
            temp_proj = dot(temp_vec, obj.v);
            if (temp_proj < 0)
                dist = norm(point - obj.endpoint{1});
            elseif temp_proj > obj.len
                dist = norm (point - obj.endpoint{2});
            else
                dist = abs(dot(temp_vec, obj.n));
            end
        end
        function plotSegment(obj, ha)
            point1 = obj.getEndPoint1();
            point2 = obj.getEndPoint2();
            plot(ha, [point1(1), point2(1)], [point1(2), point2(2)],'b');
        end
    end
    
end

