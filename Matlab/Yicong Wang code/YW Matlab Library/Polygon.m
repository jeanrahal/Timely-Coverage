classdef Polygon < handle
    %POLYGON a set of Segments
    properties
        seg_sets;
        seg_num;
    end
    
    methods
        function obj = Polygon(point_set) % need to be in sequence
            if iscell(point_set)
                obj.seg_num = length(point_set);
                assert(obj.seg_num>2);
                obj.seg_sets = Segment.empty(obj.seg_num, 0);
                for i = 1:obj.seg_num - 1
                    obj.seg_sets(i) = Segment(point_set{i}, point_set{i+1});
                end
                obj.seg_sets(obj.seg_num) = Segment(point_set{obj.seg_num}, point_set{1}); 
            else
                [h, w] = size(point_set);
                assert(w == 2);
                obj.seg_num = h;
                assert(obj.seg_num>2);
                obj.seg_sets = Segment.empty(obj.seg_num, 0);
                for i = 1:obj.seg_num - 1
                    obj.seg_sets(i) = Segment(point_set(i,:), point_set(i+1,:));
                end
                obj.seg_sets(obj.seg_num) = Segment(point_set(obj.seg_num,:), point_set(1,:));
            end
            obj.seg_sets = obj.seg_sets(:)';
        end
        function sides = getSides(obj)
            sides = obj.seg_sets;
        end
        function side_num = getSideNum(obj)
            side_num = obj.seg_num;
        end
        function shift(obj, vec)
            assert(isequal(size(vec),[1,2]));
            for seg = obj.seg_sets(:)'
                seg.shift(vec);
            end
        end
        function plotPolygon(obj, ha)
            for seg = obj.seg_sets(:)'
                seg.plotSegment(ha);
            end
        end
    end
    
end

