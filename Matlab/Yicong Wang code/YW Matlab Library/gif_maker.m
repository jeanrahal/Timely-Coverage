% 09/16/2015 Legacy Jiaxiao Zheng
% 09/16/2015 Make gif. Legacy: Jiaxiao
% TODO: Add ask before overwrite
classdef gif_maker < handle
    properties
        frame_num = 0;
        file_name;
        DelayTime = 0.7;
    end
    methods
        function obj = gif_maker()
        end
        function setFileName(obj, name)
            obj.file_name = name;
        end
        function setDelayTime(obj, delay)
            obj.DelayTime = delay;
        end
        function addFrame(obj, figure_handle)
            frame = getframe(figure_handle);
            im = frame2im(frame);
            [I, map] = rgb2ind(im, 256);
            if obj.frame_num == 0
                obj.frame_num = obj.frame_num + 1;
                imwrite(I, map, obj.file_name, 'gif', 'Loopcount', 0, 'DelayTime', 2*obj.DelayTime);
            else
                imwrite(I, map, obj.file_name, 'gif', 'WriteMode', 'append', 'DelayTime', obj.DelayTime);
            end
        end
    end
end

