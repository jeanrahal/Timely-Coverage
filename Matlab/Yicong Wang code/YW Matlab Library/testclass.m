% 09/18/2015 test whether constructor can call functions
classdef testclass < handle
    properties
        a;
    end
    methods
        function obj = testclass()
            obj.a = 1;
            obj.helper();
        end
        function helper(obj)
            display(['Works:',num2str(obj.a)]);
        end
    end
end