function [ result ] = bool2char(input_logical)
%BOOL2CHAR turn a single logical to string
assert(islogical(input_logical) && isscalar(input_logical));
tf_words = {'false','true'};
result =  tf_words{input_logical+1};
end

