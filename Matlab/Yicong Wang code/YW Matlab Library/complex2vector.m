function [ vec ] = complex2vector(complex_numbers)
%COMPLEX2VECTOR covert complex number to vectors
if isvector(complex_numbers)
    vec = [real(complex_numbers(:)), imag(complex_numbers(:))];
elseif isscalar(complex_numbers)
    vec = [real(complex_numbers), imag(complex_numbers)];
else
    error('Invalid input');
end

end

