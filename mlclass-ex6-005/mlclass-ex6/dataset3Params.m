function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%
C = 0;
sigma = 0;
error = 100000000;
% You need to return the following variables correctly.
ct = [0.01,0.03,0.1, 0.3, 1, 3,10,30];
sigmat = [0.01,0.03,0.1, 0.3, 1, 3,10,30];
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
for ii = 1:size(ct,1)
    for jj = 1:size(sigmat,1)
        model = svmTrain(X, y, ct(1,ii), @(x1, x2) gaussianKernel(x1, x2, sigmat(1,jj))); 
        predictions = svmPredict(model, Xval);
        me =  mean(double(predictions ~= yval));
        if me < error
            error = me;
            C = ct(1,ii);
            sigma = sigmat(1,jj);
        end
    end
end

% =========================================================================

end
