%Mexican hat function (m = 2)
%Calculating input-output
%For training set
%N = 40;  N = 316; (^{2}~100000)
x = zeros(N);
y = zeros(N);
x1 = linspace(-4,4,N);
x2 = linspace(-4,4,N);
for i = 1:N;
    for j = 1:N;
        x(i,j) = sqrt(x1(i)^2 + x2(j)^2);
        y(i,j) = sinc(x(i,j));
    end
end
%For validation set
n = 25;
k = zeros(n);
l = zeros(n);
k1 = linspace(-3.8,3.8,n);
k2 = linspace(-3.8,3.8,n);
for i = 1:n;
    for j = 1:n;
        k(i,j) = sqrt(k1(i)^2 + k2(j)^2);
        l(i,j) = sinc(k(i,j));
    end
end
%For test set
m = 25;
s = zeros(m);
t = zeros(m);
s1 = linspace(-4,4,m);
s2 = linspace(-4,4,m);
for i = 1:m;
    for j = 1:m;
        s(i,j) = sqrt(s1(i)^2 + s2(j)^2);
        t(i,j) = sinc(s(i,j));
    end
end
 
%Creating datasets
%Training set
train_x = zeros(2,N*N);
train_y = zeros(1,N*N);
for i = 1:N;
    for j = 1:N;
        train_x(1,(i-1)*N+j) = x1(i);
        train_x(2,(i-1)*N+j) = x2(j);
    end
end
for i = 1:N;
    for j = 1:N;
        train_y((i-1)*N+j) = y(i,j);
    end
end
%Validation set
val_x = zeros(2,n*n);
val_y = zeros(1,n*n);
for i = 1:n;
    for j = 1:n;
        val_x(1,(i-1)*n+j) = k1(i);
        val_x(2,(i-1)*n+j) = k2(j);
    end
end
for i = 1:n;
    for j = 1:n;
        val_y((i-1)*n+j) = l(i,j);
    end
end
%Test set
test_x = zeros(2,m*m);
test_y = zeros(1,m*m);
for i = 1:m;
    for j = 1:m;
        test_x(1,(i-1)*m+j) = s1(i);
        test_x(2,(i-1)*m+j) = s2(j);
    end
end
for i = 1:m;
    for j = 1:m;
        test_y((i-1)*m+j) = t(i,j);
    end
end
g = [train_x, val_x];
t = [train_y, val_y];
 
net = fitnet(27, 'trainlm');
net = configure(net, g, t);
net.performParam.regularization = 1e-6;
net.divideFcn = 'divideind';
net.divideParam = struct('trainInd', 1:1600, ...
                         'valInd', 1601:2225, ...
                         'testInd', []);
%net.initFcn = 'initlay';
%net.layers{1}.initFcn = 'initwb';
%net.inputWeights{1,1}.initFcn = 'randnr';
 
[net, tr] = train(net, g, t);
test_yhat = net(test_x);
 
plot3(test_x(1,:), test_x(2,:), test_y, 'g*');
hold on;
plot3(test_x(1,:), test_x(2,:), test_yhat, 'r-');
hold off;
legend('Training Set', 'Approximated Function');
MSE = perform(net,test_y,test_yhat)
