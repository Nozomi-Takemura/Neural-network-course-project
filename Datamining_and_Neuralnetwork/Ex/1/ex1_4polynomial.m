%% Taylor expansion
clear;
%N = 18;%N =10; N = 11;
%clear;
x = zeros(N);
y = zeros(N);
x1 = linspace(-3,3,N);
x2 = linspace(-3,3,N);
x3 = linspace(-3,3,N);
x4 = linspace(-3,3,N);
x5 = linspace(-3,3,N);
for i = 1:N;
    for j = 1:N;
        for k = 1:N;
            for l = 1:N;
                for m = 1:N;
                    x(i,j,k,l,m) = sqrt(x1(i)^2 + x2(j)^2 + x3(k)^2 + x4(l)^2 + x5(m)^2);
                    y(i,j,k,l,m) = sinc(x(i,j,k,l,m));
                end
            end
        end
    end
end
 
%Creating datasets
%Training set
train_x = zeros(5,N*N*N*N*N);
train_y = zeros(1,N*N*N*N*N);
for i = 1:N;
    for j = 1:N;
        for k = 1:N;
            for l = 1:N;
                for m = 1:N;
                    train_x(1,(i-1)*N^4+(j-1)*N^3+(k-1)*N^2+(l-1)*N+m) = x1(i);
                    train_x(2,(i-1)*N^4+(j-1)*N^3+(k-1)*N^2+(l-1)*N+m) = x2(j);
                    train_x(3,(i-1)*N^4+(j-1)*N^3+(k-1)*N^2+(l-1)*N+m) = x3(k);
                    train_x(4,(i-1)*N^4+(j-1)*N^3+(k-1)*N^2+(l-1)*N+m) = x4(l);
                    train_x(5,(i-1)*N^4+(j-1)*N^3+(k-1)*N^2+(l-1)*N+m) = x5(m);
                end
            end
        end
    end
end
for i = 1:N;
    for j = 1:N;
        for k = 1:N;
            for l = 1:N;
                for m = 1:N;
                    train_y((i-1)*N^4+(j-1)*N^3+(k-1)*N^2+(l-1)*N+m) = y(i,j,k,l,m);
                end
            end
        end
    end
end
g = [train_x];
t = [train_y];


taylor(exp(x),'order',10)
syms v w x y z
f = sinc(sqrt(v^{2}+w^{2}+x^{2}+y^{2}+z^{2}));
taylor(f, [v,w, x, y, z], 'order', 3)
taylor(f, [v,w, x, y, z], 1, 'Order', 1)
fitsurface = fit([v,w,x,y,z],f, 'poly55','Normalize','on')
X = [v,w, x, y, z];
mdl = LinearModel.fit(g(:,tr.trainInd)',t(tr.trainInd)','quadratic')
mdl = LinearModel.fit(g(:,tr.trainInd)',t(tr.trainInd)')
 [ypred,yci] = predict(mdl, g(:,tr.testInd)');
 MSE = (ypred -t(tr.testInd)')'*(ypred -t(tr.testInd)')/length(tr.testInd)
  %net = fitnet(1, 'trainlm');
    net = configure(net, g, t);
    %net.performParam.regularization = 1e-6;
    net.performParam.regularization = 1e-5;
    [net, tr] = train(net, g, t);
    inp_tyh = zeros(5,length(tr.testInd));
    

n = 2
p = polyfit(x,y,n)
[p,S] = polyfit(x,y,n)
[p,S,mu] = polyfit(x,y,n)
x = linspace(0,4*pi,10);
y = sin(x);
p = polyfit(x,y,7);


%%
