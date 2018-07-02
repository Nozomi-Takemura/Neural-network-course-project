%Five-dimensional function (m = 5)
%Calculating input-output
%For training set
N = 18;%N =10;
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

%%
count_k = 0;
for k = [1,3,5,7,27]%k = 100 % k=5
        count_k = count_k + 1;
        
        net = fitnet(k, 'trainbfg');
        net = configure(net, g, t);
        %net.performParam.regularization = 1e-6;
        net.performParam.regularization = 1e-5;
        [net, tr] = train(net, g, t);
        inp_tyh = zeros(5,length(tr.testInd));
        test_yhat=net(g(:,tr.testInd));
        %test_yhat = net(train_x);
        mat_test = [t(tr.testInd)', test_yhat'];
        %h = figure; set(h, 'Visible', 'off');
        %plot(mat_tra(:,1), mat_tra(:,2), 'r*');
        hold on;
        %[Xtr,Ytr] = meshgrid(inp_yh(1,:),inp_yh(2,:));
        %[Xte,Yte] = meshgrid(inp_tyh(1,:),inp_tyh(2,:));
        %plot(sinc(sqrt(x(tr.trainInd).^2)),train_yhat,'r*');
        figure;
        plot(mat_test(:,1), mat_test(:,2), 'bo');
        hold off;
        legend('Test set', 'Location', 'Best');
        xlabel('Sinc(r)');
        ylabel('Approximated function')
        title(['QN with #input =' num2str(m) ', #hidden units=' num2str(k) ', dimension = 5']);
        %saveas(h,sprintf('LM_data%i_noise01_huni_%i_yhat_vs_sinc.eps',m, k),'epsc'); % will create FIG1, FIG2,...
        MSE = perform(net,t(tr.testInd),test_yhat)
        tr.time(end)
        tr.stop
        cd '/home/nozomi/Documents/2017courses/DMNN/Ex/1/fig14/QN5';
        save('QN_N10_NH100.mat')
end        
        

%{
        net=fitnet(k,'trainlm');
        [net,tr] = train(net,x,ynoise);
        inp_yh = zeros(5,length(tr.trainInd));
        for p = 1:length(tr.trainInd)
            inp_yh(:,p) = x(:,tr.trainInd(p));
        end
        train_yhat=net(inp_yh);
        %test set
        inp_tyh = zeros(5,length(tr.testInd));
        for p = 1:length(tr.testInd)
            inp_tyh(:,p) = x(:,tr.testInd(p));
        end
        test_yhat=net(inp_tyh);
        %plot-for-train set
        %sinc(sqrt(norm(inp_tyh(:,1),2)))
        %sinc(sqrt(norm(inp_yh(:,1),2)))
        
        mat_tra = [y(tr.trainInd)', train_yhat'];
        %mat_tra = [sinc(sqrt(x(tr.trainInd).^2))', train_yhat'];
        %mat_test = [sinc(sqrt(x(tr.testInd).^2))', test_yhat'];
        mat_test = [y(tr.testInd)', test_yhat'];
        %plot(sinc(sqrt(x(tr.trainInd).^2)),train_yhat,'r*');
        h = figure; set(h, 'Visible', 'off');
        plot(mat_tra(:,1), mat_tra(:,2), 'r*');
        hold on;
        %[Xtr,Ytr] = meshgrid(inp_yh(1,:),inp_yh(2,:));
        %[Xte,Yte] = meshgrid(inp_tyh(1,:),inp_tyh(2,:));
         %plot(sinc(sqrt(x(tr.trainInd).^2)),train_yhat,'r*');
        plot(mat_test(:,1), mat_test(:,2), 'bo');
        hold off;
        legend('Training set', 'Test set', 'Location', 'Best');
        xlabel('Sinc(r)');
        ylabel('Approximated function')
        title(['Approximated function against sinc(r): Levenberg-Marquardt with the number of traing data=' num2str(m) ', amount of noise = 0.1, number of hidden units=' num2str(k) ', dimension = 5']);
        saveas(h,sprintf('LM_data%i_noise01_huni_%i_yhat_vs_sinc.eps',m, k),'epsc'); % will create FIG1, FIG2,...
        %hold off;
        u = count_k;
        matrix_sum(u,1) = m;
        %matrix(u,2) = j;
        matrix_sum(u,2) = k;
        %matrix(u,4) = p;
        if strcmp(tr.stop,'Minimum gradient reached.')
            matrix_sum(u,3) = 0;
        elseif strcmp(tr.stop, 'Validation stop.')
            matrix_sum(u,3) = 1;
        else
            matrix_sum(u,3) = 2;
        end
        seq{u} = tr.stop;
        disp(matrix_sum(u,3));
        tr.stop
        %matrix(u,6) = net.IW{1}(1);
        %qq(1,i) = net.IW{1}(2);
        matrix_sum(u,4) = tr.best_perf;
        matrix_sum(u,5) = tr.best_vperf;
        matrix_sum(u,6) = tr.best_epoch;
        matrix_sum(u,7) = tr.time(length(tr.epoch));
    %train_ind = tr.trainInd;
%val_ind = tr.valInd;
%test_ind = tr.testInd;
    end
end
t = toc;
save('C:\Temp\fig14\LM_dim5\stat.mat', 'matrix_sum')
save('C:\Temp\fig14\LM_dim5\stat.mat', 't')
save('C:\Temp\fig14\LM_dim5\stat.mat', 'seq')
save('C:\Temp\fig14\LM_dim5\all.mat')
%%
%%

net = fitnet(27, 'trainscg');
net = configure(net, g, t);
net.performParam.regularization = 1e-6;
%net.divideFcn = 'divideind';
%net.divideParam = struct('trainInd', 1:175, ...
                         %'valInd', 176:250, ...
                         %'testInd', []);
%net.initFcn = 'initlay';
%net.layers{1}.initFcn = 'initwb';
%net.inputWeights{1,1}.initFcn = 'randnr';
 
[net, tr] = train(net, g, t);
test_yhat = net(train_x);
 
figure
plot3(train_x(1,:), train_x(2,:), train_y, 'g*');
hold on;
plot3(train_x(1,:), train_x(2,:), test_yhat, 'r-');
hold off;
legend('Training Set', 'Approximated Function');
 
figure
plot3(train_x(2,:), train_x(3,:), train_y, 'y*');
hold on;
plot3(train_x(2,:), train_x(3,:), test_yhat, '-');
hold off;
legend('Training Set', 'Approximated Function');
}%