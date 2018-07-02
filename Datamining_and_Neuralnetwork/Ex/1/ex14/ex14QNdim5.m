%% 
clear;
clf;
rng(0408);
j = 0.2;
count_k = 0;
matrix_sum = [];
cd C:\Temp\fig14\LM_dim5;
sec = cell(4*4,1);
tic; %m = 10
for m = [30,150,450,1200]
    %x = linspace(-5,5,m);
    y = zeros(1,m);
    x = zeros(5,m);
    input = zeros(2,m);
    x(1,:) = linspace(-5,5,m); %m = 10
    x(2,:) = linspace(-5,5,m); %m = 10
    x(3,:) = linspace(-5,5,m); %m = 10
    x(4,:) = linspace(-5,5,m); %m = 10
    x(5,:) = linspace(-5,5,m); %m = 10
    %x = input;
    for i = 1:m%m =10
        %y(1,i) = sinc(sqrt(x(1,i).^2+x(2,i).^2)) + j*randn(size(x));
        %y(1,i) = sinc(sqrt(x(1,i).^2+x(2,i).^2));
        y(1,i) = sinc(norm(x(:,i),2));
    end
    %sinc(sqrt(norm(inp_tyh(:,1),2)))
    %sinc(sqrt(norm(inp_yh(:,1),2)))
    ynoise = y  + j*randn(size(y));
    for k = [1,3,5,7]%k=3
        count_k = count_k + 1;
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