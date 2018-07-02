clear;
clf;
rng(0408);
j = 0.2;
count_k = 0;
matrix = [];
cd C:\Temp\fig14\LM_dim1;
sec = cell(4*4,1);
tic;
for m = [30,150,450,1200]
    x = linspace(-5,5,m);
    y = sinc(sqrt(x.^2));
    ynoise = sinc(sqrt(x.^2)) + j*randn(size(x));
    for k = [1,3,5,7]
        count_k = count_k + 1;
        net=fitnet(1,'trainlm');
        [net,tr] = train(net,x,ynoise);
        train_yhat=net(x(tr.trainInd));
        val_yhat=net(x(tr.valInd));
        test_yhat = net(x(tr.testInd));
        mat_tra = [sinc(sqrt(x(tr.trainInd).^2))', train_yhat'];
        %plot(sinc(sqrt(x(tr.trainInd).^2)),train_yhat,'r*');
        h = figure; set(h, 'Visible', 'off');
        plot(mat_tra(:,1), mat_tra(:,2), 'r*');
        hold on;
        mat_test = [sinc(sqrt(x(tr.testInd).^2))', test_yhat'];
     %plot(sinc(sqrt(x(tr.trainInd).^2)),train_yhat,'r*');
        plot(mat_test(:,1), mat_test(:,2), 'bo');
        hold off;
        legend('Training set', 'Test set');
        title(['Approxmiated function against sinc(r): Levenberg-Marquardt with the number of traing data=' num2str(m) ', amount of noise = 0.1, number of hidden units=' num2str(k) ', dimension = 1']);
        saveas(h,sprintf('LM_data%i_noise01_huni_%i_yhat_vs_sinc.eps',m, k),'epsc'); % will create FIG1, FIG2,...
        %hold off;
        u = count_k;
        matrix(u,1) = m;
        %matrix(u,2) = j;
        matrix(u,2) = k;
        %matrix(u,4) = p;
        if strcmp(tr.stop,'Minimum gradient reached.')
            matrix(u,3) = 0;
        elseif strcmp(tr.stop, 'Validation stop.')
            matrix(u,3) = 1;
        else
            matrix(u,3) = 2;
        end
        seq{u} = tr.stop;
        disp(matrix(u,3));
        tr.stop
        %matrix(u,6) = net.IW{1}(1);
        %qq(1,i) = net.IW{1}(2);
        matrix(u,4) = tr.best_perf;
        matrix(u,5) = tr.best_vperf;
        matrix(u,6) = tr.best_epoch;
        matrix(u,7) = tr.time(length(tr.epoch));
    %train_ind = tr.trainInd;
%val_ind = tr.valInd;
%test_ind = tr.testInd;
    end
end
t = toc;
save('C:\Temp\fig14\LM_dim1\stat.mat', 'matrix')
save('C:\Temp\fig14\LM_dim1\stat.mat', 't')
save('C:\Temp\fig14\LM_dim1\stat.mat', 'seq')
save('C:\Temp\fig14\LM_dim1\all.mat')

%%
