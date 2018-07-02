%%
clear;
rng(0408);
count_p = 0;
matrix_sum = zeros(5*6,12);
load('/home/nozomi/Documents/2017courses/DMNN/Ex/Data/bcw.mat');
cd /home/nozomi;
cd package/;
addpath(pwd);
cd '/home/nozomi/Documents/2017courses/DMNN/Ex/2/fig23/CGW';
sec = cell(5*6,1);
%tic; %m = 10
%view(net)
for k = [1,5,10,30,100]%k=3
    net = patternnet(k,'trainscg');
    net = configure(net,X,Y);
    iniw = net.IW;
    iniLW = net.LW;
    iniw{1} = ones(k,30)
    iniLW{2} = ones(1,k)
    for p=[0, 0.1, 0.01, 0.001, 0.0001, 0.00001]%p=0.000001
        count_p = count_p + 1;
        pp = round(p*100000);
        net.performParam.regularization = p;
        [net tr] = train(net,X,Y);
        inp_tyh = X(:,tr.testInd);
        
         %tf = isequal(ppp,inp_tyh)
        test_yhat=net(inp_tyh);
        t = net(X);
        perf_test = perform(net,Y(tr.testInd),test_yhat);
        perf_full = perform(net,Y,t);
        classes_test = vec2ind(test_yhat);
        classes_full = vec2ind(t);
        percError_test = sum(abs(Y(tr.testInd)-classes_test))/(length(classes_test));
        percError_full = sum(abs(Y-classes_full))/(length(classes_full));
        
        g = figure; set(g, 'Visible', 'off');
        simpleclusterOutputs = sim(net,X);
        plotroc(Y,simpleclusterOutputs); 
        title({['ROC curve (full data): scaled conjugate gradient, #neurons =' num2str(k) ','] ; ['regulatization term=' num2str(p) ', all initial weights=1']});
        saveas(g,sprintf('roc_fulldata_iw1_scg_%ihuni_%ireg.eps', k, pp), 'epsc'); % will create FIG1, FIG2,...
        close(g)
        
        h = figure; set(h, 'Visible', 'off');
        simpleclusterOutputs_2 = sim(net,inp_tyh);
        plotroc(Y(tr.testInd),simpleclusterOutputs_2);
        title({['ROC curve (test data): scaled conjugate gradient, #neurons =' num2str(k) ','] ; ['regulatization term=' num2str(p) ', all initial weights=1']});
        saveas(h,sprintf('roc_test_iw1_scg_%ihuni_%ireg.eps', k, pp), 'epsc'); % will create FIG1, FIG2,...
        close(h)
       
        [at,bt,ct,auc_test] = perfcurve(Y(tr.testInd),test_yhat,1);
        [a,b,c,auc] = perfcurve(Y,t,1);
        
        %plot(a,b)
        %xlabel('False positive rate')
        %ylabel('True positive rate')
        %title('ROC for Classification by Logistic Regression')

        
       % [tpr,fpr,thresholds] = roc(Y,t);
        %[tpr_test,fpr_test,thresholds_test] = roc(Y(tr.testInd),test_yhat);

       %each time have to caluculate!!3*3*10*11
                    %u = (count_i-1)*(3*10*11)+(count_j-1)*(10*11)+(count_k-1)*11+count_p;
                    %u = (count_i - 1)*(3*2*2) + (count_j - 1)*(2*2) + (count_k - 1)*2 + count_p
        u = count_p;
        %matrix_sum(u,1) = m;
        %matrix(u,2) = j;
        matrix_sum(u,1) = k;
        matrix_sum(u,2) = p;
        if strcmp(tr.stop,'Minimum gradient reached.')
            matrix_sum(u,3) = 0;
        elseif strcmp(tr.stop, 'Validation stop.')
            matrix_sum(u,3) = 1;
        else
            matrix_sum(u,3) = 2;
        end
        seq{u} = tr.stop;
        %disp(matrix_sum(u,3));
        %tr.stop;
        %matrix(u,6) = net.IW{1}(1);
        %qq(1,i) = net.IW{1}(2);
        matrix_sum(u,4) = tr.best_perf;
        matrix_sum(u,5) = tr.best_vperf;
        matrix_sum(u,6) = tr.best_tperf;
        matrix_sum(u,7) = tr.best_epoch;
        matrix_sum(u,8) = tr.time(length(tr.epoch));
        matrix_sum(u,9) = perf_full;
        matrix_sum(u,10) = percError_full;
        matrix_sum(u,11) = perf_test;
        matrix_sum(u,12) = percError_test;
        matrix_sum(u,13) = auc_test;
        matrix_sum(u,14) = auc;
    end
%train_ind = tr.trainInd;
%val_ind = tr.valInd;
%test_ind = tr.testInd;
end

save('sum_iw1.mat', 'matrix_sum')
save('stop_iw1.mat', 'seq')
save('all_iw1.mat')
%%