%%
clear;
clf;
rng(0408);
j = 0.2;
count_k = 0;
matrix_sum = [];
cd C:\Temp\fig14\LM_dim2;
sec = cell(4*4,1);
tic; %m = 10


view(net)


for k = [1,5,10,30,100]%k=3
    count_k = 0;
    net = patternnet(k);
    if 
        net = configure(net,X,Y);
        iniw = net.IW;
        iniLW = net.LW;
        iniw{1} = ones(k,30)
        iniLW{2} = ones(1,k)
    for p=[0, 0.1, 0.01, 0.001, 0.0001, 0.00001]%p=0.001
        count_p = count_p + 1;
        pp = round(p*100000);
        net.performParam.regularization = p;
        [net tr] = train(net,X,Y);
        inp_tyh = X(:,tr.testInd);
        
         %tf = isequal(ppp,inp_tyh)
        test_yhat=net(inp_tyh);
        t = net(X);
        perf = perform(net,Y(tr.testInd),test_yhat);
        classes = vec2ind(test_yhat);
        
        
        g = figure; set(g, 'Visible', 'off');
        simpleclusterOutputs = sim(net,X);
        plotroc(Y,simpleclusterOutputs); 
        title(['ROC curve (full data): Levenberg-Marquardt, #neurons =' num2str(k) ', regulatization term=' num2str(p)]);
        saveas(g,sprintf('roc_fulldata_LM_%ihuni_%ireg.eps', k, pp), 'epsc'); % will create FIG1, FIG2,...
        close(g)
        
        h = figure; set(g, 'Visible', 'off');
        simpleclusterOutputs_2 = sim(net,inp_tyh);
        plotroc(Y(tr.testInd),simpleclusterOutputs_2)
        title(['ROC curve (test data): Levenberg-Marquardt, #neurons =' num2str(k) ', regulatization term=' num2str(p)]);
        saveas(h,sprintf('roc_test_LM_%ihuni_%ireg.eps', k, pp), 'epsc'); % will create FIG1, FIG2,...
        close(h)
        %title(['Training setting: resilient backpropagation with ' 'the number of traing data=' num2str(i) ', amount of noise=' num2str(j) ', number of hidden units=' num2str(k) ' and regulatization term=' num2str(p)]);
        %saveas(h,sprintf('QN_data%i_noise%i_%ihuni_%ireg.eps',i, jj, k, pp)); % will create FIG1, FIG2,...
        %saveas(h, sprintf('LM_data%i_noise%i_%ihuni_%ireg.eps',i, jj, k, pp),'epsc');
        %saveas(h,sprintf('CG_data%i_noise%i_%ihuni_%ireg.eps',i, jj, k, pp)); % will create FIG1, FIG2,...
        %saveas(h,sprintf('BP_data%i_noise%i_%ihuni_%ireg.eps',i, jj, k, pp)); % will create FIG1, FIG2,...

        
       % [tpr,fpr,thresholds] = roc(Y,t);
        %[tpr_test,fpr_test,thresholds_test] = roc(Y(tr.testInd),test_yhat);

       %each time have to caluculate!!3*3*10*11
                    %u = (count_i-1)*(3*10*11)+(count_j-1)*(10*11)+(count_k-1)*11+count_p;
                    %u = (count_i - 1)*(3*2*2) + (count_j - 1)*(2*2) + (count_k - 1)*2 + count_p
        u = count_p;
        %matrix_sum(u,1) = m;
        %matrix(u,2) = j;
        matrix_sum(u,1) = k;
        matrix(u,2) = p;
        if strcmp(tr.stop,'Minimum gradient reached.')
            matrix_sum(u,3) = 0;
        elseif strcmp(tr.stop, 'Validation stop.')
            matrix_sum(u,3) = 1;
        else
            matrix_sum(u,3) = 2;
        end
        seq{u} = tr.stop;
        disp(matrix_sum(u,3));
        tr.stop;
        %matrix(u,6) = net.IW{1}(1);
        %qq(1,i) = net.IW{1}(2);
        matrix_sum(u,4) = tr.best_perf;
        matrix_sum(u,5) = tr.best_vperf;
        matrix_sum(u,6) = tr.best_tperf;
        matrix_sum(u,7) = tr.best_epoch;
        matrix_sum(u,8) = tr.time(length(tr.epoch));
%train_ind = tr.trainInd;
%val_ind = tr.valInd;
%test_ind = tr.testInd;
end

t = toc;
save('C:\Temp\fig14\LM_dim2\stat.mat', 'matrix_sum')
save('C:\Temp\fig14\LM_dim2\stat.mat', 't')
save('C:\Temp\fig14\LM_dim2\stat.mat', 'seq')
save('C:\Temp\fig14\LM_dim2\all.mat')
%%
    
    
    
    
    
    
    
    