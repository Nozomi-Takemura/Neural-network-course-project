%%
%cd 'H:/Documents/package/'
cd /home/nozomi/package/
addpath(pwd)
clear;
clf;
rng(0408);
sec = cell(4,1);
m = 210;
%%
j = 0.2;
count_k = 0;
matrix_sum = zeros(4,9);
%cd C:\Temp\1221ex14\QNdim2\;
cd /home/nozomi/Documents/2017courses/DMNN/Ex/1/fig14/2d/BP2d
%sec = cell(5,1);
 %m = 10; k = 1
%tic;
%for m = [30,150,450,1200]
%Data Generation
    d1=linspace(-5,5,m);
    d2=linspace(-5,5,m);
    [gen1,gen2]=ndgrid(d1,d2);
    tr_input = ([gen1(:), gen2(:)].');
    intermed=tr_input.*tr_input;
    tr_output=sinc(sqrt((intermed(1,:)+intermed(2,:))))
    tr_output_noise = tr_output  + j*randn(size(tr_output));
  
    %for k = [5,20,60,100,200]%k=3
    %for k = [5,20,60,100]%k=3
    for k = [5,20,60,100]%k=
        count_k = count_k + 1;
        net=fitnet(k,'trainrp');
        net.divideFcn='dividerand';
        input=[tr_input];
        output=[tr_output_noise];


        net.performParam.regularization = 0.000001 ;
        [net,tr]=train(net,input,output);
        test_input = tr_input(:,tr.testInd);
        test_intermed=test_input.*test_input;
        test_output=sinc(sqrt((test_intermed(1,:)+test_intermed(2,:))));
        % 
        test_yhat = net(test_input);
        test_d2 = tr_output(tr.testInd);
        %
        %figure;
        g = figure; set(g, 'Visible', 'off'); 
        plot3(test_input(1,:),test_input(2,:),test_output,'b*');
        %tit1=title('Plot of $sinc(\sqrt{x_{1}^{2}+x_{2}^{2}} \text{for} −5 ≤ x_{1}, x_{2} ≤ 5$.')
        %set(tit1,'Interpreter','latex');
        xlabel('Input - 1stt dimension')
        ylabel('Input - 2nd dimension')
        zlabel('Output')
        %set(z1,'Interpreter','latex');
        hold on;
        plot3(test_input(1,:),test_input(2,:),test_yhat,'ro');
        leg1 = legend({'$sinc(\sqrt{x_{1}^{2}+x_{2}^{2}})$','Approximated function'}, 'Location', 'Best');
        set(leg1,'Interpreter','latex');
        hold off;
        saveas(g,sprintf('BP_data%i_noise02_huni_%i_reg000001_yhat_vs_sinc_2d.png',m, k)); % will create FIG1, FIG2,...
        %close;
        
        
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
        matrix_sum(u,6) = tr.best_tperf;
        matrix_sum(u,7) = tr.best_epoch;
        matrix_sum(u,8) = tr.time(length(tr.epoch));
        matrix_sum(u,9) = perform(net,test_d2,test_yhat);
        %train_ind = tr.trainInd;
        %val_ind = tr.valInd;
        %test_ind = tr.testInd;
        
        save(sprintf('BP_ndata%i_nneurons%i_2d_summary.mat',m, k), 'matrix_sum')
        %save(sprintf('QN_ndata%i_nneurons%i_2d_time.mat',m, k), 't')
        save(sprintf('BP_ndata%i_nneurons%i_2d_storule.mat',m, k), 'seq')
    end
    %end


%save(sprintf('ndata%i_nneurons%i_2d_all.mat',m, k))