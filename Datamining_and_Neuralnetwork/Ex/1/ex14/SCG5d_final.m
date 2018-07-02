%%
cd 'H:/Documents/package/'
%cd /home/nozomi/package/
addpath(pwd)
clear;
clf;
rng(0408);
sec = cell(5,1);
m = 20;
%%
j = 0.2;
count_k = 0;
matrix_sum = zeros(5,9);
cd C:\Temp\1221ex14\SCGdim5\obs20;
sec = cell(5,1);
 %m = 10; k = 1
tic;
%for m = [30,150,450,1200]
%Data Generation
    d1=linspace(-5,5,m);
    d2=linspace(-5,5,m);
    d3=linspace(-5,5,m);
    d4=linspace(-5,5,m);
    d5=linspace(-5,5,m);
    [gen1,gen2,gen3,gen4,gen5]=ndgrid(d1,d2,d3,d4,d5);

    tr_input = ([gen1(:), gen2(:), gen3(:), gen4(:), gen5(:) ].');
    intermed=tr_input.*tr_input;
    tr_output=sinc(sqrt((intermed(1,:)+intermed(2,:)+intermed(3,:)+intermed(4,:)+intermed(5,:))));
    tr_output_noise = tr_output  + j*randn(size(tr_output));
  
    %for k = [5,20,60,100,200]%k=3
    for k = [5,20,60,100]%k=3
        count_k = count_k + 1;
        net=fitnet(k,'trainscg');
        net.divideFcn='dividerand';
        input=[tr_input];
        output=[tr_output_noise];


        net.performParam.regularization = 0.000001 ;
        [net,tr]=train(net,input,output);
        test_input = tr_input(:,tr.testInd);
        test_intermed=test_input.*test_input;
        test_output=sinc(sqrt((test_intermed(1,:)+test_intermed(2,:)+test_intermed(3,:)+test_intermed(4,:)+test_intermed(5,:))));
        % 
        test_yhat = net(test_input);
        test_d5 = tr_output(tr.testInd);
        %

        h = figure; set(h, 'Visible', 'off'); 
        plot3(test_input(1,:),test_input(2,:),test_output,'b*');
        title('5 dimensional sinc function: 1st and 2nd dimension')
        xlabel('Input value of 1st dimension')
        ylabel('Input value of 2nd dimension')
        zlabel('Output')
        hold on;
        plot3(test_input(1,:),test_input(2,:),test_yhat,'ro');
        legend({'sinc(x)','Approximated function'});
        hold off;
        saveas(h,sprintf('SCG_data%i_noise02_huni_%i_reg000001_yhat_vs_sinc_1stvs2nd.png',m, k)); % will create FIG1, FIG2,...
        %hold off;
        %close;

        %figure;
        g = figure; set(g, 'Visible', 'off'); 
        plot3(test_input(3,:),test_input(4,:),test_output,'b*');
        title('5 dimensional sinc function: 3rd and 4th dimenstion')
        xlabel('Input - 3rd dimension')
        ylabel('Input - 4th dimension')
        zlabel('Output')
        hold on;
        plot3(test_input(3,:),test_input(4,:),test_yhat,'ro');
        legend({'sinc(x)','Approximated function'});
        hold off;
        saveas(g,sprintf('SCG_data%i_noise02_huni_%i_reg000001_yhat_vs_sinc_3rdvs4th.png',m, k)); % will create FIG1, FIG2,...
        %close;
        
        %figure;
        t = figure; set(t, 'Visible', 'off');
        plot(test_input(5,:),test_output,'b*');
        title('5 dimensional sinc function - 5th')
        xlabel('Input - 5th dimension')
        ylabel('Output')
        hold on;
        plot(test_input(5,:),test_yhat,'ro');
        legend({'sine(x)','Approximated function'});
        hold off;       
        saveas(t,sprintf('SCG_data%i_noise02_huni_%i_reg000001_yhat_vs_sinc_5th_vs_res.png',m, k)); % will create FIG1, FIG2,...
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
        matrix_sum(u,9) = perform(net,test_d5,test_yhat);
        %train_ind = tr.trainInd;
        %val_ind = tr.valInd;
        %test_ind = tr.testInd;
        t = toc;
        save(sprintf('ndata%i_nneurons%i_summary.mat',m, k), 'matrix_sum')
        save(sprintf('ndata%i_nneurons%i_time.mat',m, k), 't')
        save(sprintf('ndata%i_nneurons%i_storule.mat',m, k), 'seq')
    end
    %end


%save(sprintf('ndata%i_nneurons%i_all.mat',m, k))