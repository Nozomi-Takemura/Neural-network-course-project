function  oplossing =SVM_TRAINQP(X,Y,epsilon,type_nummer,C,par)

% The trainingsmodule of the Vapnik SVM

% paramters:
% X - matrix of training samples (each column is a sample)
% Y - labels of training samples
% C - Penalty factor for missclassification

% The method returns the Lagrange multipliers (or support values) and
% the bias


tekstuitvoer=0;



N=length(X(1,:));
Aeq=Y';
beq=[0];
lb=zeros(N,1);
if C==inf
   C=[];
else
   ub=C*ones(N,1);
end
f=-1*ones(N,1);

% Bepaling H:
K=full(kernel2(X,[],type_nummer,par));
H=diag(Y)*K*diag(Y);

% subplot(1,2,1),contourf(K)
% colorbar
% title('Strucuur in K');

options=optimset('MaxIter',1000,'LargeScale','off');
alpha=quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options); 
%disp((sort(alpha))');

figure,plot(sort(alpha)),ylabel('\alpha')


% bepalen van de bias
%********************

svi=find(abs(alpha)>epsilon);
aantalsup=length(svi);

if tekstuitvoer
   fprintf('Support Vectors : %d (%3.1f%%)\n\n',aantalsup,100*aantalsup/N);
end

bias=bias_B(alpha,K,X,Y,epsilon);
clear H K;
oplossing=cell(1,2);
oplossing{1,1}=alpha;
oplossing{1,2}=bias;
