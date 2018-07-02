function b1=bias_B(alpha,K,X,Y,epsilon)
[m n]=size(X);
svi=find(abs(alpha)>epsilon);
nbofsup=length(svi);
X=X(:,svi);
Y=Y(svi);
K=K(svi,svi);
alpha=alpha(svi);
AA=[Y';K];

AA=sortrows(AA',1)';
q=length(find(AA(1,:)<0));
AA=AA(2:nbofsup+1,:);
b1=-(max((alpha.*Y)'*AA(:,1:q))+min((alpha.*Y)'*AA(:,q+1:nbofsup)))/2;




%b2=-(sum( Y'.* alpha'.* (ones(1,nbofsup)*K)) )/nbofsup
%b3=-sum([(alpha.*Y)'*AA(:,1:q) (alpha.*Y)'*AA(:,q+1:nbofsup)])/nbofsup